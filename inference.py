import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_ADAPTER_PATH = "./qwen_latexocr_mathwriting_sft"
SYSTEM_PROMPT = "Convert the handwritten mathematical formula into valid LaTeX. Return only LaTeX."


def get_torch_dtype():
    return torch.float16 if torch.cuda.is_available() else torch.float32


def load_base_model(model_name=MODEL_NAME):
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=get_torch_dtype(),
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    return processor, model


def load_lora_model(adapter_path=DEFAULT_ADAPTER_PATH, model_name=MODEL_NAME):
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"Adapter folder not found: {adapter_path}")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    base_model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=get_torch_dtype(),
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return processor, model


def _move_inputs_to_model_device(inputs, model):
    if not torch.cuda.is_available():
        return inputs

    try:
        target_device = next(model.parameters()).device
    except StopIteration:
        return inputs

    moved = {}
    for k, v in inputs.items():
        if hasattr(v, "to"):
            moved[k] = v.to(target_device)
        else:
            moved[k] = v
    return moved


def build_messages(image, one_shot_example=None):
    if one_shot_example is None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": SYSTEM_PROMPT},
                ],
            }
        ]
        images = [image]
    else:
        ex_image, ex_answer = one_shot_example
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": ex_image},
                    {"type": "text", "text": SYSTEM_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ex_answer},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": SYSTEM_PROMPT},
                ],
            },
        ]
        images = [ex_image, image]

    return messages, images


def generate_latex(processor, model, image, one_shot_example=None, max_new_tokens=128):
    messages, images = build_messages(image, one_shot_example=one_shot_example)

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[prompt],
        images=images,
        return_tensors="pt",
    )

    inputs = _move_inputs_to_model_device(inputs, model)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    prompt_len = inputs["input_ids"].shape[1]
    output_text = processor.batch_decode(
        generated_ids[:, prompt_len:],
        skip_special_tokens=True,
    )[0].strip()

    return output_text


if __name__ == "__main__":
    processor, model = load_lora_model(DEFAULT_ADAPTER_PATH)

    image_path = "sample_formula.jpg"
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}. "
            f"Put a test image рядом с файлом inference.py или укажи правильный путь."
        )

    img = Image.open(image_path).convert("RGB")
    pred = generate_latex(processor, model, img)
    print("Predicted LaTeX:")
    print(pred)