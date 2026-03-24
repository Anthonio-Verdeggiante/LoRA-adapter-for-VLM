import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_PATH = "./qwen_latexocr_mathwriting_sft"
PROMPT = "Convert the handwritten mathematical formula into valid LaTeX. Return only LaTeX."

def load_model():
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    base_model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    return processor, model

def predict(image_path):
    processor, model = load_model()
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt"
    )

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False
        )

    answer = processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0].strip()

    return answer

if __name__ == "__main__":
    image_path = "./Examples/1.png"
    pred = predict(image_path)
    print("Predicted LaTeX:")
    print(pred)