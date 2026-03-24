from PIL import Image
from inference import load_base_model, generate_latex

image_path = "./Examples/1.png"

processor, model = load_base_model()
image = Image.open(image_path).convert("RGB")

pred = generate_latex(processor, model, image)

print("Base model prediction:")
print(pred)