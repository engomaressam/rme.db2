from PIL import Image
import torch
from io import BytesIO
import base64
from diffusers import StableDiffusionImg2ImgPipeline
import os

# Disable SSL verification for Hugging Face downloads
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

# Function to convert an image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# Load the Stable Diffusion model
def load_model():
    model = StableDiffusionImg2ImgPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16
    )
    model = model.to("cuda")
    return model

# Function to generate the image with the added text
def generate_image_with_text(base_image_path, text_prompt, strength=0.75, guidance_scale=7.5):
    # Convert image to base64 (optional step)
    image_base64 = image_to_base64(base_image_path)
    
    # Load the base image and convert to RGB
    init_image = Image.open(base_image_path).convert("RGB")
    init_image = init_image.resize((512, 512))

    model = load_model()
    prompt = f"{text_prompt}"
    with torch.autocast("cuda"):
        result = model(prompt=prompt, image=init_image, strength=strength, guidance_scale=guidance_scale)
    generated_image = result.images[0]
    
    # Save the generated image
    generated_image.save("generated_with_text.png")
    print("Image generated and saved as 'generated_with_text.png'")

# Path to your image
base_image_path = "D:/OEssam/rme.logo.jpg"
text_prompt = "Only Data, 'some' Science"

# Generate the image with the added text
generate_image_with_text(base_image_path, text_prompt)
