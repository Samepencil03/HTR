import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_sample_image(output_path="input/sample_handwritten.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create white canvas
    width, height = 800, 300
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    text_lines = [
        "Today I went to college",
        "I studied artificial intelligence",
        "The weather was very good"
    ]

    # Try to load a clean font or fallback to default
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 32)
    except IOError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
        except IOError:
            font = ImageFont.load_default()

    y_offset = 40
    for line in text_lines:
        draw.text((50, y_offset), line, fill=(10, 10, 10), font=font)
        y_offset += 75

    img.save(output_path)
    print(f"Sample test image generated at '{output_path}'")

if __name__ == "__main__":
    generate_sample_image()
