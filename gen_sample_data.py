# Generate sample data with 800x400 images of fonts in /System/Library/Fonts
# 50 images per font, 1 font per image


import os
from PIL import Image, ImageDraw, ImageFont
import nltk
from nltk.corpus import brown
import random

# Download the necessary data from nltk
nltk.download('brown')

# Sample text for prose and code
prose_text = " ".join(brown.words(categories='news')[:50]) # First 50 words from news category

font_dir = '/System/Library/Fonts/'
output_dir = './font_images'
os.makedirs(output_dir, exist_ok=True)

all_brown_words = sorted(set(brown.words(categories='news')))

def wrap_text(text, line_length=10):
    """
    Wraps the provided text every 'line_length' words.
    """
    words = text.split()
    return "\n".join([" ".join(words[i:i+line_length]) for i in range(0, len(words), line_length)])

def random_prose_text(words, num_words=200):  # Sample random words
    random_words = " ".join(random.sample(words, num_words))
    return wrap_text(random_words)

def random_code_text(base_code, num_lines=15):  # Increase number of lines
    lines = base_code.split("\n")
    return "\n".join(random.sample(lines, min(num_lines, len(lines))))

for font_file in os.listdir(font_dir):
    if font_file.endswith('.ttf'):
        font_path = os.path.join(font_dir, font_file)
        font_name = font_file.split('.')[0]
        print(font_name)

        j = 0
        for i in range(50):  # Generate 50 images per font
            prose_sample = random_prose_text(all_brown_words)

            for text in [prose_sample]:
                img = Image.new('RGB', (800, 400), color="white")  # Canvas size
                draw = ImageDraw.Draw(img)
                font_size = random.choice(range(32, 128))  # Increased minimum font size
                font = ImageFont.truetype(font_path, font_size)

                # Random offsets, but ensuring that text isn't too far off the canvas
                offset_x = random.randint(-20, 10)
                offset_y = random.randint(-20, 10)
                draw.text((offset_x, offset_y), text, fill="black", font=font)
                
                j += 1
                output_file = os.path.join(output_dir, f"{font_name}_{j}.png")
                img.save(output_file)
