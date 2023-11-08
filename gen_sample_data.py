# Generate sample data with 800x400 images of fonts in /System/Library/Fonts
# 50 images per font, 1 font per image

import os
from PIL import Image, ImageDraw, ImageFont
import nltk
from nltk.corpus import brown
import random
from consts import FONT_ALLOWLIST, IMAGES_PER_FONT, GEN_IMAGES_DIR, FONT_FILE_DIRS

# Download the necessary data from nltk
nltk.download('brown')

os.makedirs(GEN_IMAGES_DIR, exist_ok=True)

all_brown_words = sorted(set(brown.words(categories='news')))

def wrap_text(text, line_length=10):
    """Wraps the provided text every 'line_length' words."""
    words = text.split()
    return "\n".join([" ".join(words[i:i+line_length]) for i in range(0, len(words), line_length)])

def random_prose_text(words, num_words=200):
    """Returns a random selection of 'num_words' words from the provided list of words."""
    random_words = " ".join(random.sample(words, num_words))
    return wrap_text(random_words)

def random_code_text(base_code, num_lines=15):
    """Returns a random selection of 'num_lines' lines from the provided code."""
    lines = base_code.split("\n")
    return "\n".join(random.sample(lines, min(num_lines, len(lines))))

def main():
    for font_dir in FONT_FILE_DIRS:
        for font_file in os.listdir(font_dir):
            if font_file.endswith('.ttf') or font_file.endswith('.ttc'):
                font_path = os.path.join(font_dir, font_file)
                font_name = font_file.split('.')[0]
                if font_name not in FONT_ALLOWLIST:
                    continue
                # Output the font name so we can see the progress
                print(font_path, font_name)

                if font_file.endswith('.ttc'):
                    # ttc fonts have multiple fonts in one file, so we need to specify which one we want
                    font = ImageFont.truetype(font_path, random.choice(range(32, 128)), index=0)
                else:
                    # ttf fonts have only one font in the file
                    font_size = random.choice(range(32, 128))  # Increased minimum font size
                    font = ImageFont.truetype(font_path, font_size)

                # Counter for the image filename
                j = 0
                for i in range(IMAGES_PER_FONT):  # Generate 50 images per font - reduced to 10 for now to make things faster
                    prose_sample = random_prose_text(all_brown_words)

                    for text in [prose_sample]:
                        img = Image.new('RGB', (800, 400), color="white")  # Canvas size
                        draw = ImageDraw.Draw(img)

                        # Random offsets, but ensuring that text isn't too far off the canvas
                        offset_x = random.randint(-20, 10)
                        offset_y = random.randint(-20, 10)
                        draw.text((offset_x, offset_y), text, fill="black", font=font)

                        j += 1
                        output_file = os.path.join(GEN_IMAGES_DIR, f"{font_name}_{j}.png")
                        img.save(output_file)

if __name__ == '__main__':
    main()