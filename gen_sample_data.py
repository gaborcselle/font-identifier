# Generate sample data with 800x400 images of fonts in /System/Library/Fonts
# 50 images per font, 1 font per image

import os
from PIL import Image, ImageDraw, ImageFont
import nltk
from nltk.corpus import brown
import random
from consts import FONT_ALLOWLIST, IMAGES_PER_FONT, GEN_IMAGES_DIR, FONT_FILE_DIRS, GOOGLE_FONTS_DIR 

# Download the necessary data from nltk
nltk.download('inaugural')

os.makedirs(GEN_IMAGES_DIR, exist_ok=True)

def wrap_text(text, line_length=4):
    """Wraps the provided text every 'line_length' words."""
    words = text.split()
    return "\n".join([" ".join(words[i:i+line_length]) for i in range(0, len(words), line_length)])

def random_prose_text(line_length=4):
    """Returns a random snippet from the Gutenberg corpus."""
    corpus = nltk.corpus.inaugural.raw()
    start = random.randint(0, len(corpus) - 800)
    end = start + 800
    return wrap_text(corpus[start:end], line_length=line_length)

def main():
    # Collect all allowed font files
    font_files = []
    # all of the Google fonts are allowed, no matter what
    for font_file in os.listdir(GOOGLE_FONTS_DIR):
        if font_file.endswith('.ttf') or font_file.endswith('.ttc'):
            font_path = os.path.join(GOOGLE_FONTS_DIR, font_file)
            font_name = font_file.split('.')[0]
            font_files.append((font_path, font_name))

    # for the system font directories, use the FONT_ALLOWLIST
    for font_dir in FONT_FILE_DIRS:
        for font_file in os.listdir(font_dir):
            if font_file.endswith('.ttf') or font_file.endswith('.ttc'):
                font_path = os.path.join(font_dir, font_file)
                font_name = font_file.split('.')[0]
                if font_name in FONT_ALLOWLIST:
                    font_files.append((font_path, font_name))

    # Generate images for each font file
    for font_path, font_name in font_files:
        # Output the font name so we can see the progress
        print(font_path, font_name)


        # Counter for the image filename
        j = 0
        for i in range(IMAGES_PER_FONT):  # Generate 50 images per font - reduced to 10 for now to make things faster
            # Random font size
            font_size = random.choice(range(18, 72))

            if font_path.endswith('.ttc'):
                # ttc fonts have multiple fonts in one file, so we need to specify which one we want
                font = ImageFont.truetype(font_path, font_size, index=0)
            else:
                # ttf fonts have only one font in the file
                font = ImageFont.truetype(font_path, font_size)

            # Determine the number of words that will fit on a line
            font_avg_char_width = font.getbbox('x')[2]
            words_per_line = int(800 / (font_avg_char_width*5))
            prose_sample = random_prose_text(line_length=words_per_line)

            for text in [prose_sample]:
                img = Image.new('RGB', (800, 400), color="white")  # Canvas size
                draw = ImageDraw.Draw(img)

                # Random offsets, but ensuring that text isn't too far off the canvas
                offset_x = random.randint(-20, 10)
                offset_y = random.randint(-20, 10)

                # vary the line height
                line_height = random.uniform(0, 1.25) * font_size
                draw.text((offset_x, offset_y), text, fill="black", font=font, spacing=line_height)

                j += 1
                output_file = os.path.join(GEN_IMAGES_DIR, f"{font_name}_{j}.png")
                img.save(output_file)

if __name__ == '__main__':
    main()