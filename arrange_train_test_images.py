# moves the font images into train and test folders
# TODO(gabor): maybe we should copy these instead, so we don't have to regenerate the images every times?
import os
import shutil
import random

source_dir = './font_images'
organized_dir = './train_test_images'
train_dir = os.path.join(organized_dir, 'train')
test_dir = os.path.join(organized_dir, 'test')

# create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# make a list of all the font names
fonts = [f.split('_')[0] for f in os.listdir(source_dir) if f.endswith('.png')]
fonts = list(set(fonts))  # getting unique font names

for font in fonts:
    font_train_dir = os.path.join(train_dir, font)
    font_test_dir = os.path.join(test_dir, font)
    os.makedirs(font_train_dir, exist_ok=True)
    os.makedirs(font_test_dir, exist_ok=True)

    font_files = [f for f in os.listdir(source_dir) if f.startswith(font)]
    random.shuffle(font_files)

    train_files = font_files[:int(0.8 * len(font_files))]
    test_files = font_files[int(0.8 * len(font_files)):]

    # Move training files
    for train_file in train_files:
        shutil.move(os.path.join(source_dir, train_file), font_train_dir)

    # Move test files
    for test_file in test_files:
        shutil.move(os.path.join(source_dir, test_file), font_test_dir)
