# Font Identifier Project

Nov 8-14, 2023

Visual classifier to identify the font used in an image.

You can try it on HuggingFace at [gaborcselle/font-identifier](https://huggingface.co/gaborcselle/font-identifier).

The dataset used is on HuggingFace at [datasets/gaborcselle/font-examples](https://huggingface.co/datasets/gaborcselle/font-examples).

# Key learnings

1. A ResNet18 is shocklingly good at identifying fonts, I got 95%+ accuracy on the test set of 2.4k font images.
2. The model gets confused where I'd get confused too: Helvetica vs. Arial, Trebuchet vs. Verdana - see the confusion matrix on the [HuggingFace page](https://huggingface.co/gaborcselle/font-identifier)
3. For this to be practically useful, I'd have to train it on a much larger set of fonts, and maybe go to a ResNet34 or ResNet50 - perhaps a project for another day.

# Journal

I built this project in 1 day, with a minute-by-minute journal:
* [On Pebble.social](https://pebble.social/@gabor/111376050835874755)
* [On Threads.net](https://www.threads.net/@gaborcselle/post/CzZJpJCpxTz)
* [On Twitter](https://twitter.com/gabor/status/1722300841691103467)


# Pieces

1. Generate sample images (note this will work only on Mac): [gen_sample_data.py](gen_sample_data.py)
2. Arrange test images into test and train: [arrange_train_test_images.py](arrange_train_test_images.py)
3. Train a ResNet18 on the data: [train_font_identifier.py](train_font_identifier.py)
4. Scratch that, actually use HuggingFace to finetune a ResNet18, so I can upload the result to their site: [mf_model_train.py](mf_model_train.py)
5. Upload the dataset to HuggingFace: [dataset_upload.py](dataset_upload.py)
6. Generate the confusion matrix: [hf_confusion_matrix.py](hf_confusion_matrix.py)


