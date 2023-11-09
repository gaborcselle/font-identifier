---
language:
- en
tags:
- font-identification
license: mit
widget:
- text: "What font is this?"
- src: https://huggingface.co/gaborcselle/font-identify/hf_sample/ArchitectsDaughter-Regular_1.png
  example_title: Architects Daughter
- src: https://huggingface.co/gaborcselle/font-identify/hf_sample/Arial_Bold_39.png
  example_title: Arial Bold
- src: https://huggingface.co/gaborcselle/font-identify/hf_sample/Courier_28.png
  example_title: Courier
- src: https://huggingface.co/gaborcselle/font-identify/hf_sample/Helvetica_3.png
  example_title: Helvetica
- src: https://huggingface.co/gaborcselle/font-identify/hf_sample/IBMPlexSans-Regular_25.png
  example_title: IBM Plex Sans
- src: https://huggingface.co/gaborcselle/font-identify/hf_sample/Inter-Regular_43.png
  example_title: Inter
- src: https://huggingface.co/gaborcselle/font-identify/hf_sample/Lobster-Regular_25.png
  example_title: Lobster
- src: https://huggingface.co/gaborcselle/font-identify/hf_sample/Merriweather-Regular_1.png
  example_title: Merriweather
- src: https://huggingface.co/gaborcselle/font-identify/hf_sample/Poppins-Regular_22.png
  example_title: Poppins
- src: https://huggingface.co/gaborcselle/font-identify/hf_sample/RobotoMono-Regular_38.png
  example_title: Roboto Mono
- src: https://huggingface.co/gaborcselle/font-identify/hf_sample/Times_New_Roman_Bold Italic_26.png
  example_title: Times New Roman Bold Italic
- src: https://huggingface.co/gaborcselle/font-identify/hf_sample/Times_New_Roman_Italic_16.png
  example_title: Times New Roman Italic
- src: https://huggingface.co/gaborcselle/font-identify/hf_sample/TitilliumWeb-Regular_5.png
  example_title: Titillium Web
- src: https://huggingface.co/gaborcselle/font-identify/hf_sample/Trebuchet_MS_Italic_47.png
  example_title: Trebuchet MS Italic
- src: https://huggingface.co/gaborcselle/font-identify/hf_sample/Trebuchet_MS_11.png
  example_title: Trebuchet MS
- src: https://huggingface.co/gaborcselle/font-identify/hf_sample/Verdana Bold_43.png
  example_title: Verdana Bold
---

# Font Identifier Project

Tinker project, Nov 8, 2023.

Follow along:
- [On Pebble.social](https://pebble.social/@gabor/111376050835874755)
- [On Threads.net](https://www.threads.net/@gaborcselle/post/CzZJpJCpxTz)
- [On Twitter](https://twitter.com/gabor/status/1722300841691103467)

Generate sample images (note this will work only on Mac): [gen_sample_data.py](gen_sample_data.py)

Arrange test images into test and train: [arrange_train_test_images.py](arrange_train_test_images.py)

Train a ResNet18 on the data: [train_font_identifier.py](train_font_identifier.py)
