from datasets import load_dataset

# example 1: local folder
dataset = load_dataset("imagefolder", data_dir="./font_examples")

dataset.push_to_hub("font-examples")
