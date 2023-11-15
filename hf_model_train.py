import numpy as np
from datasets import load_dataset, Image
from transformers import AutoImageProcessor, ResNetForImageClassification, Trainer, TrainingArguments
from torchvision import transforms
from consts import TRAIN_TEST_IMAGES_DIR
from sklearn.metrics import accuracy_score

img_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), # Convert images to grayscale with 3 channels
    transforms.RandomCrop((224, 224)), # Resize images to the expected input size of the model
    transforms.ToTensor(), # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize with ImageNet stats
])

def do_transforms(examples):
    examples["pixel_values"] = [img_transforms(image.convert("RGB")) for image in examples["image"]]
    del examples["image"]
    return examples

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

# Load the dataset
# Manual is here: https://huggingface.co/docs/datasets/image_load
dataset = load_dataset("imagefolder", data_dir=TRAIN_TEST_IMAGES_DIR)

num_classes = len(set(dataset['train']['label']))
labels = dataset['train'].features['label'].names

model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18", num_labels=num_classes, ignore_mismatched_sizes=True)
image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")


print("Fonts list to identify:", labels)
from datasets import load_dataset

# Assuming TRAIN_TEST_IMAGES_DIR is correctly set
dataset = load_dataset("imagefolder", data_dir=TRAIN_TEST_IMAGES_DIR)

# Apply transforms
dataset.set_transform(do_transforms)

# Label dictionary
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

training_args = TrainingArguments(
    output_dir="font-identifier",
    overwrite_output_dir=True,
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics
)

trainer.train()

# upload to hub
trainer.push_to_hub("font-identifier")

