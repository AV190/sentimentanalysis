# Load Twitter dataset
#twitter_dataset = load_dataset('csv', data_files={'train': 'C:/Users/91938/Downloads/MINI/oppotraindata.csv', 'test': 'C:/Users/91938/Downloads/MINI/oppotestdata.csv'})

# Load Amazon dataset
#amazon_dataset = load_dataset('csv', data_files={'train': 'C:/Users/91938/Downloads/MINI/amazontraindata.csv', 'test': 'C:/Users/91938/Downloads/MINI/amazontestdata.csv'})

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, concatenate_datasets, DatasetDict
import torch

# Function to ensure labels are integers and within the range of [0, 1, 2] for negative, neutral, positive respectively
def ensure_labels_integers(example):
    # This function is redundant if labels are already integers,
    # but you can keep it to assert the type or for future proofing.
    assert isinstance(example['label'], int), "Label must be an integer"
    assert example['label'] in [0, 1, 2], "Label must be 0 (negative), 1 (neutral), or 2 (positive)"
    return example

# Load Twitter dataset
twitter_dataset = load_dataset('csv', data_files={'train': 'C:/Users/91938/Downloads/MINI/oppotraindata.csv', 'test': 'C:/Users/91938/Downloads/MINI/oppotestdata.csv'})

# Load Amazon dataset
amazon_dataset = load_dataset('csv', data_files={'train': 'C:/Users/91938/Downloads/MINI/amazontraindata.csv', 'test': 'C:/Users/91938/Downloads/MINI/amazontestdata.csv'})

# Ensure both datasets have the same column names
twitter_dataset = twitter_dataset.map(lambda x: {'text': x['text'], 'label': x['label']})
amazon_dataset = amazon_dataset.map(lambda x: {'text': x['text'], 'label': x['label']})

# Ensure the labels are integers and within the correct range
twitter_dataset = twitter_dataset.map(ensure_labels_integers)
amazon_dataset = amazon_dataset.map(ensure_labels_integers)

# Function to get a smaller portion of the dataset
def get_smaller_dataset(dataset, size, seed=42):
    actual_size = min(size, len(dataset))
    return dataset.shuffle(seed=seed).select(range(actual_size))

# Use a smaller portion of the datasets
twitter_dataset_small = get_smaller_dataset(twitter_dataset['train'], 1000)
amazon_dataset_small = get_smaller_dataset(amazon_dataset['train'], 1000)
twitter_test_small = get_smaller_dataset(twitter_dataset['test'], 200)
amazon_test_small = get_smaller_dataset(amazon_dataset['test'], 200)

# Concatenate training datasets
combined_train_dataset = concatenate_datasets([twitter_dataset_small, amazon_dataset_small])
# Combine the test datasets similarly
combined_test_dataset = concatenate_datasets([twitter_test_small, amazon_test_small])

# Combine into a DatasetDict
combined_dataset = DatasetDict({
    'train': combined_train_dataset,
    'test': combined_test_dataset
})

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = combined_dataset.map(tokenize_function, batched=True)

# Convert to torch Dataset
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Split the tokenized datasets
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # 3 classes: negative, neutral, positive

# Check if CUDA is available and move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Data collator (manually defining if necessary)
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",  # Use `eval_strategy` instead of `evaluation_strategy`
    learning_rate=2e-5,
    per_device_train_batch_size=16 if torch.cuda.is_available() else 4,  # Adjust batch size if using CPU
    per_device_eval_batch_size=16 if torch.cuda.is_available() else 4,   # Adjust batch size if using CPU
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),  # Enable mixed precision training only if using GPU
)

# Trainer for fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Use the tokenized train dataset
    eval_dataset=eval_dataset,    # Use the tokenized test dataset
    data_collator=data_collator   # Use the data collator
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('C:/Users/91938/Downloads/MINI/bertmodelnew')
tokenizer.save_pretrained('C:/Users/91938/Downloads/MINI/berttokennew')
