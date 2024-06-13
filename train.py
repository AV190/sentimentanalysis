from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, concatenate_datasets, DatasetDict
import torch

# Function to ensure labels are integers and within the range of [0, 1]
def fix_labels(example):
    example['label'] = int(example['label'])
    example['label'] = 0 if example['label'] < 1 else 1
    return example

# Load Twitter dataset
twitter_dataset = load_dataset('csv', data_files={'train': 'C:/Users/91938/Downloads/twittertraindata.csv', 'test': 'C:/Users/91938/Downloads/twittertestdata.csv'})

# Load Amazon dataset
amazon_dataset = load_dataset('csv', data_files={'train': 'C:/Users/91938/Downloads/amazontraindata.csv', 'test': 'C:/Users/91938/Downloads/amazontestdata.csv'})

# Ensure both datasets have the same column names
twitter_dataset = twitter_dataset.map(lambda x: {'text': x['text'], 'label': int(x['label'])})
amazon_dataset = amazon_dataset.map(lambda x: {'text': x['text'], 'label': int(x['label'])})

# Fix the labels to be within the correct range
twitter_dataset = twitter_dataset.map(fix_labels)
amazon_dataset = amazon_dataset.map(fix_labels)

# Use a smaller portion of the datasets
twitter_dataset_small = twitter_dataset['train'].shuffle(seed=42).select(range(1000))
amazon_dataset_small = amazon_dataset['train'].shuffle(seed=42).select(range(1000))
twitter_test_small = twitter_dataset['test'].shuffle(seed=42).select(range(200))
amazon_test_small = amazon_dataset['test'].shuffle(seed=42).select(range(200))

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
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Check if CUDA is available and move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
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
    eval_dataset=eval_dataset,   # Use the tokenized test dataset
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('C:/Users/91938/Downloads/MINI/bertmodel')
tokenizer.save_pretrained('C:/Users/91938/Downloads/MINI/berttoken')
