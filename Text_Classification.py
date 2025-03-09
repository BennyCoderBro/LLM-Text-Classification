import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import re
import nltk
from nltk.corpus import wordnet, stopwords
from torch.nn.functional import softmax
from collections import Counter

# Ensure NLTK resources are downloaded
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# Define category keywords for distinction
category_keywords = {
    "Models of Manipulating Qubits for Computation": "Quantum gates, Qubit control, Quantum circuits, Quantum algorithms",
    "Methods of Building Qubits": "Superconducting qubits, Trapped ions, Quantum dots, Semiconductor qubits",
    "Address Obstacles to Quantum Computation": "Quantum error correction, Decoherence, Noise mitigation",
    "Applications of Quantum Computing": "Quantum simulation, Cryptography, Optimization problems"
}

# Text cleaning function
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Keep only letters, numbers, and spaces
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# Synonym replacement for text augmentation (only applies to nouns)
def synonym_replacement(text):
    words = text.split()
    new_words = []
    for word in words:
        synonyms = wordnet.synsets(word, pos=wordnet.NOUN)  # Limit to noun replacements
        if synonyms:
            new_word = synonyms[0].lemmas()[0].name()
            if new_word != word:
                new_words.append(new_word)
            else:
                new_words.append(word)
        else:
            new_words.append(word)
    return " ".join(new_words)

# Custom dataset class
class QuantumDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx],
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_length,
                                  return_tensors='pt')

        item = {key: val.squeeze(0) for key, val in encoding.items()}

        if self.labels is not None and idx < len(self.labels):
            item['labels'] = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        return item

# Load and preprocess training data
train_df = pd.read_csv("training.csv", encoding_errors="ignore").dropna(subset=['category'])

# Ensure only defined categories are included
print("Original category distribution:\n", train_df["category"].value_counts())

train_df = train_df[train_df['category'].isin(category_keywords.keys())]

# Verify filtered categories
print("Filtered category distribution:\n", train_df["category"].value_counts())

# Clean text fields
train_df["title"] = train_df["title"].apply(clean_text)
train_df["abstract"] = train_df["abstract"].apply(clean_text)

# Append category-specific keywords to improve classification
train_df["combined_text"] = train_df.apply(
    lambda row: row["title"] + " " + row["abstract"] + " " + category_keywords[row["category"]], axis=1
)

# Apply text augmentation
train_df["augmented_text_synonym"] = train_df["combined_text"].apply(synonym_replacement)

# Process category labels
label_map = {category: idx for idx, category in enumerate(category_keywords.keys())}
train_df["label"] = train_df['category'].map(label_map).astype(int)

train_texts = train_df["combined_text"].tolist() + train_df["augmented_text_synonym"].tolist()
train_labels = train_df["label"].tolist() * 2  # Duplicate labels for augmented data

# Check final label distribution
print("Final label distribution:", Counter(train_labels))

# Load MiniLM tokenizer and model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(category_keywords))

# Create training dataset
train_dataset = QuantumDataset(train_texts, train_labels, tokenizer)

# Ensure training dataset size matches labels
assert len(train_dataset) == len(train_labels), "Mismatch between dataset size and labels"
print(f"Total training samples: {len(train_dataset)}")

# Compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = softmax(torch.tensor(logits), dim=1).numpy()
    predictions = np.argmax(probs, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

# Training parameters
training_args = TrainingArguments(
    warmup_steps=500,
    output_dir="./results",
    evaluation_strategy="no",  # No evaluation since we're using the full dataset
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=6,
    learning_rate=2e-5,
    weight_decay=0.05,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2
)

# Custom callback to log training loss
class LossLoggerCallback(TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])

loss_logger = LossLoggerCallback()

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics,
    callbacks=[loss_logger]
)

# Train the model
trainer.train()

# Plot training loss over steps
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(loss_logger.losses) + 1), loss_logger.losses, marker='o', linestyle='-')
plt.xlabel("Steps")
plt.ylabel("Training Loss")
plt.title("Training Loss per Step")
plt.grid()
plt.show()

# **Apply Model to Test Set**
# Load test data
test_df = pd.read_csv("testing.csv", encoding_errors="ignore").dropna(subset=['title', 'abstract'])

# Clean text fields
test_df["title"] = test_df["title"].apply(clean_text)
test_df["abstract"] = test_df["abstract"].apply(clean_text)

# Create input text for the model
test_df["combined_text"] = test_df["title"] + " " + test_df["abstract"]

# Convert test data into dataset format
test_texts = test_df["combined_text"].tolist()
test_dataset = QuantumDataset(test_texts, tokenizer=tokenizer)

# Make predictions
predictions = trainer.predict(test_dataset)
logits = predictions.predictions
probs = softmax(torch.tensor(logits), dim=1).numpy()
predicted_labels = np.argmax(probs, axis=1)

# Assign predicted categories to test set
test_df["category"] = [list(category_keywords.keys())[label] for label in predicted_labels]

# Save results to CSV
test_df.to_csv("Zhihao_Yang_testing.csv", index=False)

# Print sample results
print(test_df[["title", "abstract", "category"]].head())
