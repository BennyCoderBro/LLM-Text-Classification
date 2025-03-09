import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback, get_scheduler
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import re
import nltk
from nltk.corpus import wordnet, stopwords
from torch.nn.functional import softmax

# Ensure NLTK resources are downloaded
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# Define keywords for category distinction
category_keywords = {
    "Models of Manipulating Qubits for Computation": "Quantum gates, Qubit control, Quantum circuits, Quantum algorithms",
    "Methods of Building Qubits": "Superconducting qubits, Trapped ions, Quantum dots, Semiconductor qubits",
    "Address Obstacles to Quantum Computation": "Quantum error correction, Decoherence, Noise mitigation",
    "Applications of Quantum Computing": "Quantum simulation, Cryptography, Optimization problems"
}

# Text cleaning function
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Retain only letters, numbers, and spaces
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# Text augmentation - synonym replacement (applies only to nouns)
def synonym_replacement(text):
    words = text.split()
    new_words = []
    for word in words:
        synonyms = wordnet.synsets(word, pos=wordnet.NOUN)  # Restrict replacement to nouns
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

# Define category mappings
categories_original = list(category_keywords.keys())

# Load dataset
train_df = pd.read_csv("training.csv", encoding_errors="ignore").dropna(subset=['category'])
train_df = train_df[train_df['category'].isin(categories_original)]

# Clean text fields
train_df["title"] = train_df["title"].apply(clean_text)
train_df["abstract"] = train_df["abstract"].apply(clean_text)

# Append category-specific keywords to enhance category distinction
train_df["combined_text"] = train_df.apply(
    lambda row: row["title"] + " " + row["abstract"] + " " + category_keywords[row["category"]], axis=1
)

# Apply text augmentation
train_df["augmented_text_synonym"] = train_df["combined_text"].apply(synonym_replacement)

# Process category labels
train_texts = train_df["combined_text"].tolist() + train_df["augmented_text_synonym"].tolist()
label_map = {category: idx for idx, category in enumerate(categories_original)}
train_labels = train_df['category'].map(label_map).astype(int).tolist() * 2  # Duplicate labels for augmented data

# Split dataset into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    train_texts, train_labels, test_size=0.2, stratify=train_labels, random_state=42
)

# Load MiniLM tokenizer and model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)

train_dataset = QuantumDataset(train_texts, train_labels, tokenizer)
test_dataset = QuantumDataset(test_texts, test_labels, tokenizer)

# Compute metrics
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
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=6,
    learning_rate=2e-5,
    weight_decay=0.05,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2
)

# Custom callback to log evaluation loss
class LossLoggerCallback(TrainerCallback):
    def __init__(self):
        self.eval_losses = []

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if "eval_loss" in metrics:
            self.eval_losses.append(metrics["eval_loss"])

loss_logger = LossLoggerCallback()

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3), loss_logger]
)

trainer.train()

# Plot evaluation loss over epochs
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(loss_logger.eval_losses) + 1), loss_logger.eval_losses, marker='o', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel("Eval Loss")
plt.title("Evaluation Loss per Epoch")
plt.grid()
plt.show()

# Make predictions
predictions = trainer.predict(test_dataset)
logits = predictions.predictions
probs = softmax(torch.tensor(logits), dim=1).numpy()
predicted_labels = np.argmax(probs, axis=1)

# Error analysis
incorrect_rows = []
for idx, (true, pred) in enumerate(zip(test_labels, predicted_labels)):
    if true != pred:
        top2_pred = np.argsort(probs[idx])[-2:]
        incorrect_rows.append({
            "original_text": test_texts[idx],
            "predicted_category": categories_original[pred],
            "correct_category": categories_original[true],
            "confidence": probs[idx][pred],
            "top_2_categories": f"{categories_original[top2_pred[-1]]}, {categories_original[top2_pred[-2]]}"
        })

# Save misclassified samples
incorrect_df = pd.DataFrame(incorrect_rows)
incorrect_df.to_excel("wrong_set.xlsx", index=False)
