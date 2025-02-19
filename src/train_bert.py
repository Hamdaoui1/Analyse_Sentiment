import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from data_loader import load_absa_data
from preprocessing import preprocess_text_column

# Vérifier si un GPU est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Charger les données
xml_file = "data/Laptop_Train.xml"
df = load_absa_data(xml_file)

# Nettoyage des textes
df = preprocess_text_column(df, text_column="text")

# Mapping des labels
label_mapping = {"positive": 1, "negative": 0, "neutral": 2}
df["aspect_polarities"] = df["aspect_polarities"].apply(lambda x: [label_mapping.get(p, 2) for p in x])

# Supprimer les phrases sans polarité
df = df[df["aspect_polarities"].str.len() > 0]

# Définir le tokenizer BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenizer les phrases et convertir les labels
def encode_data(texts, labels, max_length=128):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    encodings["labels"] = labels
    return encodings

# Transformer les données en une seule liste
texts = []
labels = []
for i, row in df.iterrows():
    for polarity in row["aspect_polarities"]:
        texts.append(row["text"])
        labels.append(polarity)

# Encoder les données
encodings = encode_data(texts, labels)

# Convertir en dataset Hugging Face
dataset = Dataset.from_dict(encodings)

# Diviser les données en train/test
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# Définir le modèle BERT
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model.to(device)

# Définir l'entraînement
training_args = TrainingArguments(
    output_dir="models/bert",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="logs",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# Définir l'entraîneur
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Entraîner le modèle
trainer.train()

# Sauvegarder le modèle
model.save_pretrained("models/bert")
tokenizer.save_pretrained("models/bert")

print("✅ BERT Model trained and saved successfully!")
