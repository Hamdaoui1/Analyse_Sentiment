import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from data_loader import load_absa_data
from preprocessing import preprocess_text_column
import xml.etree.ElementTree as ET

# Vérifier si un GPU est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Charger le modèle BERT entraîné
model_path = "models/bert"
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# Charger le tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)

# Mapping des labels
label_mapping = {1: "positive", 0: "negative", 2: "neutral"}

# Fonction de prédiction avec BERT
def predict_polarity(xml_file, output_file):
    df = load_absa_data(xml_file)
    df = preprocess_text_column(df, text_column="text")

    predictions = []
    for text in df["text"]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).cpu().numpy()[0]
        predictions.append(label_mapping[pred])

    df["predicted_polarity"] = predictions

    # Création d'un fichier XML de sortie
    root = ET.Element("sentences")
    for i, row in df.iterrows():
        sentence = ET.SubElement(root, "sentence", id=row["sentence_id"])
        text = ET.SubElement(sentence, "text")
        text.text = row["text"]

        aspect_terms = ET.SubElement(sentence, "aspectTerms")
        for aspect in row["aspect_terms"]:
            ET.SubElement(aspect_terms, "aspectTerm", term=aspect, polarity=row["predicted_polarity"])

    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"✅ Prédictions sauvegardées dans {output_file}")

if __name__ == "__main__":
    print("🔄 Prédiction en cours sur les laptops...")
    predict_polarity("data/Laptop_Test_NoLabels.xml", "results/bert_predictions_laptops.xml")

    print("\n🔄 Prédiction en cours sur les restaurants...")
    predict_polarity("data/Restaurants_Test_NoLabels.xml", "results/bert_predictions_restaurants.xml")

    print("\n✅ Prédictions terminées avec BERT et sauvegardées dans le dossier results.")
