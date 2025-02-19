import joblib
import gensim
import numpy as np
import pandas as pd
from data_loader import load_absa_data
from preprocessing import preprocess_text_column
import xml.etree.ElementTree as ET

# Charger le modèle Random Forest optimisé et Word2Vec
rf_model = joblib.load("models/rf_model_word2vec.pkl")
w2v_model = gensim.models.Word2Vec.load("models/word2vec.model")

label_mapping = {1: "positive", 0: "negative", 2: "neutral"}

# Fonction pour convertir un texte en vecteur Word2Vec
def get_word2vec_vector(text, model, vector_size=100):
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(vector_size)

def predict_polarity(xml_file, output_file):
    df = load_absa_data(xml_file)
    df = preprocess_text_column(df, text_column="text")
    
    X = np.array([get_word2vec_vector(text, w2v_model) for text in df["text"]])

    predictions = rf_model.predict(X)
    df["predicted_polarity"] = [label_mapping[p] for p in predictions]

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

if __name__ == "__main__":
    predict_polarity("data/Laptop_Test_NoLabels.xml", "results/word2vec_predictions_laptops.xml")
    predict_polarity("data/Restaurants_Test_NoLabels.xml", "results/word2vec_predictions_restaurants.xml")
    print("Prédictions terminées avec Word2Vec et sauvegardées dans le dossier results.")
