import pandas as pd
import joblib
import gensim
from preprocessing import preprocess_text_column
from data_loader import load_absa_data

# Charger les données
xml_file = "data/Laptop_Train.xml"
df = load_absa_data(xml_file)

# Nettoyage des textes
df = preprocess_text_column(df, text_column="text")

# Tokenisation pour Word2Vec
sentences = [text.split() for text in df["text"]]

# Entraîner Word2Vec
w2v_model = gensim.models.Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Sauvegarder le modèle Word2Vec
w2v_model.save("models/word2vec.model")

print("Word2Vec model trained and saved successfully!")
