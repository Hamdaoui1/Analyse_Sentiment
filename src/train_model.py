import pandas as pd
import numpy as np
import joblib
import gensim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import preprocess_text_column
from data_loader import load_absa_data

# Charger les données
xml_file = "data/Laptop_Train.xml"
df = load_absa_data(xml_file)

# Nettoyage des textes
df = preprocess_text_column(df, text_column="text")

# Charger le modèle Word2Vec
w2v_model = gensim.models.Word2Vec.load("models/word2vec.model")

# Fonction pour convertir un texte en vecteur Word2Vec
def get_word2vec_vector(text, model, vector_size=100):
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(vector_size)

# Appliquer Word2Vec sur toutes les phrases
X = np.array([get_word2vec_vector(text, w2v_model) for text in df["text"]])

# Convertir les polarités en labels numériques
label_mapping = {"positive": 1, "negative": 0, "neutral": 2}
df["aspect_polarities"] = df["aspect_polarities"].apply(lambda x: [label_mapping.get(p, 2) for p in x])

# Supprimer les phrases sans polarité
df = df[df["aspect_polarities"].str.len() > 0]

# Créer un dataset plat (une ligne par aspect)
X_flat, y_flat = [], []
for i, row in df.iterrows():
    for polarity in row["aspect_polarities"]:
        X_flat.append(X[i])
        y_flat.append(polarity)

# Diviser les données en train/test
X_train, X_test, y_train, y_test = train_test_split(X_flat, y_flat, test_size=0.2, random_state=42)

# Entraîner un modèle Random Forest
rf_model = RandomForestClassifier(n_estimators=200, max_depth=30, min_samples_split=5, min_samples_leaf=2, random_state=42)
rf_model.fit(X_train, y_train)

# Évaluer le modèle
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Accuracy with Word2Vec:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Sauvegarder le modèle et Word2Vec
joblib.dump(rf_model, "models/rf_model_word2vec.pkl")
w2v_model.save("models/word2vec.model")
