import pandas as pd
import spacy
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

# Charger le modèle de langue anglais de spaCy
nlp = spacy.load("en_core_web_sm")

# Télécharger les stopwords de NLTK si ce n'est pas encore fait
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Nettoie un texte : suppression des caractères spéciaux, mise en minuscule.
    """
    text = text.lower()  # Mettre en minuscule
    text = re.sub(r'\d+', '', text)  # Supprimer les chiffres
    text = text.translate(str.maketrans('', '', string.punctuation))  # Supprimer la ponctuation
    text = text.strip()  # Supprimer les espaces superflus
    return text

def tokenize_and_lemmatize(text):
    """
    Tokenise et lemmatise un texte avec spaCy.
    """
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and not token.is_punct]
    return " ".join(tokens)

def preprocess_text_column(df, text_column="text"):
    """
    Applique le nettoyage et la lemmatisation sur la colonne de texte d'un DataFrame.
    """
    df[text_column] = df[text_column].apply(clean_text)
    df[text_column] = df[text_column].apply(tokenize_and_lemmatize)
    return df

def vectorize_texts(texts):
    """
    Convertit une liste de textes en vecteurs TF-IDF.
    """
    vectorizer = TfidfVectorizer(max_features=5000)  # Limite à 5000 caractéristiques
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

if __name__ == "__main__":
    # Charger les données
    from data_loader import load_absa_data
    
    xml_file = "data/Laptop_Train.xml"  # Modifier pour tester d'autres fichiers
    df = load_absa_data(xml_file)
    
    # Nettoyer et lemmatiser les textes
    df = preprocess_text_column(df, text_column="text")

    # Vectoriser les textes
    X, vectorizer = vectorize_texts(df["text"])

    # Afficher un aperçu
    print("Example de texte après prétraitement :")
    print(df["text"].head())

    print("\nTaille de la matrice TF-IDF :", X.shape)
