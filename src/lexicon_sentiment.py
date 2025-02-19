import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import sentiwordnet as swn
import xml.etree.ElementTree as ET
import pandas as pd
from data_loader import load_absa_data
from preprocessing import preprocess_text_column

# Fonction pour obtenir le score de polarité d'un mot
def get_sentiment_score(word, pos_tag):
    try:
        synsets = list(swn.senti_synsets(word, pos_tag))
        if synsets:
            # Prendre la moyenne des scores positifs et négatifs des synsets trouvés
            pos_score = sum([syn.pos_score() for syn in synsets]) / len(synsets)
            neg_score = sum([syn.neg_score() for syn in synsets]) / len(synsets)
            if pos_score > neg_score:
                return "positive"
            elif neg_score > pos_score:
                return "negative"
            else:
                return "neutral"
        else:
            return "neutral"
    except:
        return "neutral"

# Mapper les tags POS nltk en tags SentiWordNet
nltk_to_swn = {
    "NN": "n", "VB": "v", "JJ": "a", "RB": "r"
}

# Fonction de prédiction de sentiment
def predict_polarity_lexicon(xml_file, output_file):
    df = load_absa_data(xml_file)
    df = preprocess_text_column(df, text_column="text")

    predictions = []
    for text in df["text"]:
        from nltk.tokenize.punkt import PunktSentenceTokenizer
        tokenizer = PunktSentenceTokenizer()
        def custom_word_tokenize(text):
            return text.split()  # Utilisation d'un simple split au lieu de word_tokenize
        words = custom_word_tokenize(text)
        pos_tags = nltk.pos_tag(words)

        # Prédiction basée sur la moyenne des scores de mots détectés
        aspect_polarities = []
        for word, tag in pos_tags:
            swn_tag = nltk_to_swn.get(tag[:2], "n")  # Par défaut "n" pour les noms
            polarity = get_sentiment_score(word, swn_tag)
            aspect_polarities.append(polarity)

        # Choisir la polarité majoritaire
        final_polarity = max(set(aspect_polarities), key=aspect_polarities.count)
        predictions.append(final_polarity)

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
    print("🔄 Prédiction en cours avec la méthode lexicon-based...")
    predict_polarity_lexicon("data/Laptop_Test_NoLabels.xml", "results/lexicon_predictions_laptops.xml")
    predict_polarity_lexicon("data/Restaurants_Test_NoLabels.xml", "results/lexicon_predictions_restaurants.xml")
    print("\n✅ Prédictions terminées avec la méthode lexicon-based.")
