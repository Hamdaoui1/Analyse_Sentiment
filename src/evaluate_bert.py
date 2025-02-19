import xml.etree.ElementTree as ET
from sklearn.metrics import accuracy_score, classification_report

# Fonction pour charger les labels réels depuis le fichier `Gold`
def load_gold_labels(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    labels = []
    for sentence in root.findall("sentence"):
        aspect_terms = sentence.find("aspectTerms")
        if aspect_terms is not None:
            for aspect in aspect_terms.findall("aspectTerm"):
                labels.append(aspect.get("polarity"))
    return labels

# Fonction pour charger les labels prédits par BERT
def load_predicted_labels(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    labels = []
    for sentence in root.findall("sentence"):
        aspect_terms = sentence.find("aspectTerms")
        if aspect_terms is not None:
            for aspect in aspect_terms.findall("aspectTerm"):
                labels.append(aspect.get("polarity"))
    return labels

# Fonction d'évaluation des performances
def evaluate_predictions(gold_file, pred_file):
    y_true = load_gold_labels(gold_file)
    y_pred = load_predicted_labels(pred_file)

    print("📊 **Résultats d'évaluation**")
    print(f"✅ **Accuracy:** {accuracy_score(y_true, y_pred):.4f}")
    print("\n🔹 **Rapport de classification :**\n")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    print("\n🔍 **Évaluation des prédictions BERT pour les laptops :**")
    evaluate_predictions("data/Laptop_Test_Gold.xml", "results/bert_predictions_laptops.xml")

    print("\n🔍 **Évaluation des prédictions BERT pour les restaurants :**")
    evaluate_predictions("data/Restaurants_Test_Gold.xml", "results/bert_predictions_restaurants.xml")
