import xml.etree.ElementTree as ET
from sklearn.metrics import accuracy_score, classification_report

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

def evaluate_predictions(gold_file, pred_file):
    y_true = load_gold_labels(gold_file)
    y_pred = load_predicted_labels(pred_file)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    print("Évaluation des prédictions pour les laptops:")
    evaluate_predictions("data/Laptop_Test_Gold.xml", "results/predictions_laptops.xml")

    print("\nÉvaluation des prédictions pour les restaurants:")
    evaluate_predictions("data/Restaurants_Test_Gold.xml", "results/predictions_restaurants.xml")
