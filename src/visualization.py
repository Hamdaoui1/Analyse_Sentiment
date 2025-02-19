import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xml.etree.ElementTree as ET
from sklearn.metrics import confusion_matrix, classification_report

# 📊 Données des performances des modèles
models = ["Lexicon-Based", "TF-IDF + SVM", "Word2Vec + RF", "BERT"]
accuracy_laptops = [36.7, 81.6, 81.6, 89.8]
accuracy_restaurants = [44.8, 70.8, 70.8, 79.2]

# 🔹 Fonction pour charger les labels depuis les fichiers XML
def load_labels_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    labels = []
    for sentence in root.findall("sentence"):
        aspect_terms = sentence.find("aspectTerms")
        if aspect_terms is not None:
            for aspect in aspect_terms.findall("aspectTerm"):
                labels.append(aspect.get("polarity"))
    return labels

# 🔹 Comparaison des Accuracy
def plot_accuracy():
    plt.figure(figsize=(8,5))
    x = np.arange(len(models))
    width = 0.35

    plt.bar(x - width/2, accuracy_laptops, width, label="Laptops")
    plt.bar(x + width/2, accuracy_restaurants, width, label="Restaurants")

    plt.xlabel("Modèles")
    plt.ylabel("Accuracy (%)")
    plt.title("Comparaison des Accuracy des modèles")
    plt.xticks(ticks=x, labels=models, rotation=15)
    plt.legend()
    plt.show()

# 🔹 Matrice de confusion
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=["negative", "neutral", "positive"])
    labels = ["Negative", "Neutral", "Positive"]

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Prédictions")
    plt.ylabel("Réel")
    plt.title(f"Matrice de confusion - {model_name}")
    plt.show()

# 🔹 Comparaison des scores Precision, Recall, F1-score
def plot_classification_report(y_true, y_pred, model_name):
    report = classification_report(y_true, y_pred, labels=["negative", "neutral", "positive"], output_dict=True)
    
    labels = ["Negative", "Neutral", "Positive"]
    precision = [report[label]["precision"] for label in labels]
    recall = [report[label]["recall"] for label in labels]
    f1_score = [report[label]["f1-score"] for label in labels]

    x = np.arange(len(labels))
    width = 0.2

    plt.figure(figsize=(8,5))
    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1_score, width, label="F1-score")

    plt.xlabel("Classes")
    plt.ylabel("Score")
    plt.title(f"Scores Precision / Recall / F1 - {model_name}")
    plt.xticks(ticks=x, labels=labels)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # 🔹 Visualisation des Accuracy
    plot_accuracy()

    # 📂 Chargement des vraies données
    y_true_laptops = load_labels_from_xml("data/Laptop_Test_Gold.xml")
    y_pred_svm_laptops = load_labels_from_xml("results/predictions_laptops.xml")
    y_pred_bert_laptops = load_labels_from_xml("results/bert_predictions_laptops.xml")

    y_true_restaurants = load_labels_from_xml("data/Restaurants_Test_Gold.xml")
    y_pred_svm_restaurants = load_labels_from_xml("results/predictions_restaurants.xml")
    y_pred_bert_restaurants = load_labels_from_xml("results/bert_predictions_restaurants.xml")

    # 🔹 Matrices de confusion
    plot_confusion_matrix(y_true_laptops, y_pred_svm_laptops, "SVM - Laptops")
    plot_confusion_matrix(y_true_laptops, y_pred_bert_laptops, "BERT - Laptops")

    plot_confusion_matrix(y_true_restaurants, y_pred_svm_restaurants, "SVM - Restaurants")
    plot_confusion_matrix(y_true_restaurants, y_pred_bert_restaurants, "BERT - Restaurants")

    # 🔹 Comparaison des scores
    plot_classification_report(y_true_laptops, y_pred_svm_laptops, "SVM - Laptops")
    plot_classification_report(y_true_laptops, y_pred_bert_laptops, "BERT - Laptops")

    plot_classification_report(y_true_restaurants, y_pred_svm_restaurants, "SVM - Restaurants")
    plot_classification_report(y_true_restaurants, y_pred_bert_restaurants, "BERT - Restaurants")
