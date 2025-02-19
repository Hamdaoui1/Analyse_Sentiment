import xml.etree.ElementTree as ET
import pandas as pd

def load_absa_data(xml_file):
    """
    Charge un fichier XML et extrait les phrases, aspects et polarités.

    :param xml_file: Chemin du fichier XML
    :return: DataFrame contenant les phrases, aspects et polarités
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    data = []
    
    for sentence in root.findall("sentence"):
        sentence_id = sentence.get("id")
        text = sentence.find("text").text if sentence.find("text") is not None else ""
        
        # Extraction des termes d'aspects
        aspect_terms = []
        polarities = []
        
        aspect_terms_tag = sentence.find("aspectTerms")
        if aspect_terms_tag is not None:
            for aspect in aspect_terms_tag.findall("aspectTerm"):
                term = aspect.get("term")
                polarity = aspect.get("polarity")  # Peut être "", "positive", "negative", "neutral"
                
                aspect_terms.append(term)
                polarities.append(polarity)
        
        # Extraction des catégories d'aspects (uniquement pour les restaurants)
        aspect_categories = []
        category_polarities = []
        
        aspect_categories_tag = sentence.find("aspectCategories")
        if aspect_categories_tag is not None:
            for category in aspect_categories_tag.findall("aspectCategory"):
                category_name = category.get("category")
                category_polarity = category.get("polarity")
                
                aspect_categories.append(category_name)
                category_polarities.append(category_polarity)
        
        # Ajout des données dans la liste
        data.append({
            "sentence_id": sentence_id,
            "text": text,
            "aspect_terms": aspect_terms,
            "aspect_polarities": polarities,
            "aspect_categories": aspect_categories,
            "category_polarities": category_polarities
        })
    
    # Conversion en DataFrame
    df = pd.DataFrame(data)
    return df

# Exemple d'utilisation
if __name__ == "__main__":
    # Charger un fichier de test (modifier avec le bon chemin)
    xml_file = "data/Laptop_Train.xml"
    df = load_absa_data(xml_file)
    
    # Afficher les 5 premières lignes du DataFrame pour vérifier les données extraites
    print(df.head())