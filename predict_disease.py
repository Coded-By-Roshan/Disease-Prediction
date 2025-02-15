import pickle
import numpy as np
import spacy
import pandas as pd
from fuzzywuzzy import process
from fastapi import FastAPI
from pydantic import BaseModel
import wikipedia

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load trained model
with open("disease_model.pkl", "rb") as file:
    model, label_encoder, symptom_columns = pickle.load(file)

all_symptoms = [symptom.lower() for symptom in symptom_columns]


app = FastAPI(title="Disease Prediction API", description="Predict diseases based on symptom descriptions.")


class SymptomRequest(BaseModel):
    description: str

def extract_symptoms(text):
    doc = nlp(text.lower())
    extracted_symptoms = set()

    for token in doc:
        match, score = process.extractOne(token.text, all_symptoms)
        if score > 75: 
            extracted_symptoms.add(match)

    return list(extracted_symptoms)

def predict_top_3_diseases(description):
    
    symptoms = extract_symptoms(description)
    
    if not symptoms:
        return "No known symptoms detected. Please provide more details."

    symptom_dict = {symptom: 0 for symptom in symptom_columns}
    for symptom in symptoms:
        if symptom in symptom_dict:
            symptom_dict[symptom] = 1

    
    user_data = pd.DataFrame([symptom_dict])

    
    disease_probs = model.predict_proba(user_data)[0]

    
    top_3_indices = np.argsort(disease_probs)[::-1][:3]
    
    top_3_diseases = label_encoder.inverse_transform(top_3_indices)
    top_3_probabilities = disease_probs[top_3_indices]

    
    top_3_results = []
    for disease, prob in zip(top_3_diseases, top_3_probabilities):
        try:

            summary = wikipedia.summary(disease, sentences=1)
        except wikipedia.exceptions.DisambiguationError as e:
            summary = f"Multiple matches found for {disease}: {e.options}"
        except wikipedia.exceptions.PageError:
            summary = f"Page for {disease} not found."
        except wikipedia.exceptions.RedirectError:
            summary = f"{disease} page has been redirected."


        disease_symptoms = []
        for symptom in symptoms:
            if symptom in symptom_columns:
                disease_symptoms.append(symptom)
        top_3_results.append({
            "Disease": disease,
            "Summary": summary,
            "Probability": prob,
            "Matched Symptoms": disease_symptoms
        })

    return top_3_results


@app.post("/predict", summary="Predict Disease")
async def predict(request: SymptomRequest):
    """API endpoint to predict disease based on symptoms."""
    disease = predict_top_3_diseases(request.description)
    return disease


