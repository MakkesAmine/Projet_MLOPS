from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from . import model as ml_model  # Utilisation du chemin relatif
from sklearn.ensemble import RandomForestClassifier  # Importer RandomForestClassifier
from sklearn.metrics import accuracy_score  # Importer accuracy_score

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

MODEL_PATH = "models/best_model.pkl"

# Vérifier si le modèle existe
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Le fichier du modèle est introuvable à l'emplacement: {MODEL_PATH}")

# Charger le modèle préalablement sauvegardé
model = ml_model.load_model(MODEL_PATH)

class PredictionRequest(BaseModel):
    features: list  # Assurez-vous que les données sont des float et contiennent au moins une caractéristique

class RetrainRequest(BaseModel):
    n_estimators: int
    max_depth: int
    min_samples_split: int
    train_path: str
    test_path: str

@app.post("/predict/")
def predict(request: PredictionRequest):
    try:
        input_data = np.array(request.features).reshape(1, -1)
        prediction = model.predict(input_data)
        prediction_result = int(prediction[0])
        return {"prediction": prediction_result}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Erreur de valeur des données d'entrée: {ve}")
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=f"Fichier du modèle introuvable: {fnf}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {e}")

@app.post("/retrain/")
def retrain(request: RetrainRequest):
    try:
        X_train, X_test, y_train, y_test = ml_model.prepare_data(request.train_path, request.test_path)
        new_model = RandomForestClassifier(
            n_estimators=request.n_estimators,
            max_depth=request.max_depth,
            min_samples_split=request.min_samples_split
        )
        new_model.fit(X_train, y_train)
        y_pred = new_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        ml_model.save_model(new_model, MODEL_PATH)
        return {"message": "Modèle réentraîné avec succès", "accuracy": accuracy}
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=f"Fichier de données introuvable: {fnf}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du réentraînement du modèle: {e}")

# Ajouter les configurations pour Elasticsearch et Kibana
@app.get("/elasticsearch/")
def get_elasticsearch_info():
    return {"es_host": os.getenv("ES_HOST", "http://elasticsearch:9200")}

@app.get("/kibana/")
def get_kibana_info():
    return {"kibana_host": os.getenv("KIBANA_HOST", "http://kibana:5601")}
