# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pickle
import numpy as np
import pandas as pd
import os

# === CONFIGURACIÓN ===
RUTA_MODELO = r"C:\Users\tibox\OneDrive\Desktop\Magister Data Science\Curso 7 Cloud Computing\Proyecto_Final\modelo_5.pkl"

app = FastAPI(title="API de Predicción - Proyecto Final", version="1.1.0")

# Objetos globales
modelo = None
feature_names: Optional[List[str]] = None

# === Esquema de entrada (Pydantic v2) ===
class BatchPrediccion(BaseModel):
    data: List[Dict[str, Any]]

# === Utilidades para extraer modelo/columnas ===
def _try_get_features_from_dict(d: Dict[str, Any]) -> Optional[List[str]]:
    for k in ["features", "feature_names", "feature_list", "columns", "cols"]:
        if k in d and isinstance(d[k], (list, tuple)):
            return list(d[k])
    return None

def _extraer_modelo(obj: Any):
    """
    Intenta extraer (modelo, features) desde diferentes estructuras.
    Retorna (modelo_encontrado, lista_features_o_None).
    """
    # Caso 1: ya es un estimador sklearn / pipeline
    if hasattr(obj, "predict"):
        return obj, None

    # Caso 2: diccionario con distintas convenciones
    if isinstance(obj, dict):
        # intentar llaves conocidas
        for key in ["modelo", "model", "estimator", "pipeline", "best_estimator", "best_estimator_",
                    "clf", "regressor", "regressor_", "estimator_"]:
            if key in obj and hasattr(obj[key], "predict"):
                feats = _try_get_features_from_dict(obj)
                return obj[key], feats
        # si ningún key coincide pero hay algún valor con predict
        for k, v in obj.items():
            if hasattr(v, "predict"):
                feats = _try_get_features_from_dict(obj)
                return v, feats

    # Caso 3: tupla/lista donde una entrada es el modelo
    if isinstance(obj, (list, tuple)):
        for item in obj:
            if hasattr(item, "predict"):
                return item, None

    # No se encontró un estimador con .predict()
    return None, None

# === Startup: cargar y normalizar modelo ===
@app.on_event("startup")
def cargar_modelo():
    global modelo, feature_names

    if not os.path.exists(RUTA_MODELO):
        raise RuntimeError(f"No se encontró el archivo de modelo en: {RUTA_MODELO}")

    with open(RUTA_MODELO, "rb") as f:
        obj = pickle.load(f)

    m, feats = _extraer_modelo(obj)
    if m is None:
        # Mensaje de diagnóstico útil
        tipo = type(obj)
        claves = list(obj.keys()) if isinstance(obj, dict) else None
        raise RuntimeError(
            f"El objeto cargado no contiene un estimador con .predict(). "
            f"Tipo: {tipo}. Claves: {claves}. "
            f"Guarda un estimador sklearn o un dict con key 'modelo'/'model'/etc."
        )

    modelo = m
    feature_names = feats

    print("✅ Modelo cargado correctamente.")
    if feature_names is not None:
        print(f"✅ Columnas esperadas: {len(feature_names)}")

# === Health ===
@app.get("/health")
def health():
    return {"status": "ok"}

# === Predicción ===
@app.post("/predict")
def predict(payload: BatchPrediccion):
    try:
        if len(payload.data) == 0:
            raise HTTPException(status_code=400, detail="La lista 'data' no puede estar vacía.")

        df_in = pd.DataFrame(payload.data)

        # Alinear columnas si las conocemos
        global feature_names
        if feature_names is not None:
            # Agregar faltantes como NaN y reordenar
            for col in feature_names:
                if col not in df_in.columns:
                    df_in[col] = np.nan
            df_in = df_in[feature_names]

        # Predecir
        try:
            y_pred = modelo.predict(df_in)
        except Exception:
            y_pred = modelo.predict(df_in.to_numpy())

        preds = [float(p) for p in np.asarray(y_pred).ravel()]

        # Probabilidades si clasificador
        probs = None
        if hasattr(modelo, "predict_proba"):
            try:
                proba = modelo.predict_proba(df_in)
            except Exception:
                proba = modelo.predict_proba(df_in.to_numpy())
            probs = proba.tolist()

        return {"ok": True, "n_predictions": len(preds), "predictions": preds, "probs": probs}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {repr(e)}")
