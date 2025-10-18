# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import joblib, os
from .model_utils import feature_engineering_row, proba_pos

# Ruta robusta al archivo del modelo
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "modelo_fraude.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontró el archivo del modelo en: {MODEL_PATH}")

# Cargar el modelo
modelo_info = joblib.load(MODEL_PATH)
pipe = modelo_info["pipe"]
X_cols = modelo_info["X_cols"]
numericas = modelo_info["numericas"]

# Crear aplicación FastAPI
app = FastAPI(
    title="API Predicción de Fraude",
    version="1.0.0",
    description="Predice la probabilidad de fraude en transacciones."
)

# Modelo de entrada
class Transaccion(BaseModel):
    FECHA_INICIAL: str = Field(..., example="10-09-2025")
    FECHA_DETECCION: str = Field(..., example="12-09-2025")
    N_OPERACIONES: int = Field(..., ge=0, example=5)
    N_FRAUDES_ANTERIORES: int = Field(..., ge=0, example=1)
    TIPO_PRODUCTO: str = Field(..., example="TARJETA_CREDITO")
    FLAG_CLIENTE_EMPRESA: str = Field(..., example="0")

@app.get("/health")
def health():
    return {"status": "ok", "modelo": str(MODEL_PATH)}

@app.post("/predict")
def predict(tx: Transaccion):
    try:
        row_feat = feature_engineering_row(tx.dict(), X_cols, numericas)
        prob = proba_pos(pipe, row_feat)
        return {"probabilidad_fraude": round(float(prob), 4)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al predecir: {e}")
