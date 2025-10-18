# Ricardo_T
Repositorio para tarea final Cloud Computing
Grupo Ricardo Toro
# API de Predicción de Fraude

Proyecto desarrollado con **FastAPI** para predecir la probabilidad de fraude en transacciones financieras.

API de Predicción de Fraude

Aplicación desarrollada en FastAPI que permite predecir la probabilidad de fraude en transacciones financieras.
El modelo fue entrenado en Python con scikit-learn y se sirve como API REST local o desplegable en la nube.

Proyecto_Final/
├── app/
│   ├── __init__.py
│   ├── main.py               ← Servidor FastAPI
│   └── model_utils.py        ← Funciones de ingeniería y predicción
├── models/
│   └── modelo_fraude.pkl     ← Modelo entrenado serializado
├── requirements.txt          ← Dependencias del proyecto
├── archivo fraude_modelo.py original del model
└── README.md


Requisitos previos

Tener instalado Python 3.10 o superior
Tener instalado Git (para clonar el repositorio)
Tener conexión a internet (para instalar dependencias)

Instalación paso a paso
1.Clonar el repositorio
git clone https://github.com/<TU_USUARIO>/Ricardo_T.git
cd Ricardo_T

2.Crear y activar un entorno virtual
En Windows PowerShell:
python -m venv venv
.\venv\Scripts\Activate.ps1

3.Instalar las dependencias
pip install -r requirements.txt

4.Verificar el modelo
Asegúrate de tener el archivo:
models/modelo_fraude.pkl

5.Ejecución local de la API
Desde la carpeta raíz del proyecto (donde está app/):
uvicorn app.main:app --reload --port 8000


Endpoints disponibles
Verificar estado del modelo
GET → http://127.0.0.1:8000/health