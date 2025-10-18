# fraude_modelo.py
# ---------------------------------------------------------
# Modelo de probabilidad de fraude + ingreso manual (CLI)
# Autor: Ricardo Toro
# Tarea Final - Curso Cloud Computing - Magister Data Science
# ---------------------------------------------------------

import sys
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import joblib

# Rutas fijas
CSV_PATH = r"C:/Users/tibox/OneDrive/Desktop/Magister Data Science/Curso 7 Cloud Computing/Proyecto_Final/MODELAMIENTO_MONTO_FRAUDE.csv"
MODELO_PATH = r"C:/Users/tibox/OneDrive/Desktop/Magister Data Science/Curso 7 Cloud Computing/Proyecto_Final/modelo_fraude.pkl"


# -----------------------------
# Utilidades de IO seguras
# -----------------------------
def safe_input(prompt):
    """input() que no revienta si hay Ctrl+C o EOF; retorna None en esos casos."""
    try:
        return input(prompt)
    except (KeyboardInterrupt, EOFError):
        return None


# -----------------------------
# 1) Cargar y limpiar
# -----------------------------
def cargar_datos():
    df = pd.read_table(CSV_PATH, header=0, sep=";").copy()

    if "TIPO_PRODUCTO " in df.columns:
        df.rename(columns={"TIPO_PRODUCTO ": "TIPO_PRODUCTO"}, inplace=True)

    df["FECHA_INICIAL"] = pd.to_datetime(df["FECHA_INICIAL"], format="%d-%m-%Y", errors="coerce")
    df["FECHA_DETECCION"] = pd.to_datetime(df["FECHA_DETECCION"], format="%d-%m-%Y", errors="coerce")

    if "MONTO_FRAUDE" not in df.columns:
        raise ValueError("La columna 'MONTO_FRAUDE' no existe en el CSV.")
    df["FRAUDE"] = (df["MONTO_FRAUDE"] > 0).astype(int)

    return df


# -----------------------------
# 2) Feature engineering
# -----------------------------
def feature_engineering(df_in):
    X = df_in.copy()

    X["DIAS_DETECCION"] = (X["FECHA_DETECCION"] - X["FECHA_INICIAL"]).dt.days
    X["DIA_SEMANA_INICIAL"] = X["FECHA_INICIAL"].dt.dayofweek
    X["DIA_SEMANA_DETECCION"] = X["FECHA_DETECCION"].dt.dayofweek

    bins = [0, 5, 10, np.inf]
    labels = ["BAJO", "MEDIO", "ALTO"]
    X["N_OPERACIONES_CAT"] = pd.cut(X["N_OPERACIONES"], bins=bins, labels=labels)

    X["FRAUDE_PREVIO"] = (X["N_FRAUDES_ANTERIORES"] > 0).astype(int)

    X["DIAS_SEMANA_PAIR"] = X["DIA_SEMANA_INICIAL"].astype(str) + "_" + X["DIA_SEMANA_DETECCION"].astype(str)
    X["TIPO_CLIENTE"] = X["TIPO_PRODUCTO"].astype(str) + "_" + X["FLAG_CLIENTE_EMPRESA"].astype(str)

    X.drop(columns=["FECHA_INICIAL", "FECHA_DETECCION"], inplace=True, errors="ignore")

    return X


def _imprimir_dist(nombre, y):
    vals, cnts = np.unique(y, return_counts=True)
    dist = {int(v): int(c) for v, c in zip(vals, cnts)}
    print(f"{nombre} -> distribución de clases: {dist}")


# -----------------------------
# Helper: probabilidad de clase positiva (1)
# -----------------------------
def _proba_pos(pipe, X):
    """
    Devuelve la probabilidad de la clase 1 aun si el modelo se entrenó con 1 sola clase.
    - Si hay 2 clases, toma la columna correspondiente a la clase 1.
    - Si hay 1 clase, retorna 1.0 si la única clase es 1; 0.0 si es 0.
    Retorna un float si X tiene una sola fila; si no, un vector.
    """
    clf = pipe.named_steps.get("clf", None)
    if clf is None:
        clf = pipe.steps[-1][1]

    classes = getattr(clf, "classes_", None)
    proba = pipe.predict_proba(X)

    if classes is None:
        col = 1 if proba.shape[1] > 1 else 0
        out = proba[:, col]
    else:
        classes = np.array(classes)
        if classes.shape[0] == 2:
            idx_pos = int(np.where(classes == 1)[0][0])
            out = proba[:, idx_pos]
        else:
            unica = int(classes[0])
            out = np.ones(proba.shape[0], dtype=float) if unica == 1 else np.zeros(proba.shape[0], dtype=float)

    return float(out[0]) if X.shape[0] == 1 else out


# -----------------------------
# 3) Entrenar modelo (robusto a 1 sola clase)
# -----------------------------
def entrenar_modelo(df, test_size=0.30, random_state=222):
    X_full = feature_engineering(df)
    y = df["FRAUDE"].astype(int).values

    cols_excluir = {"ID", "FRAUDE", "MONTO_FRAUDE"}
    X = X_full.drop(columns=[c for c in cols_excluir if c in X_full.columns], errors="ignore")

    numericas = [c for c in ["N_OPERACIONES", "N_FRAUDES_ANTERIORES", "DIAS_DETECCION"] if c in X.columns]
    categoricas = [
        c
        for c in [
            "TIPO_PRODUCTO",
            "FLAG_CLIENTE_EMPRESA",
            "N_OPERACIONES_CAT",
            "DIA_SEMANA_INICIAL",
            "DIA_SEMANA_DETECCION",
            "DIAS_SEMANA_PAIR",
            "TIPO_CLIENTE",
        ]
        if c in X.columns
    ]

    if len(numericas) == 0 and len(categoricas) == 0:
        raise ValueError("No se encontraron columnas numéricas ni categóricas válidas para entrenar.")

    print("\n=== Chequeos de clases ===")
    _imprimir_dist("Total", y)

    clases_unicas = np.unique(y)
    if len(clases_unicas) < 2:
        print("\n⚠️ AVISO: El dataset completo tiene UNA sola clase.")
        print("Se utilizará DummyClassifier (probabilidad constante = prevalencia).")
        preprocesador = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numericas),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
            ],
            remainder="drop",
        )
        dummy = DummyClassifier(strategy="prior")
        pipe = Pipeline(steps=[("prep", preprocesador), ("clf", dummy)])
        pipe.fit(X, y)

        print(f"Métrica informativa: prevalencia FRAUDE = {y.mean():.4f}")
        return {
            "pipe": pipe,
            "X_cols": X.columns.tolist(),
            "numericas": numericas,
            "categorias_TIPO_PRODUCTO": sorted(df["TIPO_PRODUCTO"].dropna().astype(str).unique().tolist())
            if "TIPO_PRODUCTO" in df.columns
            else [],
            "categorias_FLAG_CLIENTE_EMPRESA": sorted(df["FLAG_CLIENTE_EMPRESA"].dropna().astype(str).unique().tolist())
            if "FLAG_CLIENTE_EMPRESA" in df.columns
            else [],
            "solo_una_clase": True,
        }

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    (train_idx, test_idx) = next(sss.split(X, y))
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y[train_idx], y[test_idx]

    _imprimir_dist("Train", y_train)
    _imprimir_dist("Test ", y_test)

    preprocesador = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
        ],
        remainder="drop",
    )

    modelo = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=random_state,
    )

    pipe = Pipeline(steps=[("prep", preprocesador), ("clf", modelo)])
    pipe.fit(X_train, y_train)

    y_proba_pos = _proba_pos(pipe, X_test)
    y_pred = (y_proba_pos >= 0.5).astype(int)

    print("\n=== Métricas de Validación (umbral 0.5) ===")
    print(classification_report(y_test, y_pred, digits=4))
    try:
        print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_pos):.4f}")
        print(f"PR-AUC : {average_precision_score(y_test, y_proba_pos):.4f}")
    except Exception as e:
        print(f"No fue posible calcular AUC/AP: {e}")

    return {
        "pipe": pipe,
        "X_cols": X.columns.tolist(),
        "numericas": numericas,
        "categorias_TIPO_PRODUCTO": sorted(df["TIPO_PRODUCTO"].dropna().astype(str).unique().tolist())
        if "TIPO_PRODUCTO" in df.columns
        else [],
        "categorias_FLAG_CLIENTE_EMPRESA": sorted(df["FLAG_CLIENTE_EMPRESA"].dropna().astype(str).unique().tolist())
        if "FLAG_CLIENTE_EMPRESA" in df.columns
        else [],
        "solo_una_clase": False,
    }


# -----------------------------
# 4) Predicción para nueva transacción
# -----------------------------
def feature_engineering_row(tx, X_columns, numericas):
    row = pd.DataFrame([tx]).copy()
    row["FECHA_INICIAL"] = pd.to_datetime(row["FECHA_INICIAL"], format="%d-%m-%Y", errors="coerce")
    row["FECHA_DETECCION"] = pd.to_datetime(row["FECHA_DETECCION"], format="%d-%m-%Y", errors="coerce")

    row_feat = feature_engineering(row)

    for c in X_columns:
        if c not in row_feat.columns:
            row_feat[c] = np.nan
    for c in numericas:
        if c in row_feat.columns:
            row_feat[c] = row_feat[c].fillna(0)

    row_feat = row_feat[X_columns]
    return row_feat


def predecir_fraude(info_modelo, tx):
    pipe = info_modelo["pipe"]
    X_columns = info_modelo["X_cols"]
    numericas = info_modelo["numericas"]
    row_feat = feature_engineering_row(tx, X_columns, numericas)
    proba = _proba_pos(pipe, row_feat)
    return float(proba)


# -----------------------------
# 5) Guardar / Cargar modelo .pkl
# -----------------------------
def guardar_modelo(info_modelo):
    data = {
        "pipe": info_modelo["pipe"],
        "X_cols": info_modelo["X_cols"],
        "numericas": info_modelo["numericas"],
        "categorias_TIPO_PRODUCTO": info_modelo.get("categorias_TIPO_PRODUCTO", []),
        "categorias_FLAG_CLIENTE_EMPRESA": info_modelo.get("categorias_FLAG_CLIENTE_EMPRESA", []),
        "solo_una_clase": info_modelo.get("solo_una_clase", False),
    }
    os.makedirs(os.path.dirname(MODELO_PATH), exist_ok=True)
    joblib.dump(data, MODELO_PATH)
    print(f"\n✅ Modelo guardado en: {MODELO_PATH}")


def cargar_modelo():
    if not os.path.exists(MODELO_PATH):
        raise FileNotFoundError(f"No se encontró el archivo {MODELO_PATH}")
    modelo = joblib.load(MODELO_PATH)
    print(f"📂 Modelo cargado desde: {MODELO_PATH}")
    return modelo


# -----------------------------
# 6) Ingreso manual por consola (tolerante a Ctrl+C / EOF)
# -----------------------------
def ingreso_manual(info_modelo):
    print("\n=== Ingreso manual de transacción ===")
    print("Formato fechas: dd-mm-YYYY (ej: 10-09-2025)")
    if info_modelo.get("solo_una_clase", False):
        print("⚠️ AVISO: Modelo Dummy (dataset con 1 sola clase). "
              "La probabilidad será la prevalencia observada (constante).")
    print("Valores conocidos para TIPO_PRODUCTO:", info_modelo.get("categorias_TIPO_PRODUCTO", []))
    print("Valores conocidos para FLAG_CLIENTE_EMPRESA:", info_modelo.get("categorias_FLAG_CLIENTE_EMPRESA", []))
    print("Deja vacío FECHA_INICIAL o presiona Ctrl+C para salir.\n")

    while True:
        fi = safe_input("FECHA_INICIAL: ")
        if fi is None:
            print("\nIngreso manual cancelado por el usuario.")
            break
        fi = fi.strip()
        if fi == "":
            print("Fin del ingreso manual.")
            break

        fd = safe_input("FECHA_DETECCION: ")
        if fd is None:
            print("\nIngreso manual cancelado por el usuario.")
            break
        fd = fd.strip()

        nops_str = safe_input("N_OPERACIONES (int): ")
        if nops_str is None:
            print("\nIngreso manual cancelado por el usuario.")
            break
        nfprev_str = safe_input("N_FRAUDES_ANTERIORES (int): ")
        if nfprev_str is None:
            print("\nIngreso manual cancelado por el usuario.")
            break

        tprod = safe_input("TIPO_PRODUCTO: ")
        if tprod is None:
            print("\nIngreso manual cancelado por el usuario.")
            break
        flage = safe_input("FLAG_CLIENTE_EMPRESA: ")
        if flage is None:
            print("\nIngreso manual cancelado por el usuario.")
            break

        try:
            nops = int(nops_str.strip())
            nfprev = int(nfprev_str.strip())
        except Exception:
            print("Error: N_OPERACIONES y N_FRAUDES_ANTERIORES deben ser enteros.\n")
            continue

        tx = {
            "FECHA_INICIAL": fi,
            "FECHA_DETECCION": fd,
            "N_OPERACIONES": nops,
            "N_FRAUDES_ANTERIORES": nfprev,
            "TIPO_PRODUCTO": (tprod or "").strip(),
            "FLAG_CLIENTE_EMPRESA": (flage or "").strip(),
        }

        try:
            proba = predecir_fraude(info_modelo, tx)
            print(f"→ Probabilidad de fraude: {proba:.3f}\n")
        except Exception as e:
            print("Error al predecir:", e, "\n")


# -----------------------------
# 7) Main
# -----------------------------
def main():
    print("Cargando datos desde:", CSV_PATH)
    try:
        df = cargar_datos()
    except Exception as e:
        print("Error al cargar el archivo CSV:", e)
        sys.exit(1)

    print("Entrenando modelo...")
    try:
        modelo_info = entrenar_modelo(df, test_size=0.30, random_state=222)
    except Exception as e:
        print("Error al entrenar el modelo:", e)
        sys.exit(1)

    # Guardar .pkl automáticamente
    try:
        guardar_modelo(modelo_info)
    except Exception as e:
        print("Error al guardar el modelo:", e)

    resp = safe_input("\n¿Deseas ingresar transacciones manualmente para predecir? (S/N)\n> ")
    if (resp or "").strip().lower() in ["s", "si", "sí", "y", "yes"]:
        ingreso_manual(modelo_info)
    else:
        print("Ejecución finalizada.")


if __name__ == "__main__":
    main()
