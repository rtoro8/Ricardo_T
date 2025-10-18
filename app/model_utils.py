# app/model_utils.py
import numpy as np
import pandas as pd

def feature_engineering(df_in: pd.DataFrame) -> pd.DataFrame:
    X = df_in.copy()

    if "FECHA_INICIAL" in X.columns:
        X["FECHA_INICIAL"] = pd.to_datetime(X["FECHA_INICIAL"], format="%d-%m-%Y", errors="coerce")
    if "FECHA_DETECCION" in X.columns:
        X["FECHA_DETECCION"] = pd.to_datetime(X["FECHA_DETECCION"], format="%d-%m-%Y", errors="coerce")

    X["DIAS_DETECCION"] = (X["FECHA_DETECCION"] - X["FECHA_INICIAL"]).dt.days
    X["DIA_SEMANA_INICIAL"] = X["FECHA_INICIAL"].dt.dayofweek
    X["DIA_SEMANA_DETECCION"] = X["FECHA_DETECCION"].dt.dayofweek

    bins = [0, 5, 10, np.inf]
    labels = ["BAJO", "MEDIO", "ALTO"]
    X["N_OPERACIONES_CAT"] = pd.cut(X["N_OPERACIONES"], bins=bins, labels=labels)

    X["FRAUDE_PREVIO"] = (X["N_FRAUDES_ANTERIORES"] > 0).astype(int)
    X["DIAS_SEMANA_PAIR"] = X["DIA_SEMANA_INICIAL"].astype(str) + "_" + X["DIA_SEMANA_DETECCION"].astype(str)
    X["TIPO_CLIENTE"] = X["TIPO_PRODUCTO"].astype(str) + "_" + X["FLAG_CLIENTE_EMPRESA"].astype(str)

    return X.drop(columns=["FECHA_INICIAL", "FECHA_DETECCION"], errors="ignore")


def feature_engineering_row(tx: dict, X_columns, numericas):
    row = pd.DataFrame([tx]).copy()
    row_feat = feature_engineering(row)

    for c in X_columns:
        if c not in row_feat.columns:
            row_feat[c] = np.nan
    for c in numericas:
        if c in row_feat.columns:
            row_feat[c] = row_feat[c].fillna(0)

    return row_feat[X_columns]


def proba_pos(pipe, X_df):
    clf = pipe.named_steps.get("clf", None) or pipe.steps[-1][1]
    classes = getattr(clf, "classes_", None)
    proba = pipe.predict_proba(X_df)

    if classes is None:
        col = 1 if proba.shape[1] > 1 else 0
        out = proba[:, col]
    else:
        classes = np.array(classes)
        if classes.shape[0] == 2:
            idx = int(np.where(classes == 1)[0][0])
            out = proba[:, idx]
        else:
            unica = int(classes[0])
            out = np.ones(proba.shape[0]) if unica == 1 else np.zeros(proba.shape[0])

    return float(out[0]) if X_df.shape[0] == 1 else out
