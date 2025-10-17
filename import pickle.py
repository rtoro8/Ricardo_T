import pickle

# Cargar el modelo
with open(r"C:\Users\tibox\OneDrive\Desktop\Magister Data Science\Curso 7 Cloud Computing\Proyecto_Final\modelo_5.pkl", 'rb') as archivo:
    modelo_cargado = pickle.load(archivo)

# Verificar que se cargó correctamente
print("✅ Modelo cargado correctamente:", type(modelo_cargado))

# Usar el modelo para predecir
y_pred = modelo_cargado.predict(X_test)
print("Predicciones ejemplo:", y_pred[:5])
