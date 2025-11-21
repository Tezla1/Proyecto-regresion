import joblib
import pandas as pd

class SpotifyChurnPredictor:
    """Usa el modelo guardado para hacer predicciones."""

    def __init__(self):
        self.modelo = joblib.load("models/modelo_prestamos.pkl")

    def predecir(self, datos_usuario):
        df = pd.DataFrame([datos_usuario])
        prob = self.modelo.predict_proba(df)[0][1]
        pred = self.modelo.predict(df)[0]

        resultado = "Prestamo Aprobado" if pred == 1 else "Prestamo Negado"
        print(f"Resultado: {resultado} (Probabilidad: {prob:.2%})")

        return resultado, prob