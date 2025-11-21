import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.preprocessing import Preprocesador

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class modelo_prestamos:
    def entrenar(self, df):
        # Preparar datos
        preprocesador = Preprocesador()
        X, y = preprocesador.preparar_datos(df)
        # Guardar columnas ANTES del escalado
        joblib.dump(X.columns, "models/columnas_entrenamiento.joblib")


        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print(df.info())
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


        # Entrenar el modelo
        modelo = LogisticRegression()
        modelo.fit(X_train, y_train)

        # Evaluar exactitud
        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1]

        #Calcular metricas
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 3),
            "precision": round(precision_score(y_test, y_pred), 3),
            "recall": round(recall_score(y_test, y_pred), 3),
            "f1": round(f1_score(y_test, y_pred), 3),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 3)
        }

        # Guardar modelo
        joblib.dump(modelo, "models/modelo_prestamos.joblib")
        print("Modelo de Regresion Logistica guardado en 'models/modelo_prestamos.pkl")


        joblib.dump(scaler, "models/escalador_datos_prestamos.joblib")
        print("Escalador de datos guardado en 'models/escalador_datos_prestamos.pkl")


        #Guardar métricas
        with open("reports/metricas.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("Métricas guardadas en 'reports/metrics.json'")

        return metrics