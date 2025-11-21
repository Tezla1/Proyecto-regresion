from src.data_loader import DataLoader
from src.train_model import modelo_prestamos

def main():
    print("Entrenando modelo de prestamos...")

    loader = DataLoader("data/raw/Loan_approval_data_2025.csv")

    df = loader.cargar_datos()

    modelo = modelo_prestamos()
    metricas_modelo = modelo.entrenar(df)

    print("\n✅ Entrenamiento completado con éxito.")
    print(f"Métricas del modelo: ", metricas_modelo)
    print("Modelo guardado en la carpeta 'models/'")

if __name__ == "__main__":
    main()
