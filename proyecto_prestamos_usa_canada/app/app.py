import streamlit as st
import sys
import os

# Agregar el directorio raíz del proyecto al path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from src.preprocessing import Preprocesador
import pandas as pd
import joblib

# Cargar modelos y objetos necesarios
escalador = joblib.load("models/escalador_datos_prestamos.joblib")
modelo = joblib.load("models/modelo_prestamos.joblib")
columnas_entrenamiento = joblib.load("models/columnas_entrenamiento.joblib")

# ==== CSS PROFESIONAL ====
st.markdown("""
<style>
body {
    background-color: #f6f8fa;
}
section[data-testid="stSidebar"] {
    background-color: #232f3e;
}
.stApp {
    background: linear-gradient(130deg, #e0eafc 0%, #cfdef3 50%, #e0eafc 100%);
}
h1 {
    color: #24305e;
    letter-spacing: 1.5px;
    font-weight: 900;
}
h3 {
    font-size: 1.8em !important;
    font-weight: 700 !important;
    color: #12565b !important;
    margin-bottom: 20px !important;
    margin-top: 10px !important;
}
/* Cambiar color de las etiquetas a negro */
label {
    color: #000000 !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}
form input, form select {
    border-radius: 8px;
    border: 1.5px solid #4f8a8b70 !important;
    background-color: #ffffff !important;
    color: #212529;
    font-weight: 500;
    font-size: 1.05em;
}
div.stButton > button {
    background-color: #23b6b6;
    color: #fff;
    border-radius: 10px;
    padding: 10px 30px;
    font-size: 1.1em;
    font-weight: bold;
    border: none;
    box-shadow: 0 2px 8px #8bd7d2;
    transition: all .2s;
    letter-spacing: 0.5px;
}
div.stButton>button:hover {
    background: #176b87;
    color: #e0f4ff;
    border: none;
    box-shadow: 0 4px 16px #19969450;
}
.big-card {
    background: linear-gradient(135deg,#ffffff 60%,#23b6b628 100%);
    border-radius: 18px;
    padding: 30px 20px;
    box-shadow: 0 2px 24px #176b8740;
    margin-top: 30px;
    text-align: center;
    font-size: 1.15em;
}
.result-title {
    font-weight: bold;
    font-size: 2.2em;
}
.result-aprobado {
    color: #219867;
    text-shadow: 1px 2px 1px #dafbe5;
}
.result-rechazado {
    color: #d90429;
    text-shadow: 1px 2px 1px #fad6de;
}
.result-badge {
    display: inline-block;
    background: #23b6b6;
    color: #fff;
    border-radius: 14px;
    font-size: 1em;
    padding: 6px 20px;
    margin-top: 12px;
    margin-bottom: 0px;
    font-weight: 800;
    letter-spacing: 1px;
}
.info-small {
    color: #62717a;
    font-size: 0.9em;
    margin-top: 14px;
}
</style>
""", unsafe_allow_html=True)

# ==== TÍTULO PRINCIPAL ====
st.title("💰 Simulador de Aprobaciones de crédito")
st.markdown(
    """
    <p style="font-size: 1.2em; color:#353849; text-align:justify; max-width:650px">
    Este modelo utiliza <strong>regresión logística</strong> para predecir si un cliente será aprobado o rechazado para un préstamo, analizando variables financieras, personales y de historial crediticio.<br>
    Procesa los datos del solicitante, los estandariza y genera una predicción binaria que ayuda a evaluar el riesgo y tomar decisiones crediticias de manera rápida y objetiva.
    </p>
    """, unsafe_allow_html=True
)

# ==== FORMULARIO ====
with st.form("loan_data_form"):
    st.markdown('<h3 style="color:#12565b; margin-bottom:-8px">Información del Cliente</h3>', unsafe_allow_html=True)
    cols1, cols2 = st.columns(2)
    with cols1:
        age = st.number_input("Edad", min_value=18, max_value=100, value=35, step=1)
        occupation_status = st.selectbox("Estado de Ocupación", ["Employed", "Self-Employed", "Student"], index=0)
        years_employed = st.number_input("Años Empleado", min_value=0.0, max_value=50.0, value=5.0, step=0.1, format="%.1f")
        annual_income = st.number_input("Ingreso Anual", min_value=0, max_value=10000000, value=50000, step=1000)
        credit_score = st.number_input("Puntaje de Crédito", min_value=300, max_value=850, value=650, step=1)
        credit_history_years = st.number_input("Años de Historial Crediticio", min_value=0.0, max_value=50.0, value=10.0, step=0.1, format="%.1f")
        savings_assets = st.number_input("Activos en Ahorros", min_value=0, max_value=10000000, value=10000, step=500)
        current_debt = st.number_input("Deuda Actual", min_value=0, max_value=10000000, value=5000, step=500)
        defaults_on_file = st.number_input("Incumplimientos en Archivo", min_value=0, max_value=100, value=0, step=1)
    with cols2:
        delinquencies_last_2yrs = st.number_input("Morosidades Últimos 2 Años", min_value=0, max_value=100, value=0, step=1)
        derogatory_marks = st.number_input("Marcas Derogatorias", min_value=0, max_value=100, value=0, step=1)
        product_type = st.selectbox("Tipo de Producto", ["Personal Loan", "Line of Credit", "Credit Card"], index=0)
        loan_intent = st.selectbox("Intención del Préstamo", ["Home", "Auto", "loan_intent_Education", "Business", "loan_intent_Medical", "Personal"], index=0)
        loan_amount = st.number_input("Monto del Préstamo", min_value=0, max_value=10000000, value=15000, step=500)
        interest_rate = st.number_input("Tasa de Interés %", min_value=0.0, max_value=100.0, value=7.5, step=0.1, format="%.2f")
        debt_to_income_ratio = st.number_input("Ratio Deuda/Ingreso", min_value=0.0, max_value=10.0, value=0.25, step=0.01, format="%.2f")
        loan_to_income_ratio = st.number_input("Ratio Préstamo/Ingreso", min_value=0.0, max_value=10.0, value=0.30, step=0.01, format="%.2f")
        payment_to_income_ratio = st.number_input("Ratio Pago/Ingreso", min_value=0.0, max_value=10.0, value=0.15, step=0.01, format="%.2f")

    # === Botón de enviar predicción ===
    submitted = st.form_submit_button("🔥 Generar Predicción")

    if submitted:
        data_dict = {
            'age': [age],
            'occupation_status': [occupation_status],
            'years_employed': [years_employed],
            'annual_income': [annual_income],
            'credit_score': [credit_score],
            'credit_history_years': [credit_history_years],
            'savings_assets': [savings_assets],
            'current_debt': [current_debt],
            'defaults_on_file': [defaults_on_file],
            'delinquencies_last_2yrs': [delinquencies_last_2yrs],
            'derogatory_marks': [derogatory_marks],
            'product_type': [product_type],
            'loan_intent': [loan_intent],
            'loan_amount': [loan_amount],
            'interest_rate': [interest_rate],
            'debt_to_income_ratio': [debt_to_income_ratio],
            'loan_to_income_ratio': [loan_to_income_ratio],
            'payment_to_income_ratio': [payment_to_income_ratio]
        }
        df = pd.DataFrame(data_dict)
        preprocesador = Preprocesador()
        X = preprocesador.preparar_datos(df)
        X = X.reindex(columns=columnas_entrenamiento, fill_value=0)
        X_transform = escalador.transform(X)
        prediccion = modelo.predict(X_transform)
        probabilidad_prediccion = modelo.predict_proba(X_transform)[0]
        resultado = "Aprobado" if prediccion[0] == 1 else "Rechazado"
        emoji = "✅" if prediccion[0] == 1 else "❌"
        clase_predicha = 1 if prediccion[0] == 1 else 0
        porcentaje = probabilidad_prediccion[clase_predicha] * 100

        # === Tarjeta de resultado visual ===
        st.markdown(f"""
        <div class="big-card">
            <div class="result-title result-{'aprobado' if resultado == 'Aprobado' else 'rechazado'}">
                {resultado} {emoji}
            </div>
            <div class="result-badge">
                Probabilidad: {porcentaje:.2f}%
            </div>
            <div class="info-small">
              hecho por: Mauren Vega,Alejandro Pachon,Sebastian Gonzalez
            </div>
        </div>
        """, unsafe_allow_html=True)
