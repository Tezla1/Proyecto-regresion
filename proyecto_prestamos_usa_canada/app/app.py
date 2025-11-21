import streamlit as st
import sys
import os

# Agregar el directorio raíz del proyecto al path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from src.preprocessing import Preprocesador
import pandas as pd
import joblib


escalador = joblib.load("models/escalador_datos_prestamos.joblib")
modelo = joblib.load("models/modelo_prestamos.joblib")
columnas_entrenamiento = joblib.load("models/columnas_entrenamiento.joblib")

st.title("Formulario de Datos de Préstamo")
st.write("Ingresa los datos del cliente. Cada campo tiene un valor por defecto.")


# Crear el formulario
with st.form("loan_data_form"):
    st.subheader("Información del Cliente")
    
    # Campo 1: age
    age = st.number_input(
        "Age (Edad)", 
        min_value=18, 
        max_value=100, 
        value=35,
        step=1
    )
    
    # Campo 2: occupation_status
    occupation_status = st.selectbox(
        "Occupation Status (Estado de Ocupación)",
        options=["Employed", "Self-Employed", "Unemployed", "Retired"],
        index=0
    )
    
    # Campo 3: years_employed
    years_employed = st.number_input(
        "Years Employed (Años Empleado)", 
        min_value=0.0, 
        max_value=50.0, 
        value=5.0,
        step=0.1,
        format="%.1f"
    )
    
    # Campo 4: annual_income
    annual_income = st.number_input(
        "Annual Income (Ingreso Anual)", 
        min_value=0, 
        max_value=10000000, 
        value=50000,
        step=1000
    )
    
    # Campo 5: credit_score
    credit_score = st.number_input(
        "Credit Score (Puntaje de Crédito)", 
        min_value=300, 
        max_value=850, 
        value=650,
        step=1
    )
    
    # Campo 6: credit_history_years
    credit_history_years = st.number_input(
        "Credit History Years (Años de Historial Crediticio)", 
        min_value=0.0, 
        max_value=50.0, 
        value=10.0,
        step=0.1,
        format="%.1f"
    )
    
    # Campo 7: savings_assets
    savings_assets = st.number_input(
        "Savings Assets (Activos en Ahorros)", 
        min_value=0, 
        max_value=10000000, 
        value=10000,
        step=500
    )
    
    # Campo 8: current_debt
    current_debt = st.number_input(
        "Current Debt (Deuda Actual)", 
        min_value=0, 
        max_value=10000000, 
        value=5000,
        step=500
    )
    
    # Campo 9: defaults_on_file
    defaults_on_file = st.number_input(
        "Defaults on File (Incumplimientos en Archivo)", 
        min_value=0, 
        max_value=100, 
        value=0,
        step=1
    )
    
    # Campo 10: delinquencies_last_2yrs
    delinquencies_last_2yrs = st.number_input(
        "Delinquencies Last 2 Years (Morosidades Últimos 2 Años)", 
        min_value=0, 
        max_value=100, 
        value=0,
        step=1
    )
    
    # Campo 11: derogatory_marks
    derogatory_marks = st.number_input(
        "Derogatory Marks (Marcas Derogatorias)", 
        min_value=0, 
        max_value=100, 
        value=0,
        step=1
    )
    
    # Campo 12: product_type
    product_type = st.selectbox(
        "Product Type (Tipo de Producto)",
        options=["Personal Loan", "Line of Credit", "Credit Card"],
        index=0
    )
    
    # Campo 13: loan_intent
    loan_intent = st.selectbox(
        "Loan Intent (Intención del Préstamo)",
        options=["Home", "Auto", "loan_intent_Education", "Business", "loan_intent_Medical", "Personal"],
        index=0
    )
    
    # Campo 14: loan_amount
    loan_amount = st.number_input(
        "Loan Amount (Monto del Préstamo)", 
        min_value=0, 
        max_value=10000000, 
        value=15000,
        step=500
    )
    
    # Campo 15: interest_rate
    interest_rate = st.number_input(
        "Interest Rate (Tasa de Interés %)", 
        min_value=0.0, 
        max_value=100.0, 
        value=7.5,
        step=0.1,
        format="%.2f"
    )
    
    # Campo 16: debt_to_income_ratio
    debt_to_income_ratio = st.number_input(
        "Debt to Income Ratio (Ratio Deuda/Ingreso)", 
        min_value=0.0, 
        max_value=10.0, 
        value=0.25,
        step=0.01,
        format="%.2f"
    )
    
    # Campo 17: loan_to_income_ratio
    loan_to_income_ratio = st.number_input(
        "Loan to Income Ratio (Ratio Préstamo/Ingreso)", 
        min_value=0.0, 
        max_value=10.0, 
        value=0.30,
        step=0.01,
        format="%.2f"
    )
    
    # Campo 18: payment_to_income_ratio
    payment_to_income_ratio = st.number_input(
        "Payment to Income Ratio (Ratio Pago/Ingreso)", 
        min_value=0.0, 
        max_value=10.0, 
        value=0.15,
        step=0.01,
        format="%.2f"
    )

    
    # Botón de envío
    submitted = st.form_submit_button("Generar Predicción")
    
    if submitted:
        # Crear un diccionario con los nombres de las columnas exactos
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
        
        
        
        # Crear el DataFrame
        df = pd.DataFrame(data_dict)
        preprocesador = Preprocesador()
        X = preprocesador.preparar_datos(df)
        # Reindexar para que coincidan EXACTAMENTE con las columnas del entrenamiento
        X = X.reindex(columns=columnas_entrenamiento, fill_value=0)
        X_transform = escalador.transform(X)
        prediccion = modelo.predict(X_transform)
        
        # Convertir X a tipos compatibles con Arrow antes de mostrarlo
        X_display = X.copy()
        for col in X_display.columns:
            if X_display[col].dtype == 'object':
                X_display[col] = X_display[col].astype(str)
            elif hasattr(X_display[col].dtype, 'numpy_dtype'):
                # Convertir tipos nullable de pandas a tipos estándar de numpy
                X_display[col] = X_display[col].astype(float) if 'float' in str(X_display[col].dtype) else X_display[col].astype(int)
        
        # Mensaje de resultado
        resultado = "Aprobado ✅" if prediccion[0] == 1 else "Rechazado ❌"
        st.success(f"Predicción generada de manera correcta: {resultado}")
        
        # Mostrar el DataFrame de forma interactiva con el nuevo parámetro
        st.subheader("Datos Procesados:")
        st.dataframe(X_display, width="stretch")
        
        # Mostrar información del DataFrame
        st.subheader("Información del DataFrame:")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Filas", X_display.shape[0])
        with col2:
            st.metric("Columnas", X_display.shape[1])
        
        # Mostrar los tipos de datos
        with st.expander("Ver tipos de datos"):
            st.text(str(X_display.dtypes))
