import pandas as pd

class Preprocesador:
    """Simplifica los pasos de limpieza y transformación."""

    def preparar_datos(self, datos):

        # Copia de seguridad
        df = datos.copy()

        if 'customer_id' in df.columns:
            df.drop('customer_id', axis=1, inplace=True)
        bins_credit = [300, 640, 740, 850]
        labels_credit = ['Riesgo Alto', 'Riesgo Medio', 'Riesgo Bajo'] # Riesgo Bajo = Buen Puntaje
        df['credit_score_rango2'] = pd.cut( #pd.cut(): Esta es la función clave de pandas.
            df['credit_score'],
            bins=bins_credit, #bins: Una lista de los puntos de corte.
            labels=labels_credit, #labels: Una lista de nombres para las nuevas categorías.       empieza
            right=True,
            include_lowest=True)


        bins_debt_to_income = [0, 0.36, 0.43,10]
        labels_debt_to_income = ['Bajo Riesgo', 'Medio Riesgo', 'Alto Riesgo']
        df['debt_to_income_rango'] = pd.cut(
            df['debt_to_income_ratio'],
            bins=bins_debt_to_income,
            labels=labels_debt_to_income,
            right=True,
            include_lowest=True)
        
        bins_payment_to_income = [0, 0.20, 0.35, 10]
        labels_payment_to_income = ['Bajo Riesgo', 'Medio Riesgo', 'Alto Riesgo']
        df['payment_to_income_rango'] = pd.cut(
        df['payment_to_income_ratio'],
        bins=bins_payment_to_income,
        labels=labels_payment_to_income,
        right=True,
        include_lowest=True)

        df = pd.get_dummies(df, columns=['credit_score_rango2'], prefix='credit_score_rango2', dtype=int)

        df = pd.get_dummies(df, columns=['occupation_status'], prefix='occupation_status', dtype=int)

        df = pd.get_dummies(df, columns=['product_type'], prefix='product_type', dtype=int)
        df = pd.get_dummies(df, columns=['loan_intent'], prefix='loan_intent', dtype=int)


        df.drop(['credit_score','debt_to_income_rango', 'payment_to_income_rango', 'annual_income', 'loan_amount', 'debt_to_income_ratio', 'payment_to_income_ratio'], axis=1, inplace = True)
        
        if 'loan_status' in df.columns:
            y = df['loan_status']
            X = df.drop('loan_status', axis=1)
            return X,y
        else:
            return df