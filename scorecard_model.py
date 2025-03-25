# Modelo Tradicional de Credit Scoring para Préstamos Personales
# Autores: [Daniel Sánchez & Ana Luisa Espinoza & Gustavo de Anda]
# Fecha: 25 de marzo de 2025
# Descripción: Este script implementa un scorecard para evaluar solicitantes de préstamos personales,
# basado en modelos como FICO y VantageScore.

def payment_history_score(late_payments):
    """Asigna puntos según el historial de pagos (máximo 350 puntos).
    
    Args:
        late_payments (int): Número de pagos atrasados en los últimos 12 meses.
    
    Returns:
        int: Puntaje asignado.
    """
    if late_payments == 0:
        return 350
    elif late_payments == 1:
        return 200
    elif late_payments == 2:
        return 100
    else:  # 3 o más
        return 0

def credit_utilization_score(usage_percent):
    """Asigna puntos según la utilización de crédito (máximo 300 puntos).
    
    Args:
        usage_percent (float): Porcentaje de crédito usado respecto al límite.
    
    Returns:
        int: Puntaje asignado.
    """
    if usage_percent < 30:
        return 300
    elif 30 <= usage_percent < 50:
        return 200
    elif 50 <= usage_percent < 70:
        return 100
    else:  # >= 70%
        return 50

def credit_age_score(years):
    """Asigna puntos según la antigüedad crediticia (máximo 150 puntos).
    
    Args:
        years (int): Años desde la cuenta más antigua.
    
    Returns:
        int: Puntaje asignado.
    """
    if years < 2:
        return 50
    elif 2 <= years < 5:
        return 75
    elif 5 <= years < 10:
        return 100
    else:  # >= 10
        return 150

def new_credit_score(inquiries):
    """Asigna puntos según consultas en Buró de Crédito en 6 meses (máximo 100 puntos).
    
    Args:
        inquiries (int): Número de consultas recientes.
    
    Returns:
        int: Puntaje asignado.
    """
    if inquiries == 0:
        return 100
    elif inquiries == 1:
        return 80
    elif inquiries == 2:
        return 60
    else:  # 3 o más
        return 40

def credit_mix_score(account_types):
    """Asigna puntos según la mezcla de créditos (máximo 100 puntos).
    
    Args:
        account_types (int): Número de tipos de cuentas (tarjetas, préstamos, etc.).
    
    Returns:
        int: Puntaje asignado.
    """
    if account_types == 1:
        return 60
    elif account_types == 2:
        return 80
    else:  # 3 o más
        return 100

def income_score(monthly_income):
    """Asigna puntos según ingresos mensuales en USD (máximo 100 puntos).
    
    Args:
        monthly_income (float): Ingresos mensuales del solicitante.
    
    Returns:
        int: Puntaje asignado.
    """
    if monthly_income < 1000:
        return 40
    elif 1000 <= monthly_income < 2000:
        return 60
    elif 2000 <= monthly_income < 3000:
        return 80
    else:  # >= 3000
        return 100

def job_tenure_score(years):
    """Asigna puntos según antigüedad laboral (máximo 100 puntos).
    
    Args:
        years (int): Años en el empleo actual.
    
    Returns:
        int: Puntaje asignado.
    """
    if years < 1:
        return 40
    elif 1 <= years < 3:
        return 60
    elif 3 <= years < 5:
        return 80
    else:  # >= 5
        return 100

def open_loans_score(num_loans):
    """Asigna puntos según cantidad de préstamos abiertos (máximo 100 puntos).
    
    Args:
        num_loans (int): Número de préstamos activos.
    
    Returns:
        int: Puntaje asignado.
    """
    if num_loans <= 1:
        return 100
    elif 2 <= num_loans <= 3:
        return 80
    elif 4 <= num_loans <= 5:
        return 60
    else:  # > 5
        return 40

def evaluate_applicant(data):
    """Evalúa al solicitante y calcula el puntaje total.
    
    Args:
        data (dict): Diccionario con datos del solicitante.
    
    Returns:
        tuple: Puntaje total y decisión (Aprobado/Rechazado).
    """
    total_score = (
        payment_history_score(data["late_payments"]) +
        credit_utilization_score(data["credit_usage"]) +
        credit_age_score(data["credit_age"]) +
        new_credit_score(data["inquiries"]) +
        credit_mix_score(data["account_types"]) +
        income_score(data["monthly_income"]) +
        job_tenure_score(data["job_tenure"]) +
        open_loans_score(data["open_loans"])
    )
    
    threshold = 500  # Umbral de aprobación, ajustable según riesgo
    decision = "Aprobado" if total_score >= threshold else "Rechazado"
    return total_score, decision

# Ejemplo de uso con un solicitante ficticio
applicant_data = {
    "late_payments": 0,      # Sin pagos atrasados
    "credit_usage": 25,      # 25% de utilización de crédito
    "credit_age": 7,         # 7 años de historial crediticio
    "inquiries": 1,          # 1 consulta en Buró de Crédito
    "account_types": 3,      # 3 tipos de cuentas
    "monthly_income": 2500,  # Ingresos mensuales de 2500 USD
    "job_tenure": 4,         # 4 años en el empleo actual
    "open_loans": 2          # 2 préstamos abiertos
}

score, decision = evaluate_applicant(applicant_data)
print(f"Puntaje total: {score} / 1200")
print(f"Decisión: {decision}")
