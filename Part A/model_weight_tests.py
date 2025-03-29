import random
from typing import Dict, List, Tuple

# Funciones del modelo (simplificadas para el ejemplo, usando las mismas del código anterior)
def repayment_history_score(late_payments): return 350 if late_payments == 0 else (200 if late_payments == 1 else (100 if late_payments == 2 else 0))
def total_amount_owed_score(amount_owed): return 150 if amount_owed < 1000 else (100 if amount_owed < 5000 else (75 if amount_owed < 10000 else 50))
def credit_history_length_score(years): return 50 if years < 2 else (75 if years < 5 else (100 if years < 10 else 150))
def credit_types_score(num_types): return 60 if num_types == 1 else (80 if num_types == 2 else 100)
def new_credit_score(inquiries): return 100 if inquiries == 0 else (80 if inquiries == 1 else (60 if inquiries == 2 else 40))
def available_credit_score(credit_available): return 100 if credit_available > 5000 else (80 if credit_available >= 2000 else (60 if credit_available >= 1000 else 40))
def credit_utilization_score(usage_percent): return 150 if usage_percent < 30 else (100 if usage_percent < 50 else (75 if usage_percent < 70 else 50))
def income_score(monthly_income): return 40 if monthly_income < 1000 else (60 if monthly_income < 2000 else (80 if monthly_income < 3000 else 100))
def job_tenure_score(years): return 40 if years < 1 else (60 if years < 3 else (80 if years < 5 else 100))
def open_loans_score(num_loans): return 100 if num_loans <= 1 else (80 if num_loans <= 3 else (60 if num_loans <= 5 else 40))

# Verificación de elegibilidad
def check_eligibility(age: int, monthly_income: float, loan_amount: float) -> bool:
    return age >= 18 and monthly_income >= 400 and loan_amount <= 150000

# Evaluación del solicitante con pesos ajustables
def evaluate_applicant(data: Dict[str, float], weights: Dict[str, int], threshold: int) -> Tuple[int, str]:
    if not check_eligibility(data["age"], data["monthly_income"], data["loan_amount"]):
        return None, "No elegible"
    
    total_score = (
        repayment_history_score(data["late_payments"]) * weights["repayment_history"] / 350 +
        total_amount_owed_score(data["amount_owed"]) * weights["total_amount_owed"] / 150 +
        credit_history_length_score(data["credit_age"]) * weights["credit_history_length"] / 150 +
        credit_types_score(data["credit_types"]) * weights["credit_types"] / 100 +
        new_credit_score(data["inquiries"]) * weights["new_credit"] / 100 +
        available_credit_score(data["available_credit"]) * weights["available_credit"] / 100 +
        credit_utilization_score(data["credit_usage"]) * weights["credit_utilization"] / 150 +
        income_score(data["monthly_income"]) * weights["income"] / 100 +
        job_tenure_score(data["job_tenure"]) * weights["job_tenure"] / 100 +
        open_loans_score(data["open_loans"]) * weights["open_loans"] / 100
    )
    
    decision = "Aprobado" if total_score >= threshold else "Rechazado"
    return total_score, decision

# Generar dataset simulado
def generate_test_data(n: int) -> List[Dict[str, float]]:
    data = []
    for _ in range(n):
        applicant = {
            "age": random.randint(16, 60),
            "monthly_income": random.uniform(300, 5000),
            "loan_amount": random.uniform(1000, 200000),
            "late_payments": random.randint(0, 4),
            "amount_owed": random.uniform(0, 15000),
            "credit_age": random.randint(0, 15),
            "credit_types": random.randint(1, 4),
            "inquiries": random.randint(0, 5),
            "available_credit": random.uniform(500, 10000),
            "credit_usage": random.uniform(0, 100),
            "job_tenure": random.randint(0, 10),
            "open_loans": random.randint(0, 6)
        }
        # Etiqueta esperada basada en reglas simples
        expected = "Aprobado" if (applicant["late_payments"] <= 1 and applicant["credit_usage"] < 50 and applicant["monthly_income"] >= 1000) else "Rechazado"
        if not check_eligibility(applicant["age"], applicant["monthly_income"], applicant["loan_amount"]):
            expected = "No elegible"
        applicant["expected"] = expected
        data.append(applicant)
    return data

# Función para evaluar combinaciones de pesos
def test_weights(test_data: List[Dict[str, float]], weight_configs: List[Dict[str, int]], threshold: int) -> None:
    for i, weights in enumerate(weight_configs, 1):
        print(f"\nPrueba {i} - Pesos: {weights}")
        correct = 0
        total_eligible = 0
        
        for applicant in test_data:
            score, decision = evaluate_applicant(applicant, weights, threshold)
            expected = applicant["expected"]
            
            print(f"Solicitante: Puntaje={score if score is not None else 'N/A'}, Decisión={decision}, Esperado={expected}")
            if decision == expected:
                correct += 1
            if expected != "No elegible":
                total_eligible += 1
        
        accuracy = correct / len(test_data) * 100
        print(f"Precisión total: {accuracy:.2f}% ({correct}/{len(test_data)})")
        if total_eligible > 0:
            eligible_accuracy = correct / total_eligible * 100
            print(f"Precisión entre elegibles: {eligible_accuracy:.2f}% ({correct}/{total_eligible})")

# Configuraciones de pesos para probar
weights_original = {
    "repayment_history": 350, "total_amount_owed": 150, "credit_history_length": 150,
    "credit_types": 100, "new_credit": 100, "available_credit": 100, "credit_utilization": 150,
    "income": 100, "job_tenure": 100, "open_loans": 100
}

weights_adjusted = {
    "repayment_history": 300, "total_amount_owed": 150, "credit_history_length": 150,
    "credit_types": 100, "new_credit": 100, "available_credit": 100, "credit_utilization": 200,
    "income": 120, "job_tenure": 100, "open_loans": 80  # Reducir historial, aumentar utilización e ingresos
}

# Generar datos y probar
test_data = generate_test_data(10)  # 10 solicitantes simulados
threshold = 550

print("=== Datos de Prueba Generados ===")
for i, d in enumerate(test_data, 1):
    print(f"Solicitante {i}: {d}")

test_weights(test_data, [weights_original, weights_adjusted], threshold)