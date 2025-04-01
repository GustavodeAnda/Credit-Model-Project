# Modelo de Riesgo Crediticio
# Autores: [Daniel Sánchez & Ana Luisa Espinoza & Gustavo de Anda]
# Fecha: 31 de marzo de 2025
# Descripción: Modelo de riesgo que expande el scorecard tradicional de la Parte A,
# el modelo calcula métricas de riesgo como:
# - Probability of Default (PD)
# - Loss Given Default (LGD)
# - Exposure at Default (EAD)

import random
import numpy as np
from typing import Dict, Optional, Tuple, List

class Applicant:
    """Clase que representa a un solicitante de préstamo."""
    
    def __init__(self, data: Dict[str, float]):
        """Inicializa un solicitante con sus datos.

        Args:
            data (Dict[str, float]): Diccionario con datos del solicitante.
        """
        self.data = data
        self.score = None
        self.decision = None
        self.classification = None
        self.interest_rate = None
        self.max_loan = None
        self.pd = None
        self.lgd = None
        self.ead = None

    def check_eligibility(self) -> bool:
        """Verifica si el solicitante cumple con los requisitos de elegibilidad.

        Returns:
            bool: True si es elegible, False si no.
        """
        return (self.data["age"] >= 18 and 
                self.data["monthly_income"] >= 400 and 
                self.data["loan_amount"] <= 150000)

    def calculate_score(self) -> int:
        """Calcula el puntaje del solicitante basado en el scorecard.

        Returns:
            int: Puntaje total (máximo 1350).
        """
        if not self.check_eligibility():
            return None
        
        score = (
            self._repayment_history_score() +
            self._total_amount_owed_score() +
            self._credit_history_length_score() +
            self._credit_types_score() +
            self._new_credit_score() +
            self._available_credit_score() +
            self._credit_utilization_score() +
            self._income_score() +
            self._job_tenure_score() +
            self._open_loans_score()
        )
        self.score = score
        return score

    # Métodos de puntuación (basados en la Parte A)
    def _repayment_history_score(self) -> int:
        late_payments = self.data["late_payments"]
        return 350 if late_payments == 0 else (200 if late_payments == 1 else (100 if late_payments == 2 else 0))

    def _total_amount_owed_score(self) -> int:
        amount_owed = self.data["amount_owed"]
        return 150 if amount_owed < 1000 else (100 if amount_owed < 5000 else (75 if amount_owed < 10000 else 50))

    def _credit_history_length_score(self) -> int:
        years = self.data["credit_age"]
        return 50 if years < 2 else (75 if years < 5 else (100 if years < 10 else 150))

    def _credit_types_score(self) -> int:
        num_types = self.data["credit_types"]
        return 60 if num_types == 1 else (80 if num_types == 2 else 100)

    def _new_credit_score(self) -> int:
        inquiries = self.data["inquiries"]
        return 100 if inquiries == 0 else (80 if inquiries == 1 else (60 if inquiries == 2 else 40))

    def _available_credit_score(self) -> int:
        credit_available = self.data["available_credit"]
        return 100 if credit_available > 5000 else (80 if credit_available >= 2000 else (60 if credit_available >= 1000 else 40))

    def _credit_utilization_score(self) -> int:
        usage_percent = self.data["credit_usage"]
        return 150 if usage_percent < 30 else (100 if usage_percent < 50 else (75 if usage_percent < 70 else 50))

    def _income_score(self) -> int:
        monthly_income = self.data["monthly_income"]
        return 40 if monthly_income < 700 else (50 if monthly_income < 1000 else (60 if monthly_income < 2000 else (80 if monthly_income < 3000 else 100)))

    def _job_tenure_score(self) -> int:
        years = self.data["job_tenure"]
        return 40 if years < 1 else (60 if years < 3 else (80 if years < 5 else 100))

    def _open_loans_score(self) -> int:
        num_loans = self.data["open_loans"]
        return 100 if num_loans <= 1 else (80 if num_loans <= 3 else (60 if num_loans <= 5 else 40))

class CreditRiskModel:
    """Clase que implementa el modelo de riesgo crediticio."""
    
    def __init__(self, threshold: int = 873):
        """Inicializa el modelo de riesgo crediticio.

        Args:
            threshold (int, optional): Umbral para aprobar el préstamo. Defaults to 873.
        """
        self.threshold = threshold
        self.base_rate = 9.00  # Tasa base (Banxico, Parte B)
        self.inflation_premium = 3.60
        self.liquidity_premium = 1.50
        self.admin_costs = 2.40
        self.profit_margin = 3.50

    def evaluate_applicant(self, applicant: Applicant) -> Tuple[Optional[int], str, str, Optional[float], Optional[float], float, float, float]:
        """Evalúa a un solicitante y calcula métricas de riesgo.

        Args:
            applicant (Applicant): Objeto Applicant con los datos del solicitante.

        Returns:
            Tuple: (puntaje, decisión, clasificación, tasa de interés, monto máximo, PD, LGD, EAD).
        """
        # Calcular puntaje
        score = applicant.calculate_score()
        if score is None:
            applicant.decision = "No elegible"
            applicant.classification = "No elegible"
            return None, "No elegible", "No elegible", None, None, 0.0, 0.0, 0.0

        # Determinar decisión
        decision = "Aprobado" if score >= self.threshold else "Rechazado"
        applicant.decision = decision

        # Clasificar según Buró de Crédito (escalado)
        classification, interest_rate, max_loan = self._classify_credit_score(score)
        applicant.classification = classification
        applicant.interest_rate = interest_rate
        applicant.max_loan = max_loan

        # Calcular métricas de riesgo
        pd = self._calculate_pd(score)
        lgd = self._calculate_lgd(applicant.data["loan_amount"], applicant.data["late_payments"])
        ead = self._calculate_ead(applicant.data["loan_amount"], max_loan)
        applicant.pd = pd
        applicant.lgd = lgd
        applicant.ead = ead

        return score, decision, classification, interest_rate, max_loan, pd, lgd, ead

    def _classify_credit_score(self, score: int) -> Tuple[str, Optional[float], Optional[float]]:
        """Clasifica el puntaje según los rangos de Buró de Crédito (escalados).

        Args:
            score (int): Puntaje del solicitante.

        Returns:
            Tuple: (clasificación, tasa de interés, monto máximo).
        """
        if score is None:
            return "No elegible", None, None
        elif score == 0:
            return "No satisfactoria", 30.0, 0.0
        elif 715 <= score < 794:
            return "Mala", 28.0, 5000.0
        elif 794 <= score < 873:
            return "Mala", 26.0, 5000.0
        elif 873 <= score < 953:
            return "Regular", 24.0, 10000.0
        elif 953 <= score < 1032:
            return "Regular", 22.0, 10000.0
        elif 1032 <= score < 1112:
            return "Buena", 20.0, 50000.0
        elif 1112 <= score < 1191:
            return "Excelente", 18.0, 100000.0
        else:
            return "Excepcional", 15.0, 150000.0

    def _calculate_pd(self, score: int) -> float:
        """Calcula la Probabilidad de Incumplimiento (PD) basada en el puntaje.

        Args:
            score (int): Puntaje del solicitante.

        Returns:
            float: Probabilidad de incumplimiento (0-1).
        """
        # PD inversamente proporcional al puntaje
        max_score = 1350
        pd = 1 - (score / max_score)  # Ejemplo: puntaje 873 → PD = 0.35
        return max(0.05, min(pd, 0.95))  # Limitar PD entre 5% y 95%

    def _calculate_lgd(self, loan_amount: float, late_payments: int) -> float:
        """Calcula la Pérdida Dado el Incumplimiento (LGD).

        Args:
            loan_amount (float): Monto solicitado.
            late_payments (int): Número de pagos atrasados.

        Returns:
            float: Pérdida dado el incumplimiento (0-1).
        """
        # LGD base: 70% (préstamo no garantizado)
        lgd = 0.70
        # Ajustar según historial de pagos: más atrasos → mayor LGD
        lgd += 0.05 * late_payments
        return min(lgd, 0.95)  # Limitar LGD a 95%

    def _calculate_ead(self, loan_amount: float, max_loan: float) -> float:
        """Calcula la Exposición al Incumplimiento (EAD).

        Args:
            loan_amount (float): Monto solicitado.
            max_loan (float): Monto máximo permitido según clasificación.

        Returns:
            float: Exposición al incumplimiento.
        """
        return min(loan_amount, max_loan) if max_loan is not None else 0.0

# data_preparation.py
def clean_data(applicants: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Limpia los datos de los solicitantes.

    Args:
        applicants (List[Dict[str, float]]): Lista de diccionarios con datos.

    Returns:
        List[Dict[str, float]]: Datos limpios.
    """
    cleaned_data = []
    for applicant in applicants:
        cleaned_applicant = applicant.copy()
        # Imputar valores faltantes
        for key in cleaned_applicant:
            if cleaned_applicant[key] is None:
                if key in ["late_payments", "credit_types", "inquiries", "open_loans"]:
                    cleaned_applicant[key] = 0  # Modo para categóricas
                else:
                    cleaned_applicant[key] = np.median([a[key] for a in applicants if a[key] is not None])  # Mediana para numéricas
        # Limitar outliers
        cleaned_applicant["monthly_income"] = min(cleaned_applicant["monthly_income"], 10000)
        cleaned_applicant["amount_owed"] = min(cleaned_applicant["amount_owed"], 20000)
        cleaned_applicant["available_credit"] = min(cleaned_applicant["available_credit"], 15000)
        cleaned_data.append(cleaned_applicant)
    return cleaned_data

# validation.py
def generate_test_data(n: int) -> List[Dict[str, float]]:
    """Genera datos simulados para pruebas.

    Args:
        n (int): Número de solicitantes a generar.

    Returns:
        List[Dict[str, float]]: Lista de solicitantes simulados.
    """
    data = []
    for _ in range(n):
        applicant = {
            "age": random.randint(16, 60),
            "monthly_income": random.uniform(300, 12000),
            "loan_amount": random.uniform(1000, 200000),
            "late_payments": random.randint(0, 4),
            "amount_owed": random.uniform(0, 25000),
            "credit_age": random.randint(0, 15),
            "credit_types": random.randint(1, 4),
            "inquiries": random.randint(0, 5),
            "available_credit": random.uniform(500, 20000),
            "credit_usage": random.uniform(0, 100),
            "job_tenure": random.randint(0, 10),
            "open_loans": random.randint(0, 6)
        }
        expected = "Aprobado" if (
            applicant["late_payments"] <= 1 and 
            applicant["credit_usage"] < 50 and 
            applicant["monthly_income"] >= 1000 and 
            applicant["amount_owed"] < 10000 and 
            applicant["open_loans"] <= 3
        ) else "Rechazado"
        if not Applicant(applicant).check_eligibility():
            expected = "No elegible"
        applicant["expected"] = expected
        data.append(applicant)
    return data

def validate_model(model: CreditRiskModel, test_data: List[Dict[str, float]]) -> None:
    """Valida el modelo con datos simulados.

    Args:
        model (CreditRiskModel): Modelo de riesgo crediticio.
        test_data (List[Dict[str, float]]): Datos de prueba.
    """
    correct = 0
    total = len(test_data)
    eligible_count = 0
    false_positives = 0
    false_negatives = 0
    
    print("=== Resultados de la Evaluación ===")
    for i, data in enumerate(test_data, 1):
        applicant = Applicant(data)
        score, decision, classification, interest_rate, max_loan, pd, lgd, ead = model.evaluate_applicant(applicant)
        expected = data["expected"]
        
        print(f"Solicitante {i}: Puntaje={score if score is not None else 'N/A'}, Decisión={decision}, "
              f"Clasificación={classification}, Tasa={interest_rate if interest_rate is not None else 'N/A'}%, "
              f"Monto Máximo={max_loan if max_loan is not None else 'N/A'} USD, "
              f"PD={pd:.2%}, LGD={lgd:.2%}, EAD={ead:.2f} USD, Esperado={expected}")
        if decision == expected:
            correct += 1
        if expected != "No elegible":
            eligible_count += 1
            if decision == "Aprobado" and expected == "Rechazado":
                false_positives += 1
            elif decision == "Rechazado" and expected == "Aprobado":
                false_negatives += 1
    
    accuracy = correct / total * 100
    print(f"\nPrecisión total: {accuracy:.2f}% ({correct}/{total})")
    if eligible_count > 0:
        eligible_accuracy = (correct - (total - eligible_count)) / eligible_count * 100
        print(f"Precisión entre elegibles: {eligible_accuracy:.2f}%")
        print(f"Falsos positivos: {false_positives}")
        print(f"Falsos negativos: {false_negatives}")

# main.py
def main():
    """Función principal para ejecutar el modelo."""
    # Generar y limpiar datos
    test_data = generate_test_data(20)
    cleaned_data = clean_data(test_data)
    
    # Inicializar el modelo
    model = CreditRiskModel(threshold=873)
    
    # Validar el modelo
    validate_model(model, cleaned_data)

if __name__ == "__main__":
    main()
