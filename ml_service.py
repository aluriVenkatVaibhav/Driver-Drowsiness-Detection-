from datetime import datetime
from typing import Dict

from ml.sepsis_detector import SepsisDetector
from ml.models_factory import build_population_if, build_random_forest
from ml.vitals_types import VitalsSample

# -----------------------------
# Global model initialization
# -----------------------------
population_if = build_population_if()
rf_model = build_random_forest()

# Store detector per patient
detectors: Dict[int, SepsisDetector] = {}

# Track baseline progress
baseline_counts: Dict[int, int] = {}


def get_detector(patient_id: int) -> SepsisDetector:
    if patient_id not in detectors:
        detectors[patient_id] = SepsisDetector(population_if, rf_model)
        baseline_counts[patient_id] = 0
    return detectors[patient_id]


def process_vitals(data):
    """
    Main ML entry point
    """

    detector = get_detector(data.patient_id)

    sample = VitalsSample(
        timestamp=datetime.utcnow(),
        hr=data.hr,
        rr=data.rr,
        spo2=data.spo2,
        temp=data.temp,
        movement=data.movement,
        hrv=data.hrv or 0.0,
        rrv=data.rrv or 0.0
    )

    # -----------------------------
    # Phase 1: Baseline
    # -----------------------------
    if not detector.baseline_locked:
        result = detector.add_baseline_window(sample)
        baseline_counts[data.patient_id] += 1

        if result is None:
            return {
                "phase": "BASELINE",
                "windows_left": max(0, 5 - baseline_counts[data.patient_id])
            }

        return {
            "phase": "BASELINE_COMPLETE",
            "confidence": result.confidence,
            "mode": result.mode
        }

    # -----------------------------
    # Phase 2: Monitoring
    # -----------------------------
    output = detector.process_monitoring_window(sample)

    return {
        "phase": "MONITORING",
        "status": output["status"],
        "score": output["final_score"],
        "sepsis_phase": output["sepsis_phase"],
        "full_output": output
    }