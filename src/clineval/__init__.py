"""ClinEval: auditable evaluation for clinical and biomedical prediction models."""

from .data import DataValidationError, validate_prediction_table
from .pipeline import evaluate

__version__ = "0.1.0"
__all__ = ["DataValidationError", "evaluate", "validate_prediction_table", "__version__"]

