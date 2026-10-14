import numpy as np
import pytest

from clineval.metrics import classification_metrics
from clineval.thresholds import threshold_for_target_specificity
from clineval.uncertainty import bootstrap_metric_ci


def test_binary_metrics_known_perfect_predictions_and_undefined_values():
    result=classification_metrics([0,0,1,1],[.1,.2,.8,.9],threshold=.5)
    assert result["roc_auc"]==1 and result["average_precision"]==1
    assert result["sensitivity"]==1 and result["specificity"]==1
    assert result["confusion_matrix"]=={"tn":2,"fp":0,"fn":0,"tp":2}
    one_class=classification_metrics([0,0],[.1,.9],threshold=.5)
    assert one_class["roc_auc"] is None and one_class["sensitivity"] is None
    assert one_class["undefined"]["roc_auc"] is True


def test_bad_metrics_and_validation_only_threshold_selection():
    with pytest.raises(ValueError): classification_metrics([0,1],[np.nan,.5])
    with pytest.raises(ValueError): classification_metrics([0,2],[.1,.2])
    selected=threshold_for_target_specificity([0,0,1,1],[.1,.4,.5,.9],.9)
    assert selected["status"]=="ok" and selected["specificity"]>=.9


def test_cluster_bootstrap_uses_independent_patient_units():
    ci=bootstrap_metric_ci([0,1,0,1],[.1,.8,.2,.9],metric="brier_score",groups=["a","a","b","b"],n_resamples=200,seed=9)
    assert ci["n_independent_units"]==2 and ci["unit_method"]=="cluster_bootstrap"

