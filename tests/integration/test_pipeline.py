import json

from clineval.pipeline import evaluate
from clineval.synthetic import generate_synthetic_predictions


def test_synthetic_pipeline_exports_report_and_heldout_metrics(tmp_path):
    frame,metadata=generate_synthetic_predictions(seed=11,n_patients=120,repeated_measurements=2,miscalibrated=True)
    metadata["synthetic"]=True
    result=evaluate(frame,tmp_path,metadata=metadata,target_specificity=.85,subgroup_columns=["subgroup"],n_bootstrap=100,seed=11)
    saved=json.loads((tmp_path/"report.json").read_text())
    assert saved["synthetic_data"] is True
    assert result["models"][0]["threshold_selection"]["selection_partition"]=="validation"
    assert result["models"][0]["test_metrics"]["n"]>0
    for name in ("audit.json","metrics.csv","subgroups.csv","roc_curve.png","precision_recall.png","calibration.png","report.md"):
        assert (tmp_path/name).exists()
    assert "SYNTHETIC EXAMPLE" in (tmp_path/"report.md").read_text()

