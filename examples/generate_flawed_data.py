"""Generate valid and intentionally flawed audit fixtures as local CSV/JSON files."""

from pathlib import Path
import json

from clineval.synthetic import generate_synthetic_predictions

root=Path("data/synthetic"); root.mkdir(parents=True,exist_ok=True)
for name,defects in (("valid",set()),("flawed",{"patient_overlap","temporal_leakage","invalid_probabilities","missing_labels","test_threshold","target_leakage","duplicate_records"})):
    frame,metadata=generate_synthetic_predictions(seed=5,defects=defects)
    frame.to_csv(root/f"{name}.csv",index=False)
    (root/f"{name}.metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
print(f"Generated synthetic audit fixtures under {root.resolve()}; not real clinical data.")

