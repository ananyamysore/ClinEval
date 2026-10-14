"""Machine-readable report serialization and reproducible figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

matplotlib.use("Agg")


def write_json(path: str|Path, obj: Any) -> Path:
    """Write stable, indented JSON, converting NumPy values and missing floats."""
    target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
    def clean(x):
        if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [clean(v) for v in x]
        if isinstance(x,(np.integer,)): return int(x)
        if isinstance(x,(np.floating,)): return float(x) if np.isfinite(x) else None
        if isinstance(x,(np.bool_,)): return bool(x)
        if isinstance(x,np.ndarray): return clean(x.tolist())
        if isinstance(x,(pd.Timestamp,)): return x.isoformat()
        if pd.isna(x) if not isinstance(x,(dict,list,tuple,np.ndarray)) else False: return None
        return x
    target.write_text(json.dumps(clean(obj),indent=2,sort_keys=True,allow_nan=False)+"\n")
    return target


def write_evaluation_figures(frame: pd.DataFrame, out_dir: str|Path, *, outcome="y_true",
                             probability="y_prob", split="test") -> list[str]:
    """Write ROC, PR, and reliability plots for rows selected by split."""
    root=Path(out_dir); root.mkdir(parents=True,exist_ok=True)
    data=frame[frame.split==split] if "split" in frame else frame
    if data.empty: return []
    y=data[outcome].to_numpy(dtype=int); p=data[probability].to_numpy(dtype=float)
    paths=[]
    fig,ax=plt.subplots(figsize=(5,4))
    if len(np.unique(y))==2: RocCurveDisplay.from_predictions(y,p,ax=ax)
    else: ax.text(.5,.5,"ROC curve undefined: one outcome class",ha="center")
    ax.set_title(f"ROC curve ({split}; synthetic/observed cohort)"); fig.tight_layout()
    path=root/"roc_curve.png"; fig.savefig(path,dpi=150); plt.close(fig); paths.append(str(path))
    fig,ax=plt.subplots(figsize=(5,4)); PrecisionRecallDisplay.from_predictions(y,p,ax=ax)
    ax.set_title(f"Precision-recall curve ({split})"); fig.tight_layout()
    path=root/"precision_recall.png"; fig.savefig(path,dpi=150); plt.close(fig); paths.append(str(path))
    fig,ax=plt.subplots(figsize=(5,4))
    obs,pred=calibration_curve(y,p,n_bins=min(10,max(2,len(y)//10)),strategy="quantile")
    ax.plot(pred,obs,"o-",label="Model"); ax.plot([0,1],[0,1],"--",label="Ideal")
    ax.set(xlim=(0,1),ylim=(0,1),xlabel="Mean predicted probability",ylabel="Observed fraction positive",title=f"Calibration ({split})")
    ax.legend(); fig.tight_layout(); path=root/"calibration.png"; fig.savefig(path,dpi=150); plt.close(fig); paths.append(str(path))
    return paths


def write_tables(out_dir: str|Path, *, metrics: list[dict], subgroups: list[dict], audit: list[dict],
                 comparisons: list[dict]) -> list[str]:
    """Export primary and audit tables as JSON and CSV where tabular."""
    root=Path(out_dir); root.mkdir(parents=True,exist_ok=True); paths=[]
    for name,data in (("metrics",metrics),("subgroups",subgroups),("audit",audit),("comparisons",comparisons)):
        json_path=write_json(root/f"{name}.json",data); paths.append(str(json_path))
        if data:
            csv_path=root/f"{name}.csv"
            pd.json_normalize(data).to_csv(csv_path,index=False); paths.append(str(csv_path))
    return paths

