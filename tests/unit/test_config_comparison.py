import json

import pandas as pd
import pytest

from clineval.comparison import compare_models
from clineval.config import load_config


def test_config_loader_validates_unknown_fields_and_values(tmp_path):
    p=tmp_path/"config.json"
    p.write_text(json.dumps({"threshold":.4,"subgroup_columns":["site"],"seed":4}))
    cfg=load_config(p)
    assert cfg.threshold==.4 and cfg.subgroup_columns==("site",) and cfg.seed==4
    p.write_text('{"unexpected": true}')
    with pytest.raises(ValueError,match="unknown configuration"):
        load_config(p)


def test_model_comparison_requires_same_outcome_for_common_records():
    frame=pd.DataFrame({"record_id":["a","b","a","b"],"model":["m1","m1","m2","m2"],
        "split":["test"]*4,"y_true":[0,1,0,1],"y_prob":[.2,.8,.3,.7]})
    result=compare_models(frame,n_resamples=100)
    assert result[0]["status"]=="descriptive_paired" and result[0]["n_common"]==2
    frame.loc[2,"y_true"]=1
    assert compare_models(frame)[0]["status"]=="invalid"

