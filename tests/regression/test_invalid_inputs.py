import numpy as np
import pandas as pd
import pytest

from clineval.data import DataValidationError, validate_prediction_table


def test_validator_reports_nan_probability_and_missing_outcome():
    frame=pd.DataFrame({"y_true":[1,None],"y_prob":[np.nan,.4]})
    with pytest.raises(DataValidationError) as err:
        validate_prediction_table(frame)
    assert {x.code for x in err.value.issues}=={"missing_outcome","invalid_probability"}


def test_validator_flags_duplicate_record_identifier():
    frame=pd.DataFrame({"y_true":[0,1],"y_prob":[.1,.8],"record_id":["a","a"]})
    with pytest.raises(DataValidationError,match="duplicate_record_id"):
        validate_prediction_table(frame)

