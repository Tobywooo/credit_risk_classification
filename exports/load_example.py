import json
import numpy as np
import xgboost as xgb

booster = xgb.Booster()
booster.load_model("exports/v3_searched.json")
meta = json.load(open("exports/v3_searched_metadata.json"))

FEATURES  = meta["feature_names"]          # order is mandatory
THRESHOLD = meta["decision_threshold"]

def score(record: dict) -> dict:
    """record: {feature_name: value} for all 211 features."""
    missing = set(FEATURES) - set(record)
    if missing:
        raise ValueError(f"missing {len(missing)} features, e.g. {sorted(missing)[:3]}")
    row = np.array([[record[f] for f in FEATURES]], dtype=np.float32)
    p = float(booster.predict(xgb.DMatrix(row, feature_names=FEATURES))[0])
    return {"probability": p, "flagged": p >= THRESHOLD}
