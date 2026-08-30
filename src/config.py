import pickle as pk
import os

# Build a path relative to this file's location, not the caller's location.
# This works no matter where config.py is imported from (notebooks/, root, etc.)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

# Cols excluded from residual/mean-std aggregation
EXCEPTION_COLS = ["unit", "cycle", "Fc", "hs",]
FIRST_COLS = ["Fc", "hs",]

# The 8 sensors the final model was trained on (in this exact order)
TARGETED_COLS = ["T50_mean", "T48_mean", "T40_mean", "phi_mean",
                  "SmLPC_mean", "SmHPC_mean", "Ps30_mean", "P15_mean"]

CONDITION_COLS = ["alt_mean", "Mach_mean", "TRA_mean"]


# Loading preprocessing artifacts (residual models only — Fc encoder not used
# by the final model, kept here only in case it's needed for a future experiment)
with open(os.path.join(MODEL_DIR, "preprocessing_artifacts.pkl"), "rb") as f:
    Preprocessing_artifacts = pk.load(f)
    
RESIDUAL_MODELS = Preprocessing_artifacts["residual_models"]

# Loading final model
with open(os.path.join(MODEL_DIR, "XGBoost_finetuned.pkl"), "rb") as m:
    MODEL = pk.load(m)
    
    
