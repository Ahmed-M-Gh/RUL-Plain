import pandas as pd
from src.config import RESIDUAL_MODELS, EXCEPTION_COLS, FIRST_COLS, CONDITION_COLS, TARGETED_COLS

def preprocessing_engine_data(raw_df:pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw engine sensor readings (row-level) and returns the 8 residual-corrected
    features ready for the final XGBoost model.
    """
    
    if not isinstance(raw_df, pd.DataFrame):
        raise TypeError("THIS APP ACCEPTS ONLY PANDAS DATAFRAME !!!!!!!!!!")
    
    required_cols = ["unit", "cycle", "alt", "Mach", "TRA"] + \
                 [col.replace("_mean", "") for col in TARGETED_COLS]

    missing_cols = [col for col in required_cols if col not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in uploaded data: {missing_cols}")
    
    # ----AGGREGATION----
    agg_cols = [col for col in raw_df.columns if col not in EXCEPTION_COLS]
    
    agg_dict = {}
    for col in agg_cols:
        agg_dict[col] = ["mean", "std"]
        
    for col in FIRST_COLS:
        if col in raw_df.columns:
            agg_dict[col] = ["first"]
    
    grouped_df = raw_df.groupby(["unit", "cycle"]).agg(agg_dict)
    
    # ----FLATTEN COLUMN NAMES----
    grouped_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in grouped_df.columns]
    grouped_df = grouped_df.reset_index()
    
    # ----RESIDUAL CORRELATION----
    X = grouped_df[CONDITION_COLS]
    
    predicted_residual = pd.DataFrame()
    for col in TARGETED_COLS:
        model = RESIDUAL_MODELS[col]
        predicted_residual[col] = model.predict(X)
        
    final_df = pd.DataFrame()
    for col in TARGETED_COLS:
        final_df[f"{col}_residual"] = grouped_df[col] - predicted_residual[col]
        
    final_df["unit"] = grouped_df["unit"]
    final_df["cycle"] = grouped_df["cycle"]
    
    return final_df