import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from utils.data import currency_to_float

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def train_and_predict(df: pd.DataFrame):
    df = df.copy()

    # Clean text columns
    for c in ["cpt_code", "insurance_company", "physician", "denial_reason"]:
        df[c] = df[c].where(pd.notna(df[c]), None).apply(lambda v: v.strip() if isinstance(v, str) else v)

    # Clean numeric columns
    for c in ["payment", "balance"]:
        df[c] = df[c].apply(currency_to_float).fillna(df[c].median())

    # Fill categorical nulls
    for c in ["cpt_code", "insurance_company", "physician"]:
        df[c] = df[c].fillna("Unknown").astype(str)

    # Split labeled vs unlabeled
    missing_mask = df["denial_reason"].isna() | df["denial_reason"].astype(str).str.strip().str.lower().isin(["", "none", "nan"])
    labeled_df, unlabeled_df = df.loc[~missing_mask].copy(), df.loc[missing_mask].copy()

    if labeled_df.empty:
        raise ValueError("No labeled Denial Reason rows found.")

    y_le = LabelEncoder()
    y = y_le.fit_transform(labeled_df["denial_reason"].astype(str))
    features_cat, features_num = ["cpt_code", "insurance_company", "physician"], ["payment", "balance"]

    pre = ColumnTransformer([("cat", make_ohe(), features_cat), ("num", "passthrough", features_num)])
    clf = LogisticRegression(solver="saga", multi_class="multinomial", class_weight="balanced",
                             max_iter=2000, random_state=42)
    pipe = Pipeline([("prep", pre), ("clf", clf)])

    metrics = {"accuracy": None, "classification_report": None}
    if labeled_df["denial_reason"].nunique() >= 2 and len(labeled_df) >= 5:
        X_train, X_test, y_train, y_test = train_test_split(
            labeled_df[features_cat + features_num], y, test_size=0.2, stratify=y, random_state=42)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["classification_report"] = classification_report(
            y_test, y_pred, zero_division=0, target_names=y_le.classes_, output_dict=True)

    # Train final model
    pipe.fit(labeled_df[features_cat + features_num], y)

    # Predict missing
    if len(unlabeled_df) > 0:
        y_pred_u = pipe.predict(unlabeled_df[features_cat + features_num])
        unlabeled_df["predicted_denial_reason"] = y_le.inverse_transform(y_pred_u)
        df.loc[unlabeled_df.index, "predicted_denial_reason"] = unlabeled_df["predicted_denial_reason"]
        df["denial_reason_final"] = df["denial_reason"].where(~missing_mask, df["predicted_denial_reason"])
    else:
        df["predicted_denial_reason"], df["denial_reason_final"] = None, df["denial_reason"]

    return df, metrics, labeled_df.shape[0], unlabeled_df.shape[0]
