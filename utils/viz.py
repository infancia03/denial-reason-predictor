import plotly.express as px

def plot_top_cpt(df):
    top_cpt = df.groupby("cpt_code").size().reset_index(name="count").sort_values("count", ascending=False).head(10)
    if top_cpt.empty: return None
    return px.bar(top_cpt, x="cpt_code", y="count",
                  labels={"cpt_code": "CPT Code", "count": "Denial Count"},
                  title="Top 10 Denied CPT Codes")

def plot_denials_by_insurance(df):
    den_by_ins = df.groupby("insurance_company").size().reset_index(name="count").sort_values("count", ascending=False)
    if den_by_ins.empty: return None
    return px.bar(den_by_ins, x="insurance_company", y="count",
                  labels={"insurance_company": "Insurance Company", "count": "Denial Count"},
                  title="Denials by Insurance Company")

def plot_denials_by_physician(df):
    den_by_phy = df.groupby("physician").size().reset_index(name="count").sort_values("count", ascending=False)
    if den_by_phy.empty: return None
    return px.bar(den_by_phy, x="physician", y="count",
                  labels={"physician": "Physician", "count": "Denial Count"},
                  title="Denials by Physician")
