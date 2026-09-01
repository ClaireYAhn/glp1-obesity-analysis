import streamlit as st
import pandas as pd
import plotly.express as px

import sklearn
import xgboost

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

#Page setup - must be the first Streamlit command.
st.set_page_config(page_title="GLP-1 & Obesity Dashboard", layout="wide")

#Title & Brief
st.title("Can GLP-1 Drugs Solve the Obesity Epidemic?")
st.caption("Interactive dashboard | 5 data sources")
st.write("This dashboard explores whether GLP-1 drugs are an effective response to rising obesity.")

#Chart 1
@st.cache_data
def load_obesity_global():
    df = pd.read_csv("data/obesity_global.csv")
    df.columns = ["Entity", "Code", "Year", "Obesity_Pct"]
    return df

obesity_global = load_obesity_global()

#Chart 2 & 5
@st.cache_data
def load_obesity_risk():
    df = pd.read_csv("data/obesity_level.csv")
    df = df.rename(columns={
        "0be1dad": "Obesity_Level", 
        "FAVC": "Freq_High_Cal_Food", 
        "FCVC": "Freq_Vegetables", 
        "NCP": "Num_Meals", 
        "CAEC": "Eating_Between_Meals", 
        "CH2O": "Water_Daily",
        "SCC": "Calorie_Monitoring", 
        "FAF": "Physical_Activity_Freq", 
        "TUE": "Technology_Use", 
        "CALC": "Alcohol_Consumption",
        "MTRANS": "Transportation"
    })
    
    df["Obesity_Level"] = df["Obesity_Level"].replace("0rmal_Weight", "Normal_Weight")
    return df

obesity_risk = load_obesity_risk()

#Chart 3
@st.cache_data
def load_prescriptions():
    df = pd.read_csv("data/glp1_prescriptions.csv")
    return df

prescriptions = load_prescriptions()

#Chart 4 
@st.cache_data
def load_effectiveness():
    df = pd.read_csv("data/glp1_effectiveness.csv")
    df = df[df["Drug"] != "Zepbound"]
    df["Drug"] = df["Drug"].replace("Placeholder_Lifestyle", "Lifestyle Only")
    return df

effectiveness = load_effectiveness()

#Obesity categories in clinical order. Used by the family history chart and by the
#confusion matrix, where it makes neighbouring-category confusion show up as a
#band along the diagonal.
level_order = ["Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II",
               "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]
short_labels = ["Insufficient", "Normal", "Overweight I", "Overweight II",
                "Obesity I", "Obesity II", "Obesity III"]

tab1, tab2, tab3 = st.tabs(["The Problem", "The Response", "Prediction"])

with tab1:
    #Chart 1: Global obesity trend
    st.subheader("Obesity Rate Over Time (1980 to 2024)")

    #List of countries/regions to choose from
    options = sorted(obesity_global["Entity"].unique())

    #Find index of "World" to set as default
    default_index = options.index("World")

    #Selecbox: user picks one
    selected = st.selectbox("Select a country or region:", options, index=default_index)

    #Filter data by selected value
    filtered = obesity_global[obesity_global["Entity"] == selected]

    #Chart1 
    fig1 = px.line(filtered, x="Year", y="Obesity_Pct",
                  labels={"Obesity_Pct": "Obesity Prevalence (%)"} )

    st.plotly_chart(fig1, use_container_width=True)

    st.caption("Global obesity has more than tripled since 1980, and the trend is rising across every region.")


    #Chart2: Family

    #Prep data
    obesity_risk_display = obesity_risk.copy()
    obesity_risk_display["family_history_with_overweight"] = obesity_risk_display["family_history_with_overweight"].map({1: "Yes", 0: "No"})

        #Count by obesity level x family history
    family = obesity_risk_display.groupby(["Obesity_Level", "family_history_with_overweight"]).size().reset_index(name="count")

    family["total"] = family.groupby("Obesity_Level")["count"].transform("sum")
    family["percent"] = family["count"] / family["total"] * 100

    fig2 = px.bar(family, x="Obesity_Level", y="percent",
                color="family_history_with_overweight",
                category_orders={"Obesity_Level": level_order},
                labels={"family_history_with_overweight": "Family History"})

    fig2.update_yaxes(title="Percentage (%)", ticksuffix="%")

    st.plotly_chart(fig2, use_container_width=True)
    st.caption("The share reporting a family history of overweight climbs steeply across obesity levels. Family history ranks among the strongest predictors in every model built here, though this is an association in survey data and does not establish a cause.")



with tab2:  #Chart 3: GLP-1 spending
    st.subheader("Total Spending on GLP-1 Drugs (2018 to 2024)")

    fig3 = px.bar(prescriptions, x="Year", y="Total_Spending_Billion_USD",
                text_auto=True,
                labels={"Total_Spending_Billion_USD": "Total Spending (Billion USD)"})

    fig3.update_traces(texttemplate="$%{y}B", textposition="outside")

    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Spending on GLP-1 drugs grew from $2.6B to $52B in six years, signalling explosive market demand.")

    #Chart 4: compare effectiveness
    st.subheader("GLP-1 Drug Effectiveness: Average Weight Loss")

    effectiveness = effectiveness.sort_values("Avg_Weight_Loss_Percent")

    fig4 = px.bar(effectiveness, x="Avg_Weight_Loss_Percent", y="Drug",
                color="Type", orientation="h",
                text_auto=True,
                labels={"Avg_Weight_Loss_Percent": "Average Weight Loss (%)",
                        "Drug": "", "Type": "Treatment Type"})
    fig4.update_traces(texttemplate="%{x}%", textposition="outside")

    st.plotly_chart(fig4, use_container_width=True)
    st.caption("GLP-1 drugs far outperform lifestyle changes alone: Mounjaro achieves 20.9% weight loss versus 2.4% for lifestyle only.")

#Chart 5: Model comparison, all features against lifestyle only

# All features for Model A (including weight/height)
features_a = ["Age", "Height", "Weight", "family_history_with_overweight",
              "Freq_High_Cal_Food", "Freq_Vegetables", "Num_Meals",
              "SMOKE", "Water_Daily", "Calorie_Monitoring","Physical_Activity_Freq", "Technology_Use"]

# Lifestyle features for Model B (excluding weight/height)
features_b = ["Age", "family_history_with_overweight",
              "Freq_High_Cal_Food", "Freq_Vegetables", "Num_Meals",
              "SMOKE", "Water_Daily", "Calorie_Monitoring","Physical_Activity_Freq", "Technology_Use"]


@st.cache_data(show_spinner="Training models on first load...")
def train_models():
    """Train Model A and the three Model B variants once, then cache the results.

    Streamlit reruns the whole script on every interaction, so training without a
    cache would retrain on each click. Cross-validation is deliberately left out
    here: the notebook settles those figures once, and they are quoted as fixed
    text below rather than recomputed on every cold start.
    """
    df = load_obesity_risk()
    y = df["Obesity_Level"]

    # Model A keeps weight and height, which is exactly what makes it leak.
    X_a = df[features_a]
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
        X_a, y, test_size=0.2, random_state=42)
    scaler_a = StandardScaler()
    model_a = LogisticRegression(max_iter=1000, random_state=42)
    model_a.fit(scaler_a.fit_transform(X_train_a), y_train_a)
    pred_a = model_a.predict(scaler_a.transform(X_test_a))

    # Model B uses the same seed and split size, so the same rows land in both
    # test sets and the two models stay directly comparable.
    X_b = df[features_b]
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_b, y, test_size=0.2, random_state=42)

    # XGBoost needs integer class labels, and one encoder keeps all three models
    # scored against identical targets.
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_b)
    y_test_enc = le.transform(y_test_b)

    # The scaler sits inside the pipeline for logistic regression. Trees split on
    # thresholds, so rescaling makes no difference to them.
    logistic = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=1000, random_state=42))])
    forest = RandomForestClassifier(n_estimators=300, random_state=42)
    boosted = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                            tree_method="hist", random_state=42)
    for model in (logistic, forest, boosted):
        model.fit(X_train_b, y_train_enc)

    pred_xgb = boosted.predict(X_test_b)

    # Obesity Type III carries a data-generation fingerprint, so both models are
    # also scored with that class removed. Same test rows, so this stays fair.
    keep = y_test_enc != list(le.classes_).index("Obesity_Type_III")

    order_idx = [list(le.classes_).index(c) for c in level_order]
    cm = confusion_matrix(y_test_enc, pred_xgb, labels=order_idx)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    report = classification_report(y_test_enc, pred_xgb, labels=order_idx,
                                   target_names=level_order, output_dict=True,
                                   zero_division=0)
    per_class = (pd.DataFrame(report).T.loc[level_order,
                 ["precision", "recall", "f1-score"]].round(3))

    return {
        "a": accuracy_score(y_test_a, pred_a),
        "a_adjusted": accuracy_score(y_test_a[keep], pred_a[keep]),
        "b_logistic": accuracy_score(y_test_enc, logistic.predict(X_test_b)),
        "b_forest": accuracy_score(y_test_enc, forest.predict(X_test_b)),
        "b_xgb": accuracy_score(y_test_enc, pred_xgb),
        "b_xgb_adjusted": accuracy_score(y_test_enc[keep], pred_xgb[keep]),
        "cm_pct": cm_pct,
        "per_class": per_class,
    }


scores = train_models()

with tab3:
    st.subheader("Predicting Obesity Level: All Features against Lifestyle Only")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model A: all features", f"{scores['a']:.1%}")
    col2.metric("Model B: logistic", f"{scores['b_logistic']:.1%}")
    col3.metric("Model B: XGBoost", f"{scores['b_xgb']:.1%}")
    col4.metric("Model B: XGBoost, adjusted", f"{scores['b_xgb_adjusted']:.1%}",
                help="Accuracy once Obesity Type III is excluded. See caveat 1 below.")

    st.caption(
        f"Model A includes weight and height, so its {scores['a']:.1%} largely reflects how obesity "
        f"is defined. Model B drops both columns. Under logistic regression it reaches only "
        f"{scores['b_logistic']:.1%}, and that figure turned out to be a property of the model: "
        f"random forest reaches {scores['b_forest']:.1%} on the same features and XGBoost reaches "
        f"{scores['b_xgb']:.1%}, falling to {scores['b_xgb_adjusted']:.1%} once one artefact-driven "
        "class is set aside. Five-fold cross-validation in the notebook confirms the pattern, at "
        "52.0% +/- 0.3 for logistic regression against 71.6% +/- 0.3 for XGBoost. The majority-class "
        "floor sits at 19.5%."
    )

    with st.expander("Where the model succeeds and where it fails"):
        st.write(
            "Row-normalised confusion matrix for XGBoost on the held-out test set. Each row shows "
            "where that true category ended up, so the diagonal is per-class recall. Categories run "
            "in clinical order, which turns confusion between neighbouring categories into a band "
            "along the diagonal."
        )

        fig5 = px.imshow(scores["cm_pct"], x=short_labels, y=short_labels,
                         color_continuous_scale="Blues", zmin=0, zmax=100,
                         text_auto=".0f", aspect="auto",
                         labels={"x": "Predicted category", "y": "True category",
                                 "color": "% of true class"})
        st.plotly_chart(fig5, use_container_width=True)

        st.dataframe(scores["per_class"], use_container_width=True)

        st.caption(
            "The model holds up at the severe end, where GLP-1 prescription criteria sit, and "
            "weakens badly through the middle. Obesity Type II reaches an F1 of 0.89, while "
            "Overweight Level I falls to 0.42. Most errors are near-misses, with 51% landing in an "
            "adjacent severity category. The middle of the scale is where BMI cutoffs of 25 and "
            "27.5 draw thin lines through genuinely similar lifestyle profiles."
        )

    st.markdown("#### Two separate caveats")
    st.write(
        "These are different problems and are easy to confuse. The first concerns how the data was "
        "generated. The second concerns which columns go into a model."
    )

    cav1, cav2 = st.columns(2)
    with cav1:
        st.warning(
            "**1. The dataset is partly synthetic.** This is about how the data was made. The Kaggle "
            "data was largely expanded with SMOTE. Inside Obesity Type III, six of the ten lifestyle "
            f"features are constant or near-constant and 99.9% of the class is female, so the model "
            f"recognises it at 98.6% recall by matching a generation artefact. Excluding that class "
            f"drops accuracy from {scores['b_xgb']:.1%} to {scores['b_xgb_adjusted']:.1%}, which is "
            "the figure to trust."
        )
    with cav2:
        st.info(
            "**2. Model A leaks its target.** This is about which columns go in, and it has no "
            "connection to caveat 1. Weight and height mechanically determine BMI, and BMI defines "
            f"the categories being predicted, so Model A's {scores['a']:.1%} measures arithmetic. It "
            "exists to make the contrast visible. Dropping Obesity Type III from both models gives "
            f"{scores['a_adjusted']:.1%} for Model A against {scores['b_xgb_adjusted']:.1%} for "
            "Model B, a gap of 17.2 points."
        )

st.caption("Data sources: WHO Global Health Observatory, Kaggle Obesity Risk Dataset (partly synthetic), "
           "STEP & SURMOUNT clinical trials, FAIR Health, and KFF (Kaiser Family Foundation).")

#Printed so that a library upgrade shows up on the page instead of quietly moving
#the accuracy figures. Both versions are pinned in requirements.txt.
st.caption(f"Model figures trained live with xgboost {xgboost.__version__} and "
           f"scikit-learn {sklearn.__version__}. These versions are pinned so the app "
           "reproduces the notebook exactly, at 71.6% for XGBoost and 65.1% adjusted.")

