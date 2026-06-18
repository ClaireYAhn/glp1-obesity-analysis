import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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
    level_order = ["Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II",
               "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]

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
    st.caption("Family history rises sharply with obesity level, suggesting biological or inherited factors that lifestyle alone cannot explain.")



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

#Chart 5: Model A vs Model B

# All features for Model A (including weight/height)
features_a = ["Age", "Height", "Weight", "family_history_with_overweight",
              "Freq_High_Cal_Food", "Freq_Vegetables", "Num_Meals",
              "SMOKE", "Water_Daily", "Calorie_Monitoring","Physical_Activity_Freq", "Technology_Use"]

# Lifestyle features for Model B (excluding weight/height)
features_b = ["Age", "family_history_with_overweight",
              "Freq_High_Cal_Food", "Freq_Vegetables", "Num_Meals",
              "SMOKE", "Water_Daily", "Calorie_Monitoring","Physical_Activity_Freq", "Technology_Use"]

# Train models
y = obesity_risk["Obesity_Level"]

# Train Model A
X_a = obesity_risk[features_a]
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_a, y, test_size=0.2, random_state=42)

scaler_a = StandardScaler()
X_train_a = scaler_a.fit_transform(X_train_a)
X_test_a = scaler_a.transform(X_test_a)

model_a = LogisticRegression(max_iter=1000, random_state=42)
model_a.fit(X_train_a, y_train_a)
accuracy_a = accuracy_score(y_test_a, model_a.predict(X_test_a))

# Train Model B
X_b = obesity_risk[features_b]
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_b, y, test_size=0.2, random_state=42)

scaler_b = StandardScaler()
X_train_b = scaler_b.fit_transform(X_train_b)
X_test_b = scaler_b.transform(X_test_b)

model_b = LogisticRegression(max_iter=1000, random_state=42)
model_b.fit(X_train_b, y_train_b)
accuracy_b = accuracy_score(y_test_b, model_b.predict(X_test_b))

with tab3: 
#Show the two side by side
    st.subheader("Predicting Obesity: All Features vs Lifestyle Only")
    col1, col2 = st.columns(2)
    col1.metric("Model A: All Features", f"{accuracy_a:.1%}")
    col2.metric("Model B: Lifestyle Only", f"{accuracy_b:.1%}")

    st.caption("Model A includes weight and height, so its high accuracy partly reflects how obesity is defined (BMI). "
                "Model B uses only lifestyle factors and drops to 52.5%, showing lifestyle alone cannot predict obesity well.")
    
st.caption("Data sources: WHO Global Health Observatory, Kaggle Obesity Risk Dataset, "
           "STEP & SURMOUNT clinical trials, FAIR Health, and KFF (Kaiser Family Foundation).")