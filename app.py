import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#Setting page - Must the first order for Streamlit
st.set_page_config(page_title="GLP-1 & Obesity Dashboard", layout="wide")

#Title & Brif
st.title("Can GLP-1 Drugs Solve the Obesity Epidemic?")
st.caption("Interactive dashboard | 5 datasets")

#Chart 1관련 
@st.cache_data
def load_obesity_global():
    df = pd.read_csv("data/obesity_global.csv")
    df.columns = ["Entity", "Code", "Year", "Obesity_Pct"]
    return df

obesity_global = load_obesity_global()

#Chart 2, 5관련
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

#Chart 3관련
@st.cache_data
def load_prescriptions():
    df = pd.read_csv("data/glp1_prescriptions.csv")
    return df

prescriptions = load_prescriptions()

#Chart 4 관련
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

        #고를 수 있는 나라/지역 목록
    options = sorted(obesity_global["Entity"].unique())

        #기본값을 "World"로 잡기 위해 위치 찾기
    default_index = options.index("World")

        #선택박스: 사용자가 여기서 하나 고름
    selected = st.selectbox("Select a country or region:", options, index=default_index)

        #고른 값으로 데이터 필터링
    filtered = obesity_global[obesity_global["Entity"] == selected]

     #Chart1 
    fig1 = px.line(filtered, x="Year", y="Obesity_Pct",
                  labels={"Obesity+Pct": "Obesity Prevalence (%)"} )

    st.plotly_chart(fig1, use_container_width=True)


        #Chart2: Family

    #Prep data
    level_order = ["Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II",
               "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]

    obesity_risk_display = obesity_risk.copy()
    obesity_risk_display["family_history_with_overweight"] = obesity_risk_display["family_history_with_overweight"].map({1: "Yes", 0: "No"})

        #count = 비만 등급별 * 가족력
    family = obesity_risk_display.groupby(["Obesity_Level", "family_history_with_overweight"]).size().reset_index(name="count")

    family["total"] = family.groupby("Obesity_Level")["count"].transform("sum")
    family["percent"] = family["count"] / family["total"] * 100

    fig2 = px.bar(family, x="Obesity_Level", y="percent",
                color="family_history_with_overweight",
                category_orders={"Obesity_Level": level_order},
                labels={"family_history_with_overweight": "Family History"})

    fig2.update_yaxes(title="Percentage (%)", ticksuffix="%")

    st.plotly_chart(fig2, use_container_width=True)


with tab2:  #Chart 3: GLP-1 spending
    st.subheader("Total Spending on GLP-1 Drugs (2018 to 2024)")

    fig3 = px.bar(prescriptions, x="Year", y="Total_Spending_Billion_USD",
                text_auto=True,
                labels={"Total_Spending_Billion_USD": "Total Spending (Billion USD)"})

    fig3.update_traces(texttemplate="$%{y}B", textposition="outside")

    st.plotly_chart(fig3, use_container_width=True)

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

#Chart 5: Model A vs Model B

#Model A에 쓸 전체 feature (including weight/height)
features_a = ["Age", "Height", "Weight", "family_history_with_overweight",
              "Freq_High_Cal_Food", "Freq_Vegetables", "Num_Meals",
              "SMOKE", "Water_Daily", "Calorie_Monitoring","Physical_Activity_Freq", "Technology_Use"]

#Model B에 쓸 Lifestyle feature (excluding weight/height)
features_b = ["Age", "family_history_with_overweight",
              "Freq_High_Cal_Food", "Freq_Vegetables", "Num_Meals",
              "SMOKE", "Water_Daily", "Calorie_Monitoring","Physical_Activity_Freq", "Technology_Use"]

##모델 학습
y = obesity_risk["Obesity_Level"]

#Model A 학습
X_a = obesity_risk[features_a]
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_a, y, test_size=0.2, random_state=42)

scaler_a = StandardScaler()
X_train_a = scaler_a.fit_transform(X_train_a)
X_test_a = scaler_a.transform(X_test_a)

model_a = LogisticRegression(max_iter=1000, random_state=42)
model_a.fit(X_train_a, y_train_a)
accuracy_a = accuracy_score(y_test_a, model_a.predict(X_test_a))

#Model B 학습
X_b = obesity_risk[features_b]
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_b, y, test_size=0.2, random_state=42)

scaler_b = StandardScaler()
X_train_b = scaler_b.fit_transform(X_train_b)
X_test_b = scaler_b.transform(X_test_b)

model_b = LogisticRegression(max_iter=1000, random_state=42)
model_b.fit(X_train_b, y_train_b)
accuracy_b = accuracy_score(y_test_b, model_b.predict(X_test_b))

with tab3: 
#두개 나란히 보여주기
    st.subheader("Predicting Obesity: All Features vs Lifestyle Only")
    col1, col2 = st.columns(2)
    col1.metric("Model A: All Features", f"{accuracy_a:.1%}")
    col2.metric("Model B: Lifestyle Only", f"{accuracy_b:.1%}")

