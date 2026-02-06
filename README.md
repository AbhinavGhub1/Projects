# NBA CLASSIFICATION
NBA CLASSIFICATION is a ML project that helps coaches or managers to manages their players based on their previous performances. It acts like a virtual manager and manages players position according to the performance or stats of the player. Modules used in this are scikit-learn, numpy and pandas. 


import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="NBA Classification",
    layout="wide",
    page_icon="🏀",
)

st.markdown("""
<style>
    .title {
        font-size:40px !important;
        color:#FF4B4B;
        text-align:center;
        font-weight:700;
    }
    .sub {
        font-size:20px !important;
        text-align:center;
        color:#999;
    }
    .box {
        padding:20px;
        border-radius:15px;
        background-color:#1E1E1E;
        border:1px solid #444;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🏀 NBA Classification Web App</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Machine Learning Based NBA Performance Predictor</div><br>", unsafe_allow_html=True)

df = pd.read_csv("nba_classification_realistic.csv")

st.sidebar.header("📂 Dataset Controls")
if st.sidebar.checkbox("Show Dataset Preview"):
    st.write(df.head())

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

model = KNeighborsClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

st.sidebar.header("🎮 Input Player Stats")
user_inputs = {}

for col in X.columns:
    if df[col].dtype in ["int64", "float64"]:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        default_val = float(df[col].mean())

        val = st.sidebar.slider(
            label=f"{col}",
            min_value=min_val,
            max_value=max_val,
            value=default_val
        )
        user_inputs[col] = val

user_df = pd.DataFrame([user_inputs])
user_df_scaled = scaler.transform(user_df)

st.markdown("### 🔍 Model Accuracy")
st.metric("Accuracy", f"{acc*100:.2f}%")

st.markdown("### 🎯 Predict Outcome Based on Inputs")

if st.button("Predict Now"):
    result = model.predict(user_df_scaled)[0]

    st.markdown(f"""
    <div class='box'>
        <h3 style='color:#FFDD00;'>Prediction Result: {result}</h3>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><hr><center>Developed with ❤️ using Streamlit</center>", unsafe_allow_html=True)

