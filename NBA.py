import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ---------------------- PAGE SETUP ----------------------
st.set_page_config(page_title="NBA Classification", layout="wide")

# ---------------------- LOAD DATA ----------------------
df = pd.read_csv("nba_classification_realistic.csv")

# Encode categorical features
label_encoders = {}
for col in df.columns:
    if df[col].dtype == "object":     # Identify columns with strings
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le      # Store encoder for later use

# ---------------------- ML SETUP ----------------------
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

model = KNeighborsClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

# UI 
st.title("🏀 NBA Classification (Fixed Version)")
st.write("Now supports categorical columns like 'SG', 'PG', team names, etc.")

st.sidebar.header("Input Player Features")
user_inputs = {}

for col in X.columns:
    if col in label_encoders:  
        # Categorical column → dropdown
        options = list(label_encoders[col].classes_)
        selected = st.sidebar.selectbox(f"{col}", options)
        encoded_value = label_encoders[col].transform([selected])[0]
        user_inputs[col] = encoded_value
    else:
        # Numeric → slider
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        default_val = float(df[col].mean())
        user_inputs[col] = st.sidebar.slider(col, min_val, max_val, default_val)


# Convert to DataFrame
user_df = pd.DataFrame([user_inputs])

# Scale
user_scaled = scaler.transform(user_df)



#PREDICT 
if st.button("Predict"):
    result = model.predict(user_scaled)[0]

    # If target column was categorical, decode it
    target_col = df.columns[-1]
    if target_col in label_encoders:
        result = label_encoders[target_col].inverse_transform([result])[0]

    st.success(f"Prediction: {result}")

# accuracy
st.metric("Model Accuracy", f"{acc*100:.2f}%")



