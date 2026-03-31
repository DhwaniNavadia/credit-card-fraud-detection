import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, classification_report

import streamlit as st

# -------------------- LOAD DATA ----------
df = pd.read_csv("creditcard.csv")

# -------------------- CHECK DATA --------
print(df.head())
print(df.shape)
print(df['Class'].value_counts())
print(df.isnull().sum())

# -------------------- HANDLE IMBALANCE ----
fraud = df[df['Class'] == 1]
normal = df[df['Class'] == 0]

normal_sample = normal.sample(n=len(fraud), random_state=42)

new_df = pd.concat([fraud, normal_sample])
new_df = new_df.sample(frac=1, random_state=42)

# ------------------ FEATURES ------------
X = new_df.drop('Class', axis=1)
y = new_df['Class']

# ------------- SPLIT --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------- MODELS --------------------

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# ------ RESULTS -----------
print("Logistic:", accuracy_score(y_test, y_pred_lr))
print("Decision Tree:", accuracy_score(y_test, y_pred_dt))
print("Random Forest:", accuracy_score(y_test, y_pred_rf))
print("KNN:", accuracy_score(y_test, y_pred_knn))

print("\nLogistic Report")
print(classification_report(y_test, y_pred_lr))

print("\nDecision Tree Report")
print(classification_report(y_test, y_pred_dt))

print("\nRandom Forest Report")
print(classification_report(y_test, y_pred_rf))

print("\nKNN Report")
print(classification_report(y_test, y_pred_knn))

# ------- STREAMLIT APP -----------

import streamlit as st
import numpy as np

# -------------------- PAGE SETTINGS --------------------
st.set_page_config(page_title="Fraud Detection", layout="centered")

# -------------------- TITLE --------------------
st.markdown("## 💳 Credit Card Fraud Detection System")
st.markdown("### Enter transaction details below:")

# -------------------- INPUT SECTION --------------------
st.markdown("#### 🧾 Transaction Info")

col1, col2 = st.columns(2)

with col1:
    time = st.number_input("Time", value=0.0)

with col2:
    amount = st.number_input("Amount", value=0.0)

st.markdown("#### 📊 PCA Features (V1 - V28)")

features = []

# Using sliders for better UI
for i in range(1, 29):
    val = st.slider(f"V{i}", -5.0, 5.0, 0.0)
    features.append(val)

# Combine input
input_data = [time] + features + [amount]

# -------------------- PREDICTION --------------------
st.markdown("---")

if st.button("🔍 Predict Transaction"):
    try:
        data = np.array(input_data).reshape(1, -1)
        result = rf.predict(data)

        st.markdown("### 🧠 Prediction Result")

        if result[0] == 1:
            st.error("⚠️ Fraudulent Transaction Detected")
        else:
            st.success("✅ Legitimate Transaction")

    except:
        st.warning("⚠️ Something went wrong. Please check inputs.")

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("⚡ Built using Machine Learning & Streamlit")
        
        
        
        

#sample input:
#0, -1.359807, -0.072781, 2.536346, 1.378155, -0.338321, 0.462388, 0.239599, 0.098698, 0.363787, 0.090794, -0.551600, -0.617801, -0.991390, -0.311169, 1.468177, -0.470401, 0.207971, 0.025791, 0.403993, 0.251412, -0.018307, 0.277838, -0.110474, 0.066928, 0.128539, -0.189115, 0.133558, -0.021053, 149.62
#200, -2.312227, 1.951992, -1.609851, 3.997906, -0.522188, -1.426545, -2.537387, 1.391657, -2.770089, -2.772272, 3.202033, -2.899907, -0.595222, -4.289254, 0.389724, -1.140747, -2.830056, -0.016822, 0.416955, 0.126911, 0.517232, -0.035049, -0.465211, 0.320198, 0.044519, 0.177840, 0.261145, 0.100000
#0, 1.191857, 0.266151, 0.166480, 0.448154, 0.060018, -0.082361, 0.078501, 0.085101, -0.255425, -0.166974, 1.612726, 1.065235, 0.489095, -0.143772, 0.635558, 0.463917, -0.114804, -0.183361, 0.407533, 0.095921, -0.567945, -0.017722, -0.311169, -0.361003, 1.373673, 1.341743, 0.000000
#0, -1.358354, -1.340163, 1.773209, 0.379780, -0.503198, 1.800499, 0.791461, 0.247676, -1.514654, 0.207643, 0.624501, 0.066084, 0.717292, -0.165946, 2.345864, -2.890083, 1.109969, -0.121359, -0.008983, 0.014724, -0.295964, 0.157491, -0.005115, -0.057752, 0.483977, 0.407533, 0.000000