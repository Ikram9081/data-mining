import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

st.set_page_config(layout="wide")

# Sidebar
menu = st.sidebar.selectbox("Pilih Halaman", ["📊 Klasifikasi", "🧩 Clustering"])
st.title("Aplikasi Analisis Diabetes")
st.markdown("Nama: **Muhammad ikram**  \nNIM: **22146007**")

# Load data
df = pd.read_csv("diabetes.csv")

if menu == "📊 Klasifikasi":
    st.subheader("Klasifikasi Diabetes (KNN)")

    # Data preprocessing
    fitur = df.drop(columns="Outcome")
    target = df["Outcome"]
    fitur.replace(0, np.nan, inplace=True)
    fitur.fillna(fitur.median(), inplace=True)

    # Split dan scaling
    X_train, X_test, y_train, y_test = train_test_split(fitur, target, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    st.write("**Akurasi:**", accuracy_score(y_test, y_pred))
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))
    st.text("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))
    
    st.subheader("🧮 Prediksi Diabetes")

    # Form input untuk prediksi manual
    with st.form("form_prediksi"):
        col1, col2, col3 = st.columns(3)

        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 1)
            glucose = st.number_input("Glucose", 0, 200, 120)
            blood_pressure = st.number_input("Blood Pressure", 0, 150, 70)

        with col2:
            skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
            insulin = st.number_input("Insulin", 0, 900, 80)
            bmi = st.number_input("BMI", 0.0, 70.0, 30.0)

        with col3:
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
            age = st.number_input("Age", 0, 120, 30)

        submitted = st.form_submit_button("🔍 Prediksi")

    if submitted:
        # Buat DataFrame dari input
        input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                                  columns=fitur.columns)
        input_scaled = scaler.transform(input_data)
        pred = knn.predict(input_scaled)

        # Tampilkan hasil prediksi
        st.markdown("---")
        st.subheader("📢 Hasil Prediksi")
        if pred[0] == 1:
            st.error("❌ Pasien diprediksi **POSITIF Diabetes**")
        else:
            st.success("✅ Pasien diprediksi **NEGATIF Diabetes**")


elif menu == "🧩 Clustering":
    st.subheader("Clustering Pasien Diabetes (KMeans)")

    # Pilih fitur
    fitur_cluster = ['Glucose', 'BloodPressure', 'BMI', 'Age']
    X = df[fitur_cluster].copy()
    X.replace(0, np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    st.write(df[['Glucose', 'BMI', 'Cluster']].head())

    # Plot
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Glucose", y="BMI", hue="Cluster", palette="Set2", ax=ax)
    st.pyplot(fig)
