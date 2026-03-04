import streamlit as st
import numpy as np
import pickle
from sklearn.datasets import load_iris

iris = load_iris()

with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Iris prediction app")

sepal_length = st.number_input("Sepal length", 4.0, 8.0, 5.1)
sepal_width = st.number_input("Sepal width", 2.0, 4.5, 3.5)
petal_length = st.number_input("Petal length", 1.0, 7.0, 1.4)
petal_width = st.number_input("Petal width", 0.1, 2.5, 0.2)

if st.button("Predict"):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = model.predict(data)

    st.success("Loai hoa: " + iris.target_names[pred[0]])