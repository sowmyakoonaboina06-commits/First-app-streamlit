import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
st.title("ML Model Demo")
st.header("Prediction System")
st.subheader("Enter Inputs below")
st.write("This app demostrates Ml model deployment")
name=st.text_input("Enter Customer name")
age=st.number_input("Enter Age:",min_value=0,max_value=100)
salary=st.slider("Select Salary",10000,100000)
gender=st.selectbox("Select Gender",["Male","Female"])
education=st.radio("Education Level",["UG","PG","PHD"])
agree=st.checkbox("I agree to terms")
uploaded_file=st.file_uploader("Upload Image",type=["jpg","png"])
if st.button("Predict"):
    st.success("Prediction Successful")
    st.warning("Warning")
    st.error("Error,enter valid input")
df=pd.DataFrame({"A":[1,2],"B":[3,4]})
st.dataframe(df)

appointment=st.date_input("Select the appointment data")
st.write("Appointment Date:",appointment)
time=st.time_input("Select Appointment time")
#To display Accuracy ,Precision,Recall
st.metric("Accuracy","92%","+2")
#Image Display CNN app ,Computer vision based apps
st.image("peacock.jpg",caption="peacock")
#Audio & Video
#st.audio("<audio file name>")
#st.video("<video file name>")

#Sidebar
st.sidebar.title("Navigation")
page=st.sidebar.selectbox("Choose Page",["Home","Prediction"])

#Layout Control
col1,col2=st.columns(2)
with col1:
    st.write("Left")
with col2:
    st.write("Right")
#Progress & Spinners (ML/DL MOdels)
with st.spinner("Processing..."):
    #result=model.predict(data)
    st.progress(50)

#Caching model (Very Important)
#Prevents reloading model everytime
#@st.cache_resource
#def load_model():
    #return joblib.load("model.pkl")

#model=load_model()
#DataFrame
data={
    'Name':['Abdul Aziz','Anna','Bob','Peter','Ram','Sita','Radha'],
    'Age':[17,19,18,16,15,19,17],
    'City':['India','Paris','London','Berlin','Mumbai','Chennai','America']
}
df=pd.DataFrame(data)
st.dataframe(df) #Interactive table

#display some charts
rand=np.random.normal(1,2,size=20)
fig,ax=plt.subplots()
ax.hist(rand,bins=15) #color='pink'
st.pyplot(fig)