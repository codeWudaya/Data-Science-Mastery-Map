import streamlit as st
import joblib
import numpy as np

def main():
    html_temp = """
    <div style="background-color:lightblue;padding:16px">
    <h2 style="color:black";text-align:center> Heart Disease Prediction Using ML</h2>
    </div>
    
    """
    
    st.markdown(html_temp,unsafe_allow_html=True)
    
    model = joblib.load('model_joblib_heart')
    
    p1 = st.number_input('Enter Your Age', min_value=1, max_value=100, format='%d')
    
    s1 = st.selectbox('Sex',('Male','Female'))
    
    if s1=='Male':
        p2=1
    else:
        p2=0
        
    p3 = st.slider("Enter value of CP",0,3)
    p4=st.number_input("Enter Value of trestbps")
    p5=st.number_input("Enter Value of chol")
    p6=st.slider("Enter Value of fbs",0,1)
    p7=st.slider("Enter Value of restecg",0,1,2)
    p8=st.number_input("Enter Value of thalach")
    p9=st.slider("Enter Value of exang",0,1)
    p10 = st.number_input("Enter Value of oldpeak", min_value=0.0, max_value=7.0, step=0.1)
    p11=st.slider("Enter Value of slope",0,2)
    p12=st.slider("Enter Value of ca",0,4)
    p13=st.slider("Enter Value of thal",0,3)
    
    if st.button('Predict'):
        pred= model.predict([[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13]])
        
        st.balloons()
        if pred == 0:
            st.write("<h3 style='color:green;'>No Heart Disease</h3>", unsafe_allow_html=True)
        else:
            st.write("<h3 style='color:red;'>Heart Disease Found</h3>", unsafe_allow_html=True)
    
        st.success('Prediction Complete')
        st.success('')
    


if __name__ == '__main__':
    main()