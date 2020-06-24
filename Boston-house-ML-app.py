import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

st.write("""
# Simple Boston House Price Prediction App
This app predicts the **Boston House Price** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    #'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    #   'PTRATIO', 'B', 'LSTAT', 'PRICE'
        
    crim = st.sidebar.slider('CRIM', 0.0, 89.0, 44.0)
    zn = st.sidebar.slider('ZN', 0.0, 100.0, 50.0)
    indus = st.sidebar.slider('INDUS', 0.46, 28.0, 14.0)
    chas = st.sidebar.slider('CHAS', 0, 1, 0)
    
    nox = st.sidebar.slider('NOX', 0.3, 0.88, 0.44)
    rm = st.sidebar.slider('RM', 3.56, 8.78, 4.0)
    age = st.sidebar.slider('AGE', 1, 100, 50)
    dis = st.sidebar.slider('DIS', 1.12, 12.12, 6.0)
    
    rad = st.sidebar.slider('RAD', 1.0, 24.0, 12.0)
    tax = st.sidebar.slider('TAX', 187.0, 711.0, 300.0)
    ptratio = st.sidebar.slider('PTRATIO', 12.0, 22.0, 18.0)
    b = st.sidebar.slider('B', 0.3, 399.0, 5.0)
    
    lstat = st.sidebar.slider('LSTAT', 1.73, 40.0, 5.0)
    
     #'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    #   'PTRATIO', 'B', 'LSTAT', 'PRICE'
    
    data = {'CRIM': np.sqrt(crim),
            'ZN': np.sqrt(zn),
            'INDUS': np.sqrt(indus),
            'CHAS': np.sqrt(chas),
            'NOX': np.sqrt(nox),
            'RM': np.sqrt(rm),
            'AGE': np.sqrt(age),
            'DIS': np.sqrt(dis),
            'RAD': np.sqrt(rad),
            'TAX': np.sqrt(tax),
            'PTRATIO': np.sqrt(ptratio),
            'B': np.sqrt(b),
            'LSTAT': np.sqrt(lstat)
           
           }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)


boston = load_boston()
df1=pd.DataFrame(data=boston.data, columns= boston.feature_names)
df1['PRICE'] = boston.target
X = np.sqrt(df1.copy())
Y = X.pop('PRICE')

lm = LinearRegression()
lm.fit(X, Y)

prediction = lm.predict(df)

#st.subheader('Class labels and their corresponding index number')
#st.write(iris.target_names)

st.subheader('Prediction')
#st.write(iris.target_names[prediction])
st.write(prediction)

#st.subheader('Prediction Probability')
#st.write(prediction_proba)




