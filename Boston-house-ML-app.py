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
        
    crim = st.sidebar.slider('CRIM', 4.3, 7.9, 5.4)
    zn = st.sidebar.slider('ZN', 2.0, 4.4, 3.4)
    indus = st.sidebar.slider('INDUS', 1.0, 6.9, 1.3)
    chas = st.sidebar.slider('CHAS', 0.1, 2.5, 0.2)
    
    nox = st.sidebar.slider('NOX', 4.3, 7.9, 5.4)
    rm = st.sidebar.slider('RM', 2.0, 4.4, 3.4)
    age = st.sidebar.slider('AGE', 1, 100, 50)
    dis = st.sidebar.slider('DIS', 0.1, 2.5, 0.2)
    
    rad = st.sidebar.slider('RAD', 4.3, 7.9, 5.4)
    tax = st.sidebar.slider('TAX', 2.0, 4.4, 3.4)
    ptratio = st.sidebar.slider('PTRATIO', 1.0, 6.9, 1.3)
    b = st.sidebar.slider('B', 0.1, 2.5, 0.2)
    
    lstat = st.sidebar.slider('LSTAT', 1.0, 6.9, 1.3)
    
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




