import pandas as pd
import streamlit as st
import joblib
import os
import warnings
import time
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

st.title("sentiment analysis chatbot")

name = st.text_input("enter your msg")


data=pd.read_csv(r'final.csv',encoding='latin-1')
# print(data.columns)
data['texts'] = data['texts'].fillna("")
label_encoder = LabelEncoder()

def preprocessing():
    X = data['texts']  
    Y = data['sentiment']  
    vectorizer=TfidfVectorizer()
    X_TRANS=vectorizer.fit_transform(X)
    
    y = label_encoder.fit_transform(Y)
    return X_TRANS,y,vectorizer

X_TRANS, y, vectorizer = preprocessing()
X_train,X_test,Y_train,Y_test=train_test_split(X_TRANS,y , random_state=3,test_size=0.3)
model=MultinomialNB()
loaded_model=model.fit(X_train,Y_train)
        

if st.button("Analyze sentiment"):
    if name:
        start_time = time.time()
        userinput=vectorizer.transform([name])
        prediction=loaded_model.predict(userinput)
        text_output=label_encoder.inverse_transform(prediction)
        
        end_time = time.time()  
        elapsed_time = end_time - start_time  
        
        
        st.write(f"Sentiment: {text_output}")
        st.write(f"Time taken for prediction: {elapsed_time:.4f} seconds")
    else:
        st.write("please write a msg to analyze")



