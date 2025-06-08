import streamlit as st
import numpy as np
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

#Load the trained model 
model = tf.keras.models.load_model('model.h5')

#load the encoder and scaler
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


#streamlit app
st.title('customer churn prediction')

#user input 
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age',18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Estimated Salary')
estimated_salary = st.number_input('estimated salary')
tenure = st.slider('tenure', 0, 10)
num_of_products = st.slider('number of products', 1, 4)
has_Cr_Card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('is active member', [0,1])

#prepare the input data 
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'tenure': [tenure],
    'balance': [balance],
    'num_of_products': [num_of_products],
    'has_Cr_Card' : [has_Cr_Card],
    'is_active_member' : [is_active_member],
    'estimated_salary' : [estimated_salary]
})

#one-hot encode 'geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns = onehot_encoder_geo.get_feature_names_out(['Geography']))

#concat onehot encoded
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# scaling the input data
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

if prediction_proba > 0.5:
    print('The coustomer is likely to churn')
else:
    print('The coustomer is not likely to churn')