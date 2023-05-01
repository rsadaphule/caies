import streamlit as st
import random
from typing import List, Dict, Tuple, Callable
import math
from copy import deepcopy
import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as psubs
from sklearn.neighbors import KNeighborsClassifier as KNC 
from sklearn.model_selection import train_test_split as TTS
import joblib



df = pd.read_csv('survey.csv')
# Get a sample on how user input should look like
row = df.loc[0]
# Convert the selected row to a dictionary
data_dict = row.to_dict()
ct = df.Country.value_counts()
countries, conts = ct.index, ct.values

def convert_user_input_to_model_input(input_dict):
    df_input = pd.DataFrame([input_dict])
    df_output =  pd.DataFrame([data_dict])
    #convert Gender
    mapping = {
    'male' : 0 ,
    'm' : 0,
    'male' :0,
    'make' : 0,
    'cis male' : 0,
    'f' : 1,
    'female' : 1,
    'woman' : 1,
    'identity_problems' : 2
    }
    df_output['Gender'] = df_input['Gender'].map(mapping)
    
    
    #convert Age
    df_output['Age'] = df_input['Age'].clip(lower=18, upper=80)
    df_output['Age'] = float(pd.cut(df_output['Age'], 4,  labels=[0, 1, 2, 3]))
    
    df_output['Year'] = 1973
    #convert comment
    df_output['iscomment'] = df_input['comments'].isnull().astype(int)
    
    #Convert Country
    country_map = {}
    for i, c in enumerate(conts):
        if c == conts[i] and i <= 8:
            country_map[ countries[i] ] = i
        else:
             country_map[ countries[i] ] = 9

    df_output['Country'] = df_input['Country'].map(country_map)
    
    
    
    #convert work map
    work_map = {
    'Sometimes' : 0,
    'Never' : 1,
    'Often' : 2
    }
    df_output['work_interfere'] = df_input['work_interfere'].map(work_map)
    df_output['work_interfere'] = df_output['work_interfere'].fillna(0)
    
    
    df_output = df_output.loc[:, ~ df_output.columns.isin(['treatment'])]
    df_output = df_output.loc[:, ~ df_output.columns.isin(['Age_int'])]
    df_output = df_output.loc[:, ~ df_output.columns.isin(['comments'])]
    df_output = df_output.loc[:, ~ df_output.columns.isin(['state'])]
    df_output = df_output.loc[:, ~ df_output.columns.isin(['Timestamp'])]
    df_output['self_employed'] = 1.0
    df_output['family_history'] = 0.0
    df_output['no_employees'] = df_input['no_employees']
    df_output['remote_work'] = df_input['remote_work']
    df_output['tech_company'] = df_input['tech_company']

    df_output['benefits'] = df_input['benefits']
    df_output['care_options'] = df_input['care_options']
    df_output['wellness_program'] = df_input['wellness_program']
    df_output['seek_help'] = df_input['seek_help']
    df_output['anonymity'] = df_input['anonymity']
    df_output['leave'] = df_input['leave']

    df_output['phys_health_consequence'] = df_input['phys_health_consequence']
    df_output['mental_health_consequence'] = df_input['mental_health_consequence']
    df_output['coworkers'] = df_input['coworkers']
    df_output['supervisor'] = df_input['supervisor']
    df_output['mental_health_interview'] = df_input['mental_health_interview']
    df_output['phys_health_interview'] = df_input['phys_health_interview']
    df_output['mental_vs_physical'] = df_input['mental_vs_physical']
    df_output['obs_consequence'] = df_input['obs_consequence']
    
    return df_output










st.header("AI Mental Health Tracker")
st.write("*********************************************************************************************************************")

st.write("Please answer the survey Questions Below. Based on the survey question, AI Mental Health Tracker will indicate if you need a treatment or not.")

form = st.form(key='my-form')
age = form.slider("Select Age ", 18, 50, 21)
no_employees = form.slider("Number of employees ", 0, 100, 4)
gender = form.selectbox('Select Gender ', ('male', 'female'))
country = form.selectbox('Select Country ', ('United States', 'United Kingdom', 'Canada','Germany','Ireland','Brazil','Australia','France'))
remote_work = form.radio("Do you work remotely (outside of an office) at least 50% of the time?",('Yes', 'No'))
is_tech_company = form.radio("Is your employer primarily a tech company/organization?",('Yes', 'No'))
is_benefits = form.radio("Do you have benefits at work?",('Yes', 'No'))
care_option = form.radio("Do you have care options at work?",('Yes', 'No'))
wellness_program = form.radio("Do you have wellness program at work?",('Yes', 'No'))
family_history = form.radio("Do you have a family history of mental illness?",('Yes', 'No'))
work_interfere = form.selectbox("If you have a mental health condition, do you feel that it interferes with your work?",('Sometimes', 'Never','Often'))
leave = form.radio("How easy is it for you to take medical leave for a mental health condition?",('Easy', 'Difficult','Very Difficult'))
mental_health_consequence = form.radio("Do you think that discussing a mental health issue with your employer would have negative consequences?",('Yes', 'No'))
phys_health_consequence = form.radio("Do you think that discussing a physical health issue with your employer would have negative consequences?",('Yes', 'No'))
coworkers = form.radio("Would you be willing to discuss a mental health issue with your coworkers?",('Yes', 'No'))
supervisor = form.radio("Would you be willing to discuss a mental health issue with your supervisor?",('Yes', 'No'))
comments = "blah blah"
submit = form.form_submit_button('Submit')


st.write("*********************************************************************************************************************")


pred = False
#st.write("Based on your answers to survey question. Here is AI prediction from health tracker")
#st.write(f"age: {age}")
#st.write(f"no_employees: {no_employees}")
#st.write(f"gender: {gender}")
#st.write(f"country: {country}")
#st.write(f"remote_work: {remote_work}")
#st.write(f"is_tech_company: {is_tech_company}")
#st.write(f"is_benefits: {is_benefits}")
#st.write(f"care_option: {care_option}")
#st.write(f"wellness_program: {wellness_program}")
#st.write(f"family_history: {family_history}")
#st.write(f"work_interfere: {work_interfere}")
#st.write(f"leave: {leave}")
#st.write(f"mental_health_consequence: {mental_health_consequence}")
#st.write(f"phys_health_consequence: {phys_health_consequence}")
#st.write(f"coworkers: {coworkers}")
#st.write(f"supervisor: {supervisor}")

if submit:
    #st.write("form submitted")
    # Load the saved model from disk
    loaded_forest = joblib.load('random_forest_model.pkl')
   


    # TEst code
    row_dict = {'Age':float(age), 
                'Gender': gender, 
                'Country': country, 
                'self_employed': 0.0, 
                'family_history': 1.0 if( family_history=='Yes') else 0.0, 
                'treatment': 1.0, 
                'work_interfere': 'Often', 
                'no_employees': float(no_employees),    
                'remote_work': 1.0 if( remote_work=='Yes') else 0.0, 
                'tech_company': 1.0 if( is_tech_company=='Yes') else 0.0,
                'benefits': 1.0 if( is_benefits=='Yes') else 0.0, 
                'care_options': 1.0 if( care_option=='Yes') else 0.0, 
                'wellness_program': 1.0 if( wellness_program=='Yes') else 0.0,       
                'seek_help': 1.0 ,
                'anonymity': 0.0, 
                'leave': 1.0 if( leave=='Easy') else 0.0, 
                'mental_health_consequence': 1.0 if( mental_health_consequence=='Yes') else 0.0, 
                'phys_health_consequence': 1.0 if( phys_health_consequence=='Yes') else 0.0, 
                'coworkers': 1.0 if( coworkers=='Yes') else 0.0, 
                'supervisor': 1.0 if( supervisor=='Yes') else 0.0, 
                'mental_health_interview': 1.0, 
                'phys_health_interview': 0.0, 
                'mental_vs_physical': 1.0, 
                'obs_consequence': 0.0, 
                'Year': 0.0, 
                'comments': "Blah Blah"}
    
    

    # Create a dictionary
    X_input = convert_user_input_to_model_input(row_dict)
    #perform model prediction on user input
    #st.write("X Input")
    #st.write(X_input)
    loaded_forest.predict(X_input)
    st.write(f"model output is {loaded_forest.predict(X_input)[0]}")
    pred = loaded_forest.predict(X_input)[0]
    if (pred==1):
        st.markdown(f'<h1 style="color:red;font-size:24px;">{"Attention: Please seek immediate mental health assistance. Call 1-800-Mental-Health now"}</h1>', unsafe_allow_html=True)

    else:
        st.markdown(f'<h1 style="color:#33ff33;font-size:24px;">{"Your mental health is in perfect condition. Do come back and repeat the asseement every quarter"}</h1>', unsafe_allow_html=True)



