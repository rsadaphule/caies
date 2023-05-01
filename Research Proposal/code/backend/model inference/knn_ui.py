import streamlit as st
import random
from typing import List, Dict, Tuple, Callable
import math
from copy import deepcopy
import os



FILE_PATH = "concrete_compressive_strength.csv"    # chamge the path to your own directory. Relative path does not seem to work
NUM_FEATURES : int = 8
LABEL_INDEX: int = 8
distance_dict_g:dict = {} 
value_dictionary:dict = {}

def parse_data(file_name: str) -> List[List]:
    data = []
    file = open(file_name, "r")
    for line in file:
        datum = [float(value) for value in line.rstrip().split(",")]
        data.append(datum)
    random.shuffle(data)
    return data

#cwd = os.getcwd() 
data = parse_data(FILE_PATH)
#print(data[0])


def compute_Eucladian_distance(point1: list, point2: list, num_features: int) -> float :
    distance: float = 0.0
    for i in range(num_features):
        diff = point1[i] - point2[i] 
        distance = distance +  diff * diff
    distance = math.sqrt(distance) 
    return distance


#def sort_distance_dict(distance_dict: dict[int, float]) -> dict[int, float] :
    #sorted_distance_dict = distance_dict
    #print(sorted_distance_dict)
    #sorted_distance_list = sorted(distance_dict.items(), key=lambda x: x[1])
    #for t in sorted_distance_list:
    #    sorted_distance_dict[t[0]] = t[1]
    #top_key = sorted_distance_list[0]   
  #  return (distance_dict, None)


def KNN(K: int, query: list, train_data:  List[List], num_features: int, label_index: int) -> float :
    distance_dict_g = {}
    y_hat : float = 0.0
    n  = len(train_data)
    for i in range(n):
        ed = compute_Eucladian_distance(query, train_data[i], num_features)
        distance_dict_g[i] = ed

    # sort the neighbors by distance
    #(distance_dict_g, _) =  sort_distance_dict(distance_dict_g)
    counter = 0
    for key in distance_dict_g:
         value_dictionary[key] = train_data[key]
    


    #select k neighbors and compute average of y_actual as final avlaue
    sum: float = 0.0
    neighbour_counter = 0 
    for key in distance_dict_g:
        observation = train_data[key]
        y_actual = observation[label_index]
        sum = sum + y_actual
        neighbour_counter = neighbour_counter + 1
        if neighbour_counter == K :
            break

    y_hat = sum / float(K) 
    return (y_hat, distance_dict_g, value_dictionary)





st.header("AI Mental Health Tracker")

#st.sidebar.header("Survey Questions")
#cement = st.sidebar.slider("Cement", 0.0, 540.0, 220.0)
#slag = st.sidebar.slider("Slag", 0.0, 400.0, 10.0)
#ash = st.sidebar.slider("Ash", 0.0, 250.0, 10.0)
#water = st.sidebar.slider("Water", 0.0, 300.0, 10.0)
#superplasticizer = st.sidebar.slider("Super Plasticizer", 0.0, 50.0, 10.0)
#coarseaggregate = st.sidebar.slider("Coarse aggregate", 0.0, 1200.0, 10.0)
#fineaggregate = st.sidebar.slider("Fine Aggregate 7", 0.0, 1000.0, 10.0)
#age = st.sidebar.slider("Age", 0.0, 365.0, 10.0)

st.write("Please answer the survey Questions Below. Based on the survey question, AI will indicate if you need a treatment or not.")
age = st.slider("Select Age ", 18, 50, 21)
no_employees = st.slider("Number of employees ", 0, 100, 4)
gender = st.selectbox('Select Gender ', ('Male', 'Female'))
country = st.selectbox('Select Country ', ('United States', 'United Kingdom', 'Canada','Germany','Ireland','Brazil','Australia','France'))
remote_work = st.radio("Do you work remotely (outside of an office) at least 50% of the time?",('Yes', 'No'))
is_tech_company = st.radio("Is your employer primarily a tech company/organization?",('Yes', 'No'))
is_benefits = st.radio("Do you have benefits at work?",('Yes', 'No'))
care_option = st.radio("Do you have care options at work?",('Yes', 'No'))
wellness_program = st.radio("Do you have wellness program at work?",('Yes', 'No'))
family_history = st.radio("Do you have a family history of mental illness?",('Yes', 'No'))
work_interfere = st.selectbox("If you have a mental health condition, do you feel that it interferes with your work?",('Sometimes', 'Never','Often'))
leave = st.radio("How easy is it for you to take medical leave for a mental health condition?",('Easy', 'Difficult','Very Difficult'))
mental_health_consequence = st.radio("Do you think that discussing a mental health issue with your employer would have negative consequences?",('Yes', 'No'))
phys_health_consequence = st.radio("Do you think that discussing a physical health issue with your employer would have negative consequences?",('Yes', 'No'))
coworkers = st.radio("Would you be willing to discuss a mental health issue with your coworkers?",('Yes', 'No'))
supervisor = st.radio("Would you be willing to discuss a mental health issue with your supervisor?",('Yes', 'No'))
comments = "blah blah"


st.write("*********************************************************************************************************************")


pred = False
st.write("Based on your answers to survey question. Here is AI prediction from health tracker")

if (pred):
    st.markdown(f'<h1 style="color:red;font-size:24px;">{"Attention: Please seek immediate mental health assitsance. Call 1-800-Mental-Health now"}</h1>', unsafe_allow_html=True)

else:
    st.markdown(f'<h1 style="color:#33ff33;font-size:24px;">{"Your mental health is in perfect condition. Do come back and repeat the asseement every quarter”"}</h1>', unsafe_allow_html=True)





