import streamlit as st
import pandas as pd
import joblib

st.title("NYC Crime Borough Predictor")
st.write("""
This application predicts the **borough** where a crime likely occurred in New York City based on various features. 
Use the options below to provide the necessary details for prediction.
""")

model = joblib.load('knn_model3.pkl')  

feature_order = [
    'law_cat_cd_encoded',
    'juris_desc_encoded',
    'ofns_desc_encoded',
    'fr_year',
    'fr_month_name_encoded',
    'susp_race_encoded',
    'susp_sex_encoded',
    'susp_age_group_encoded',
    'vic_race_encoded', 'vic_sex_encoded', 'vic_age_group_encoded'
]

value_mappings = {
    'law_cat_cd_encoded': {'felony': 0, 'misdemeanor': 1, 'violation': 2},
    'juris_desc_encoded': {'nypd': 0, 'housing': 1, 'transit': 2, 'other': 3},
    'ofns_desc_encoded': {'assault': 0, 'theft': 1, 'burglary': 2, 'drugs': 3, 'other': 4},
    'fr_month_name_encoded': {
        'January': 0, 'February': 1, 'March': 2, 'April': 3, 'May': 4,
        'June': 5, 'July': 6, 'August': 7, 'September': 8, 'October': 9,
        'November': 10, 'December': 11},
    'susp_race_encoded': {'white': 0, 'black': 1, 'asian': 2, 'other': 3},
    'susp_sex_encoded': {'M': 0, 'F': 1},
    'susp_age_group_encoded': {'<18': 0, '18-24': 1, '25-44': 2, '45-64': 3, '65+': 4},
    'vic_race_encoded': {'white': 0, 'black': 1, 'asian': 2, 'other': 3},
    'vic_sex_encoded': {'M': 0, 'F': 1},
    'vic_age_group_encoded': {'<18': 0, '18-24': 1, '25-44': 2, '45-64': 3, '65+': 4},
}

borough_mapping = {
    0: 'Manhattan',
    1: 'Brooklyn',
    2: 'Queens',
    3: 'Bronx',
    4: 'Staten Island'
}

with st.container():
    st.header("Primary Details")
    user_inputs = {
        'law_cat_cd_encoded': st.selectbox("Select Law Category:", list(value_mappings['law_cat_cd_encoded'].keys())),
        'juris_desc_encoded': st.selectbox("Select Jurisdiction:", list(value_mappings['juris_desc_encoded'].keys())),
        'ofns_desc_encoded': st.selectbox("Select Offense Description:", list(value_mappings['ofns_desc_encoded'].keys())),
        'fr_year': st.selectbox("Select Complaint Year:", [2020, 2021, 2022, 2023, 2024]),
        'fr_month_name_encoded': st.selectbox("Select Complaint Month:", list(value_mappings['fr_month_name_encoded'].keys())),
    }

with st.expander("Additional Details"):
    st.subheader("Suspect Details")
    user_inputs.update({
        'susp_race_encoded': st.selectbox("Select Suspect Race:", list(value_mappings['susp_race_encoded'].keys())),
        'susp_sex_encoded': st.selectbox("Select Suspect Sex:", list(value_mappings['susp_sex_encoded'].keys())),
        'susp_age_group_encoded': st.selectbox("Select Suspect Age Group:", list(value_mappings['susp_age_group_encoded'].keys())),
    })

    st.subheader("Victim Details")
    user_inputs.update({
        'vic_race_encoded': st.selectbox("Select Victim Race:", list(value_mappings['vic_race_encoded'].keys())),
        'vic_sex_encoded': st.selectbox("Select Victim Sex:", list(value_mappings['vic_sex_encoded'].keys())),
        'vic_age_group_encoded': st.selectbox("Select Victim Age Group:", list(value_mappings['vic_age_group_encoded'].keys())),
    })

with st.sidebar:
    st.header("Make Prediction")
    if st.button("Predict Borough"):
        encoded_inputs = {key: value_mappings[key][value] for key, value in user_inputs.items() if key in value_mappings}
        encoded_inputs['fr_year'] = user_inputs['fr_year']  

        ordered_inputs = [encoded_inputs[feature] for feature in feature_order]

        input_data = pd.DataFrame([ordered_inputs], columns=feature_order)
        
        prediction = model.predict(input_data)
        
        predicted_borough = borough_mapping.get(prediction[0], "Unknown")

        st.success(f"The predicted borough is: **{predicted_borough}**")
