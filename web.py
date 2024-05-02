"""
Sardor Nodirov
05/02/2024
PH398 Final Project

Developed by Sardor Nodirov for PH398 Final Project. Data source: Kaggle. Data Science Job Salaries by Ruchi Bhatia (2022). https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries/data
"""

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pickle

def country_code(code):
    codes = ['DE', 'JP', 'GB', 'HN', 'US', 'HU', 'NZ', 'FR', 'IN', 'PK', 'PL', 'PT', 'CN', 'GR', 'AE', 'NL', 'MX', 'CA', 'AT', 'NG', 'PH', 'ES', 'DK', 'RU', 'IT', 'HR', 'BG', 'SG', 'BR', 'IQ', 'VN', 'BE', 'UA', 'MT', 'CL', 'RO', 'IR', 'CO', 'MD', 'KE', 'SI', 'HK', 'TR', 'RS', 'PR', 'LU', 'JE', 'CZ', 'AR', 'DZ', 'TN', 'MY', 'EE', 'AU', 'BO', 'IE', 'CH', 'IL', 'AS']
    names = ['Germany', 'Japan', 'United Kingdom', 'Honduras', 'United States', 'Hungary', 'New Zealand', 'France', 'India', 'Pakistan', 'Poland', 'Portugal', 'China', 'Greece', 'United Arab Emirates', 'Netherlands', 'Mexico', 'Canada', 'Austria', 'Nigeria', 'Philippines', 'Spain', 'Denmark', 'Russia', 'Italy', 'Croatia', 'Bulgaria', 'Singapore', 'Brazil', 'Iraq', 'Vietnam', 'Belgium', 'Ukraine', 'Malta', 'Chile', 'Romania', 'Iran', 'Colombia', 'Moldova', 'Kenya', 'Slovenia', 'Hong Kong', 'Turkey', 'Serbia', 'Puerto Rico', 'Luxembourg', 'Jersey', 'Czechia', 'Argentina', 'Algeria', 'Tunisia', 'Malaysia', 'Estonia', 'Australia', 'Bolivia', 'Ireland', 'Switzerland', 'Israel', 'American Samoa']

    return names[codes.index(code)]

features = ['work_year', 'experience_level', 'remote_ratio', 'company_size', 'employment_type_CT', 'employment_type_FL', 'employment_type_FT', 'employment_type_PT', 'job_title_3D Computer Vision Researcher', 'job_title_AI Scientist', 'job_title_Analytics Engineer', 'job_title_Applied Data Scientist', 'job_title_Applied Machine Learning Scientist', 'job_title_BI Data Analyst', 'job_title_Big Data Architect', 'job_title_Big Data Engineer', 'job_title_Business Data Analyst', 'job_title_Cloud Data Engineer', 'job_title_Computer Vision Engineer', 'job_title_Computer Vision Software Engineer', 'job_title_Data Analyst', 'job_title_Data Analytics Engineer', 'job_title_Data Analytics Lead', 'job_title_Data Analytics Manager', 'job_title_Data Architect', 'job_title_Data Engineer', 'job_title_Data Engineering Manager', 'job_title_Data Science Consultant', 'job_title_Data Science Engineer', 'job_title_Data Science Manager', 'job_title_Data Scientist', 'job_title_Data Specialist', 'job_title_Director of Data Engineering', 'job_title_Director of Data Science', 'job_title_ETL Developer', 'job_title_Finance Data Analyst', 'job_title_Financial Data Analyst', 'job_title_Head of Data', 'job_title_Head of Data Science', 'job_title_Head of Machine Learning', 'job_title_Lead Data Analyst', 'job_title_Lead Data Engineer', 'job_title_Lead Data Scientist', 'job_title_Lead Machine Learning Engineer', 'job_title_ML Engineer', 'job_title_Machine Learning Developer', 'job_title_Machine Learning Engineer', 'job_title_Machine Learning Infrastructure Engineer', 'job_title_Machine Learning Manager', 'job_title_Machine Learning Scientist', 'job_title_Marketing Data Analyst', 'job_title_NLP Engineer', 'job_title_Principal Data Analyst', 'job_title_Principal Data Engineer', 'job_title_Principal Data Scientist', 'job_title_Product Data Analyst', 'job_title_Research Scientist', 'job_title_Staff Data Scientist', 'company_location_AE', 'company_location_AS', 'company_location_AT', 'company_location_AU', 'company_location_BE', 'company_location_BR', 'company_location_CA', 'company_location_CH', 'company_location_CL', 'company_location_CN', 'company_location_CO', 'company_location_CZ', 'company_location_DE', 'company_location_DK', 'company_location_DZ', 'company_location_EE', 'company_location_ES', 'company_location_FR', 'company_location_GB', 'company_location_GR', 'company_location_HN', 'company_location_HR', 'company_location_HU', 'company_location_IE', 'company_location_IL', 'company_location_IN', 'company_location_IQ', 'company_location_IR', 'company_location_IT', 'company_location_JP', 'company_location_KE', 'company_location_LU', 'company_location_MD', 'company_location_MT', 'company_location_MX', 'company_location_MY', 'company_location_NG', 'company_location_NL', 'company_location_NZ', 'company_location_PK', 'company_location_PL', 'company_location_PT', 'company_location_RO', 'company_location_RU', 'company_location_SG', 'company_location_SI', 'company_location_TR', 'company_location_UA', 'company_location_US', 'company_location_VN']


def group_col(col, target):
    output = []
    for i in features:
        if i.startswith(col):
            if i == target:
                output.append(True)
            else:
                output.append(False)

    return output


def main():
    st.title('AI Salary Predictor for Data Science Jobs')

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Input fields
    work_year = st.selectbox("Year", [2020, 2021, 2022])
    job_title = st.selectbox('Job Title', ['3D Computer Vision Researcher', 'AI Scientist', 'Analytics Engineer', 'Applied Data Scientist', 'Applied Machine Learning Scientist', 'BI Data Analyst', 'Big Data Architect', 'Big Data Engineer', 'Business Data Analyst', 'Cloud Data Engineer', 'Computer Vision Engineer', 'Computer Vision Software Engineer', 'Data Analyst', 'Data Analytics Engineer', 'Data Analytics Lead', 'Data Analytics Manager', 'Data Architect', 'Data Engineer', 'Data Engineering Manager', 'Data Science Consultant', 'Data Science Engineer', 'Data Science Manager', 'Data Scientist', 'Data Specialist', 'Director of Data Engineering', 'Director of Data Science', 'ETL Developer', 'Finance Data Analyst', 'Financial Data Analyst', 'Head of Data', 'Head of Data Science', 'Head of Machine Learning', 'Lead Data Analyst', 'Lead Data Engineer', 'Lead Data Scientist', 'Lead Machine Learning Engineer', 'ML Engineer', 'Machine Learning Developer', 'Machine Learning Engineer', 'Machine Learning Infrastructure Engineer', 'Machine Learning Manager', 'Machine Learning Scientist', 'Marketing Data Analyst', 'NLP Engineer', 'Principal Data Analyst', 'Principal Data Engineer', 'Principal Data Scientist', 'Product Data Analyst', 'Research Scientist', 'Staff Data Scientist'])
    experience_level = st.selectbox('Experience Level', ['Entry Level', 'Mid Level', 'Senior Level', 'Executive Level'])
    company_size = st.selectbox('Company Size', ['Small', 'Medium', 'Large'])
    employment_type = st.selectbox('Employment Type', ['Full-time', 'Part-time', 'Contract', 'Freelance'])
    remote_ratio = st.slider('Remote Ratio (%)', 0, 100, 0)
    company_location = st.selectbox('Company Main Office Location', sorted(['DE', 'JP', 'GB', 'HN', 'US', 'HU', 'NZ', 'FR', 'IN', 'PK', 'CN', 'GR', 'AE', 'NL', 'MX', 'CA', 'AT', 'NG', 'ES', 'PT', 'DK', 'IT', 'HR', 'LU', 'PL', 'SG', 'RO', 'IQ', 'BR', 'BE', 'UA', 'IL', 'RU', 'MT', 'CL', 'IR', 'CO', 'MD', 'KE', 'SI', 'CH', 'VN', 'AS', 'TR', 'CZ', 'DZ', 'EE', 'MY', 'AU', 'IE']), format_func=country_code)
    
    # Mapping
    experience_level_map = {'Entry Level': 0, 'Mid Level': 1, 'Senior Level': 2, 'Executive Level': 3}
    company_size_map = {'Small': 0, 'Medium': 1, 'Large': 2}
    employment_type_map = {'Full-time': 'FT', 'Part-time': 'PT', 'Contract': 'CT', 'Freelance': 'FL'}

    user_input = [[]]
    user_input[0].append(work_year) # work_year
    user_exp = experience_level_map[experience_level]
    user_input[0].append(user_exp) #experience_level
    user_size = company_size_map[company_size]
    user_input[0].append(remote_ratio) # remote_ratio
    user_input[0].append(user_size) # company_size
    user_employment = employment_type_map[employment_type]
    user_type = "employment_type_" + user_employment
    user_input[0] = user_input[0] + group_col('employment_type', user_type) # employment_type
    user_job_title = "job_title_" + job_title
    user_input[0] = user_input[0] + group_col('job_title', user_job_title) # job title
    user_company_location = "company_location_" + company_location
    user_input[0] = user_input[0] + group_col('company_location', user_company_location) # company location
    
    if st.button('Predict Salary'):
        # Prediction and confidence interval
        prediction = model.predict(user_input)
        
        st.subheader('Predicted Salary (USD)')
        st.title(f'${prediction[0]:.2f}')

if __name__ == '__main__':
    main()