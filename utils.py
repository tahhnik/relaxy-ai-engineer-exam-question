from flask import Flask, jsonify, render_template, request
import joblib
import pandas as pd
# import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier

preprocessor = joblib.load("./models/preprocessor.pkl")

def feature_engineering(X):
    X['loan_to_income_ratio'] = X['loan_amount'] / X['income_annum']
    X['total_asset_value'] = (X['residential_assets_value'] +
                              X['commercial_assets_value'] +
                              X['luxury_assets_value'] +
                              X['bank_asset_value'])
    
    X['income_per_dependent'] = X['income_annum'] / (X['no_of_dependents'] + 1) 

    def cibil_category(score):
        if score < 500:
            return 'Low'
        elif 500 <= score <= 700:
            return 'Medium'
        else:
            return 'High'
    X['cibil_category'] = X['cibil_score'].apply(cibil_category)

    return X

final_model = joblib.load("./models/stacked_model.pkl")

numerical_features = [
    'income_annum', 'loan_amount', 'loan_term', 
    'cibil_score', 'residential_assets_value', 
    'commercial_assets_value', 'luxury_assets_value', 
    'bank_asset_value', 'no_of_dependents'
]
categorical_features = ['education', 'self_employed']
selected_features = numerical_features + categorical_features

def preprocess(input_data):
    feature_engineering(input_data)

    print(input_data.columns)

    numerical_cols = [
    'income_annum', 'loan_amount', 'loan_term', 
    'cibil_score', 'residential_assets_value', 
    'commercial_assets_value', 'luxury_assets_value', 
    'bank_asset_value', 'income_per_dependent'] 

    categorical_cols = ['education', 'self_employed', 'cibil_category']

    selected_cols = numerical_cols + categorical_cols 
    columns_names = numerical_cols + categorical_cols + ['education_2','self_employed_2','cibil_category_2','cibil_category_3']

    # Apply preprocessing to input data
    processed_data = preprocessor.transform(input_data[selected_cols])

    processed_df = pd.DataFrame(processed_data, columns=columns_names) 
    processed_df['total_asset_value'] = input_data['total_asset_value']
    processed_df['loan_to_income_ratio'] = input_data['loan_to_income_ratio']
    processed_df['no_of_dependents'] = input_data['no_of_dependents']

    print(processed_df)

    return processed_df
