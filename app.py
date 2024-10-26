from flask import Flask, jsonify, render_template, request
import joblib
import pandas as pd
# import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier

from utils import preprocess




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


app = Flask(__name__)


preprocessor = joblib.load("./models/preprocessor.pkl")

model = joblib.load("./models/best_xgb.pkl")
model2 = joblib.load("./models/best_lgbm.pkl")
model3 = joblib.load("./models/best_rf.pkl")
model4 = joblib.load("./models/best_dt.pkl")

final_model = joblib.load("./models/stacked_model.pkl")


numerical_features = [
    'income_annum', 'loan_amount', 'loan_term', 
    'cibil_score', 'residential_assets_value', 
    'commercial_assets_value', 'luxury_assets_value', 
    'bank_asset_value', 'no_of_dependents'
]
categorical_features = ['education', 'self_employed']
selected_features = numerical_features + categorical_features


@app.route('/', methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    numerical_inputs = [float(request.form[feature]) for feature in numerical_features]
    categorical_inputs = [request.form[feature] for feature in categorical_features]

    # Combine numerical and categorical inputs
    input_data = pd.DataFrame([numerical_inputs + categorical_inputs], columns=selected_features)

    processed_df = preprocess(input_data)

    predictions = final_model.predict(processed_df)
    predictions = 1 if predictions == 1 else -1
    return render_template("index.html", predictions=predictions)


@app.route('/xgboost', methods=["GET", "POST"])
def xgboost():
    return render_template("model1.html")

@app.route("/predict_xgboost", methods=["POST"])
def predict_xgboost():
    numerical_inputs = [float(request.form[feature]) for feature in numerical_features]
    categorical_inputs = [request.form[feature] for feature in categorical_features]

    input_data = pd.DataFrame([numerical_inputs + categorical_inputs], columns=selected_features)

    processed_df = preprocess(input_data)

    predictions = model.predict(processed_df)
    predictions = 1 if predictions == 1 else -1
    return render_template("model1.html", predictions=predictions)


@app.route('/lgbm', methods=["GET", "POST"])
def lgbm():
    return render_template("model2.html")

@app.route("/predict_lgbm", methods=["POST"])
def predict_lgbm():
    numerical_inputs = [float(request.form[feature]) for feature in numerical_features]
    categorical_inputs = [request.form[feature] for feature in categorical_features]

    input_data = pd.DataFrame([numerical_inputs + categorical_inputs], columns=selected_features)

    processed_df = preprocess(input_data)
    
    predictions = model2.predict(processed_df)
    predictions = 1 if predictions == 1 else -1
    return render_template("model2.html", predictions=predictions)

@app.route('/rf', methods=["GET", "POST"])
def rf():
    return render_template("model3.html")

@app.route("/predict_rf", methods=["POST"])
def predict_rf():
    numerical_inputs = [float(request.form[feature]) for feature in numerical_features]
    categorical_inputs = [request.form[feature] for feature in categorical_features]

    input_data = pd.DataFrame([numerical_inputs + categorical_inputs], columns=selected_features)

    processed_df = preprocess(input_data)

    predictions = model3.predict(processed_df)
    predictions = 1 if predictions == 1 else -1
    return render_template("model3.html", predictions=predictions)


@app.route('/dt', methods=["GET", "POST"])
def dt():
    return render_template("model4.html")

@app.route("/predict_dt", methods=["POST"])
def predict_dt():
    numerical_inputs = [float(request.form[feature]) for feature in numerical_features]
    categorical_inputs = [request.form[feature] for feature in categorical_features]

    # Combine numerical and categorical inputs
    input_data = pd.DataFrame([numerical_inputs + categorical_inputs], columns=selected_features)

    processed_df = preprocess(input_data)

    predictions = model4.predict(processed_df)
    predictions = 1 if predictions == 1 else -1
    return render_template("model4.html", predictions=predictions)


@app.route('/about')
def about():
    return render_template('about.html') 


if __name__ == "__main__":
    app.run(debug=True)