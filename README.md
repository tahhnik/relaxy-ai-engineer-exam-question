# Loan-Approval-Model

## Overview

This project leverages machine learning to automate and enhance the loan approval process by predicting whether loan applications will be approved or rejected. By analyzing applicant profiles and financial histories, the model achieves high prediction accuracy, providing a powerful tool for financial institutions to streamline decision-making. A user-friendly web interface, developed using Flask and styled with Tailwind CSS, allows for real-time predictions, making this project both accessible and practical for end-users.

## Features

- Data-Driven Prediction: Machine learning models including XGBoost, LightGBM, and Random Forest analyze applicant data to predict loan approval outcomes.
- High Accuracy: Advanced feature engineering and model optimization ensure high prediction accuracy and reliability.
- User-Friendly Deployment: A web interface built with Flask and styled with Tailwind CSS allows for seamless interaction, enabling users to input data and receive instant predictions.
- Explainable AI: With the help of LIME, individual predictions can be explained to understand feature impact on the decision.

## Project Structure

1. Data Preprocessing & Feature Engineering: Prepares and cleanses data to improve model training efficiency and accuracy.
2. Model Training: Implements and tunes multiple machine learning algorithms to select the best-performing model.
3. Stacked Model Implementation: Combines the strengths of multiple algorithms using a stacked ensemble model.
4. Deployment: Integrates the model with Flask and Tailwind CSS for an interactive and responsive web interface.

## Installation

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/loan-approval-prediction.git
    cd loan-approval-prediction
    ```

2. **Set up the virtual environment**:
    ```bash
    python -m venv myenv
    source myenv/bin/activate    # For MacOS/Linux
    myenv\Scripts\activate       # For Windows
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
## Usage

1. **Run the Web Application**:
    Start the Flask application to access the loan approval prediction tool:
    ```bash
    python app.py
    ```
    Access the app at `http://127.0.0.1:5000` in your browser.

2. **Use the Prediction Interface**:
    Enter applicant details in the web interface to get real-time predictions on loan approval.


## Model Explainability with LIME
This project incorporates LIME (Local Interpretable Model-Agnostic Explanations) to enhance model interpretability:
- **Global Interpretability**: Highlights the general feature importance across all predictions.
- **Instance-Level Interpretability**: Provides explanations for individual loan applications, offering insights into the factors affecting each prediction.

## Model Deployment

The Loan Approval Prediction model is deployed using Render, enabling users to interact with the model in real-time. This deployment provides a seamless way for end-users to enter loan application details and receive instant predictions on approval status.

### Access the Model:
- **[Loan Approval Prediction App on Render](https://loan-approval-model.onrender.com/)**

Visit the link to test the model's predictive capabilities and explore the user-friendly interface built for efficient loan assessment.


## Tech Stack

- **Backend**: Flask
- **Frontend**: Tailwind CSS
- **Machine Learning**: Scikit-Learn, XGBoost, LightGBM, Decision Tree, Random Forest
- **Model Serialization**: Joblib
- **Explainability**: LIME

## Results
The model was tested on historical loan application data, showing a high degree of accuracy and reliability. Detailed metrics such as Accuracy, MCC, and AUC-ROC scores can be found in the **Results** section of this repository.

## Future Work
- **Additional Model Optimization**: Experiment with more complex architectures and hyperparameter tuning.
- **Enhanced Frontend**: Build a more detailed UI for visualization of results and explanations.
- **Real-Time Data Integration**: Connect the model with real-time data sources for up-to-date predictions.
- **Expand Explainability**: Integrate more explainability methods for improved transparency.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue to suggest improvements.

## License
This project is licensed under the MIT License.
  
