# Loan-Approval-Model

## Overview

This project leverages machine learning to automate and enhance the loan approval process by predicting whether loan applications will be approved or rejected. By analyzing applicant profiles and financial histories, the model achieves high prediction accuracy, providing a powerful tool for financial institutions to streamline decision-making. A user-friendly web interface, developed using Flask and styled with Tailwind CSS, allows for real-time predictions, making this project both accessible and practical for end-users.

## Features

- Data-Driven Prediction: Machine learning models including XGBoost, LightGBM, and Random Forest analyze applicant data to predict loan approval outcomes.
- High Accuracy: Advanced feature engineering and model optimization ensure high prediction accuracy and reliability.
- User-Friendly Deployment: A web interface built with Flask and styled with Tailwind CSS allows for seamless interaction, enabling users to input data and receive instant predictions.
- Explainable AI: With the help of LIME, individual predictions can be explained to understand feature impact on the decision.

## Dataset

The dataset used for this project contains various features related to loan applications, including applicant demographics, loan amount, and loan purpose. The data can be found on Kaggle:

- **Dataset Link:** [Loan Approval Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)

## Project Structure

1. Data Preprocessing & Feature Engineering: Prepares and cleanses data to improve model training efficiency and accuracy.
2. Model Training: Implements and tunes multiple machine learning algorithms to select the best-performing model.
3. Stacked Model Implementation: Combines the strengths of multiple algorithms using a stacked ensemble model.
4. Deployment: Integrates the model with Flask and Tailwind CSS for an interactive and responsive web interface.

## Installation

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/bses7/Loan-Approval-Model.git
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

## Detailed Report

For a comprehensive understanding of the project, including methodology, model development, and deployment steps, refer to the detailed report:

- **[Loan Approval Report](Loan_Approval_Report.pdf)**

This report provides insights into data preprocessing, feature engineering, model selection, and evaluation techniques applied to build a robust loan approval prediction model.

## Project Notebook on Kaggle

The complete code and exploratory data analysis can be found on Kaggle:

- **[Loan Approval Prediction Notebook on Kaggle](https://www.kaggle.com/code/bisheshgiri/loan-approval)**

This notebook includes all the steps from data analysis to model building, allowing for easy replication and further experimentation.

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
  
