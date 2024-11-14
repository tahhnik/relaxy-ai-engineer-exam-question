# AI Engineer Technical Assessment - Relaxy Intelligence

![image](https://github.com/user-attachments/assets/b1d535e9-49d5-4bc5-9a9a-c55ebd4a34f7)


## Dear Candidate,

Welcome to the hands-on technical assessment for the AI Engineer position at Relaxy! We're excited to evaluate your MLOps and engineering capabilities through this practical examination.

In this assessment, you'll be working with our Loan Approval prediction system, a real-world small-scale ML pipeline that demonstrates many of the challenges you'll tackle at Relaxy. We've designed this assessment to evaluate not just your coding abilities, but also your understanding of MLOps best practices, system design, and production considerations.

You'll be working on implementing critical production features that would help scale and monitor our ML systems. Take your time to understand the existing codebase before diving into the tasks. Remember, we value clean, maintainable code with proper documentation over quick implementations

## Getting Started with the Codebase

### Context

The Loan Approval Model codebase represents a production-grade ML pipeline that handles loan approval predictions. The system includes modular components for data ingestion, transformation, and model training, built with scalability and maintainability in mind. This foundation provides an excellent starting point for implementing advanced MLOps capabilities.

![svgviewer-png-output (2)](https://github.com/user-attachments/assets/6fdfc7fc-48e9-41be-b327-abc3bd6687bd)



### Initial Setup Instructions

1. **Clone the Repository**
    
    ```bash
    git clone https://github.com/relaxy-intel/relaxy-ai-engineer-exam-question.git
    cd Loan-Approval-Model
    ```
    
2. **Create and Activate Virtual Environment**
    
    ```bash
    sudo apt-get update
    
    sudo apt install python3.8-venv
    
    python -m venv venv
    
    # On Unix/MacOS
    source venv/bin/activate
    ```
    
3. **Install Dependencies**
    
    ```bash
    pip install -r requirements.txt
    ```
    
4. **Prepare Dataset**
    - Place the provided `loan_approval_dataset.csv` in the `dataset` directory
    - Ensure the file permissions are correctly set
5. **Run the Base Pipeline**
    
    ```bash
    python src/main.py
    ```
    
6. **Verify Setup**
    - Check `logs` directory for execution logs
    - Examine `artifacts` directory for pipeline outputs
    - Verify model training completion

### Pipeline Exploration Tips

1. **Code Structure Review**
    - Examine the modular architecture
    - Understand component interactions
    - Review configuration management
    - Study error handling implementation
2. **Output Analysis**
    - Review logging patterns
    - Examine generated artifacts
    - Check model performance metrics
    - Analyze data transformation steps
3. **Experimentation Ideas**
    - Try different model parameters
    - Test with various data samples
    - Modify transformation logic
    - Add new validation steps
4. **Debugging Support**
    - Use logging information
    - Check error messages
    - Review execution flow
    - Monitor resource usage

Remember: Take time to understand the existing implementation before starting your tasks. This will help you make better design decisions for your enhancements.

## Ready to Begin?

Once you're comfortable with the codebase:

1. Review the assessment tasks thoroughly
2. Plan your implementation approach
3. Create a new branch for your work
4. Document your design decisions
5. Implement your solutions
6. Test thoroughly before submission

If you have any questions about the setup or requirements, please don't hesitate to ask. Good luck with your assessment!

# Task 1: Experiment Tracking, Data and Model Versioning Implementation

The first part of your task focuses on implementing comprehensive experiment tracking using MLFlow. You need to set up MLFlow to track all aspects of the model development process. This includes configuring MLFlow to use a local SQLite database for storing experiment data and setting up proper artifact storage. The system should track various model metrics like accuracy, precision, recall, and F1-score, along with training parameters and model artifacts.

![svgviewer-png-output (3)](https://github.com/user-attachments/assets/2367f379-a85d-4a89-a45a-4a31a92cf19b)


A crucial aspect of this implementation is data versioning. As model performance heavily depends on the training data, you need to track and version all data characteristics. This includes logging feature distributions, missing value counts, and any transformations applied to the data. The preprocessing pipeline should be versioned and stored alongside the model artifacts to ensure reproducibility.

After training and evaluating all models, you must identify the best performing model based on the metrices and register it in the MLFlow Model Registry. This involves:

- Registering the model with an appropriate name (e.g., "loan_approval_model")
- Adding detailed model metadata and descriptions
- Tagging the model version appropriately
- Transitioning the model to "Production" stage in the registry
- Ensuring the model's preprocessing pipeline is included in the registered artifact

## Task 2: Model Deployment from Model Registry

Once you have the MLFlow tracking system in place, the second part involves deploying the best performing model as a REST API. You'll need to create a Flask application that serves predictions from your model. The application should include endpoints for both single and batch predictions, along with health checks and model information endpoints.

![svgviewer-png-output (4)](https://github.com/user-attachments/assets/eb7e79ca-84e1-48a4-80b5-6f0ae12c74fb)


The Flask application needs to be containerized using Docker for deployment. This involves creating a proper Dockerfile that includes all necessary dependencies and configurations. The container should be set up to automatically load the latest production model from the MLflow registry. Special attention should be paid to proper error handling, input validation, and request logging to ensure the application is production-ready.

The deployment should follow best practices for performance and security. This includes implementing proper input validation, error handling, and logging mechanisms. The API should be able to handle concurrent requests efficiently and maintain consistent response times. 

### Required API Endpoints Implementation

When implementing the Flask API for the loan approval model, you need to create the following core endpoints:

### 1. Model Prediction Endpoints

### Single Prediction

```
Endpoint: /predict
Method: POST

```

- Handles individual loan application predictions
- Accepts JSON input with loan application features
- Returns prediction with confidence score

### Batch Prediction

```
Endpoint: /predict/batch
Method: POST

```

- Handles multiple loan applications in one request
- Accepts array of loan applications
- Returns predictions for all instances
- Includes batch processing metrics

## 2. Health Check Endpoint

```
Endpoint: /health
Method: GET

```

Purpose:

- Confirms API is operational
- Verifies model is loaded correctly
- Checks database connections (if any)
- Validates dependencies are working

Expected Response:

```json
{
    "status": "healthy",
    "checks": {
        "model_loaded": true,
        "api_status": "ok",
        "model_version": "1.0.0",
        "timestamp": "2024-03-13T10:30:00Z"
    }
}
```

## Implementation Requirements

### Testing the Model API

After implementing your solutions, you can use the following sample request scripts to test the model endpoints:

1. **Single Prediction Test**
    
    ```python
    # test_single_prediction.py
    import requests
    import json
    
    # Sample loan application data
    sample_data = {
        "income_annum": 100000,
        "loan_amount": 500000,
        "loan_term": 20,
        "cibil_score": 750,
        "residential_assets_value": 1200000,
        "commercial_assets_value": 500000,
        "luxury_assets_value": 100000,
        "bank_asset_value": 200000,
        "education": "Graduate",
        "self_employed": "Yes",
        "no_of_dependents": 2
    }
    
    # Test prediction endpoint
    def test_prediction():
        url = "<http://localhost:5000/predict>"
        response = requests.post(url, json=sample_data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if __name__ == "__main__":
        test_prediction()
    
    ```
    
2. **Batch Prediction Test**
    
    ```python
    # test_batch_prediction.py
    import requests
    import json
    import pandas as pd
    import time
    
    # Sample batch data
    batch_data = [
        {
            "income_annum": 100000,
            "loan_amount": 500000,
            "loan_term": 20,
            "cibil_score": 750,
            "residential_assets_value": 1200000,
            "commercial_assets_value": 500000,
            "luxury_assets_value": 100000,
            "bank_asset_value": 200000,
            "education": "Graduate",
            "self_employed": "Yes",
            "no_of_dependents": 2
        },
        {
            "income_annum": 150000,
            "loan_amount": 700000,
            "loan_term": 15,
            "cibil_score": 800,
            "residential_assets_value": 1500000,
            "commercial_assets_value": 800000,
            "luxury_assets_value": 150000,
            "bank_asset_value": 300000,
            "education": "Post Graduate",
            "self_employed": "No",
            "no_of_dependents": 1
        }
        # Add more samples as needed
    ]
    
    def test_batch_prediction():
        url = "<http://localhost:5000/predict/batch>"
    
        # Record start time for latency measurement
        start_time = time.time()
    
        response = requests.post(url, json={"instances": batch_data})
    
        # Calculate latency
        latency = time.time() - start_time
    
        print(f"Status Code: {response.status_code}")
        print(f"Batch Processing Latency: {latency:.2f} seconds")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    
        # If you want to save predictions to CSV
        if response.status_code == 200:
            predictions_df = pd.DataFrame(response.json()['predictions'])
            predictions_df.to_csv('batch_predictions.csv', index=False)
            print("\\nPredictions saved to 'batch_predictions.csv'")
    
    if __name__ == "__main__":
        test_batch_prediction()
    
    ```
    
3. **Expected Response Formats**
    
    Single Prediction:
    
    ```json
    {
        "prediction": 1,
        "probability": 0.85,
        "model_version": "1.0.0",
        "timestamp": "2024-03-13T10:30:00Z"
    }
    
    ```
    
    Batch Prediction:
    
    ```json
    {
        "predictions": [
            {
                "prediction": 1,
                "probability": 0.85,
                "row_id": 0
            },
            {
                "prediction": 0,
                "probability": 0.32,
                "row_id": 1
            }
        ],
        "model_version": "1.0.0",
        "timestamp": "2024-03-13T10:30:00Z",
        "batch_size": 2,
        "processing_time": "0.15s"
    }
    
    ```
    
4. **API Health and Metrics**
    
    ```bash
    # Health check
    curl <http://localhost:5000/health>
    ```
    

### Testing Tips

- Test both single and batch predictions
- Verify response formats and structures
- Monitor batch processing latency
- Test with different batch sizes
- Check error handling for invalid inputs
- Validate all required fields in responses
- Monitor memory usage during batch processing
- Test concurrent batch requests
- Verify metric collection for batch operations

[Rest of the document remains the same]

## Deliverables

For successful completion of these two tasks, you need to submit:

1. A complete implementation of MLFlow tracking integrated with the existing pipeline
2. A Flask application serving the model with all required endpoints
3. Docker configuration files for containerization
4. Comprehensive documentation covering setup and usage
5. Test cases demonstrating the functionality

Your solution should be submitted as a new repository containing all implementation files, documentation, and necessary configuration files. Include screenshots of the MLFlow UI showing tracked experiments and the model registry to demonstrate successful implementation.

## Success Criteria

Your implementation will be evaluated based on:

1. Proper setup and configuration of MLFlow tracking
2. Comprehensive experiment tracking including metrics, parameters, and artifacts
3. Functional model registry with proper versioning strategy
4. Working Flask API with proper error handling and validation
5. Successful Docker implementation with all required configurations
6. Quality of documentation and tests

# Task 2: Implementing Production Model Monitoring and Observability

The loan approval prediction system is currently deployed in production with MLFlow tracking, but lacks real-time monitoring of the model's behavior and performance. Your task is to implement a comprehensive monitoring solution using Prometheus to track model predictions, performance metrics, and system health. This will enable the team to detect issues early and maintain model reliability in production.

![svgviewer-png-output (5)](https://github.com/user-attachments/assets/49ef4989-5c9b-4cf5-a254-11b78e7efe20)


## Part 1: Metrics Implementation

The first part of your task involves instrumenting your Flask application to expose metrics that Prometheus can collect. You need to identify and implement key metrics that provide insights into both model behavior and system performance.

For model monitoring, you need to track prediction patterns, including the distribution of feature values entering the model and the distribution of predictions being made. This helps detect potential data drift or concept drift. You also need to monitor prediction latencies to ensure the model meets performance requirements and track error rates to identify potential issues.

System-level metrics are equally important. These include basic metrics like CPU and memory usage, but also application-specific metrics like request rates and endpoint health. Your implementation should efficiently collect these metrics without significantly impacting system performance.

The metrics endpoint should be properly structured and documented, following Prometheus best practices. You should think carefully about which metric types (counters, gauges, histograms) are appropriate for different measurements, and how to label metrics for effective querying.

## Part 2: Monitoring Infrastructure

Once you have your metrics defined, you need to set up the infrastructure to collect and store them. This involves configuring Prometheus to scrape your metrics endpoint and setting up appropriate storage retention policies.

Your Prometheus configuration should be robust and production-ready. This includes setting appropriate scrape intervals, configuring storage retention, and setting up recording rules for commonly-used queries. Security considerations should be taken into account, such as ensuring metrics are only accessible to authorized users.

A critical component of this part is setting up alerting rules. You need to define thresholds for various metrics that, when crossed, indicate potential issues. These could include high latency, elevated error rates, or unusual prediction patterns. The alerts should be actionable and avoid false positives.

## Deliverables

For successful completion of this task, you need to submit:

1. Updated application code with metric instrumentation
2. Prometheus configuration files
3. Alert rule definitions
4. Documentation covering:
    - Available metrics and their meanings
    - Alert configurations
    - Setup instructions
    - Maintenance procedures

Your solution should be submitted as part of the previous repository containing all implementation files, documentation, and configuration files. Include screenshots of Prometheus graphs showing collected metrics to demonstrate successful implementation.

## Success Criteria

Your implementation will be evaluated based on:

1. Quality and completeness of metric implementation
2. Efficiency of metric collection
3. Appropriateness of alert rules
4. Quality of documentation
5. Overall system impact
6. Security considerations
