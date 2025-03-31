# Customer Churn Prediction Project

This project is a **Customer Churn Prediction** system that uses machine learning to predict whether a customer is likely to churn (leave the service) based on their demographic and service-related information. The project includes data preprocessing, exploratory data analysis, model training, evaluation, and deployment using **Streamlit** for an interactive web application.



## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [How to Use the Streamlit App](#how-to-use-the-streamlit-app)
5. [Model Comparison and Results](#model-comparison-and-results)
6. [File Structure](#file-structure)
7. [Future Improvements](#future-improvements)



## Project Overview

The goal of this project is to predict customer churn using a dataset containing customer demographics, subscription details, and usage patterns. The system leverages multiple machine learning models to identify the best-performing algorithm for churn prediction.

The final model is deployed using **Streamlit**, allowing users to input customer details and receive predictions in real-time.



## Features

- **Data Preprocessing**: Handles missing values, encodes categorical variables, and scales numerical features.
- **Exploratory Data Analysis (EDA)**: Includes correlation heatmaps, visualizations (e.g., histograms, bar plots, pie charts), and group-based analysis.
- **Model Training**: Trains multiple machine learning models:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Decision Tree
  - Random Forest
- **Hyperparameter Tuning**: Uses `GridSearchCV` to optimize model parameters.
- **Model Evaluation**: Evaluates models using metrics such as Accuracy, Precision, Recall, and F1 Score.
- **Deployment**: Deploys the best-performing model using a Streamlit web app.



## Technologies Used

### Programming Languages:
- Python

### Libraries:
- **Data Analysis & Visualization**: `pandas`, `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`, `joblib`
- **Web Deployment**: `streamlit`



## How to Use the Streamlit App

1. Open the Streamlit app in your browser.
2. Enter customer details in the input fields:
   - Age
   - Tenure (number of months as a customer)
   - Monthly Charge (subscription cost)
   - Gender (Male/Female)
3. Click the **Predict!** button.
4. View the prediction result:
   - "Customer is likely to churn."
   - "Customer is unlikely to churn."



## Model Comparison and Results

The following models were trained and evaluated:


| Model                  | Accuracy | Precision | Recall  | F1 Score |
|------------------------|----------|-----------|---------|----------|
| Logistic Regression    | 0.88     | 0.8883    | 0.9887  | 0.9358   |
| K-Nearest Neighbors    | 0.87     | 0.8872    | 0.9774  | 0.9301   |
| Support Vector Machine | 0.885    | 0.8850    | 1.0000  | 0.9389   |
| Decision Tree          | 0.85     | 0.8848    | 0.9548  | 0.9185   |
| Random Forest          | 0.87     | 0.8912    | 0.9718  | 0.9297   |

  
The best-performing model was saved as `model.pkl` for deployment.



## File Structure
```
customer-churn-prediction/
├── app.py 
├── churn_model_training.py 
├── customer_churn_data.csv 
├── scaler.pkl 
├── model.pkl 
```





## Future Improvements

1. Add more advanced machine learning models like Gradient Boosting or XGBoost.
2. Perform feature engineering to extract additional insights from existing data.
3. Improve UI/UX of the Streamlit app for better user experience.
4. Integrate real-time data fetching from a database or API.
5. Deploy the Streamlit app on cloud platforms like AWS, Azure, or Heroku for wider accessibility.



