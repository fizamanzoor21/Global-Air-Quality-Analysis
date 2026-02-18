🌍 Global Air Quality Analysis
Introduction to Data Science (IDS) – Complex Computing Problem

A comprehensive data science project analyzing global air pollution patterns and building predictive machine learning models for Air Quality Index (AQI) assessment.

📌 Project Overview

Air pollution is one of the most serious global environmental and public health challenges.

This project analyzes a Global Air Quality Dataset (10,000 observations) containing pollutant concentrations and meteorological variables from major cities worldwide.

The project follows a complete data science pipeline:

Data Cleaning & Preprocessing

Exploratory Data Analysis (EDA)

Feature Engineering

Machine Learning Modeling

Model Evaluation

Interpretation & Environmental Recommendations

🎯 Problem Statement

To analyze global air quality data and build predictive models for AQI assessment using pollutant and meteorological variables.

Since the dataset did not contain an official AQI column, a proxy AQI was constructed using a weighted composite formulation.

📊 Dataset Information
Pollutants:

PM2.5

PM10

NO₂

SO₂

CO

O₃

Meteorological Variables:

Temperature

Humidity

Wind Speed

Target Variable:

AQI (constructed proxy)

🧹 Data Preprocessing & Cleaning

✔ Missing Value Handling:

City-wise time-series interpolation

Forward/backward filling

Median imputation (fallback)

✔ Outlier Treatment:

IQR-based clipping method

✔ AQI Construction:

Weighted linear combination of normalized pollutants

Clipped to standard AQI range (0–500)

✔ AQI Categorization:

Good

Moderate

Unhealthy

Hazardous

✔ Feature Scaling:

StandardScaler (Mean = 0, Std = 1)

✔ Train-Test Split:

80% Training

20% Testing

Fixed random seed

📈 Exploratory Data Analysis (EDA)

Key Insights:

PM2.5 and PM10 show strong positive correlation with AQI

Wind speed negatively correlated with AQI (dispersion effect)

Seasonal pollution peaks observed

Higher pollution levels during weekdays

Strong inter-correlation between particulate matter variables

🤖 Machine Learning Models Implemented
🔹 Regression Models (AQI Prediction)

Linear Regression

Ridge Regression

Lasso Regression

Decision Tree Regressor

Random Forest Regressor

Gradient Boosting Regressor

🔹 Classification Models (AQI Category Prediction)

Logistic Regression

Random Forest Classifier

All models implemented using pipelines to prevent data leakage.

📊 Model Evaluation Metrics

R² Score

RMSE

MAE

Cross-Validated R²

Precision

Recall

F1-score

🏆 Key Results
Best Regression Model:

Linear Regression

R² ≈ 1.00

Extremely low RMSE & MAE

Perfect recovery of AQI structure

Reason:
AQI was constructed as a weighted linear combination, making linear models naturally optimal.

Best Classification Model:

Logistic Regression

99.8% Accuracy

Strong minority class recall

Balanced performance

🔍 Feature Importance

PM2.5 and PM10 are dominant contributors

NO₂ significantly impacts AQI

Wind speed reduces AQI levels

Temperature moderately influences ozone

🏥 Health Impact Insights

PM2.5 → Cardiovascular & respiratory diseases

NO₂ → Asthma & lung inflammation

O₃ → Reduced respiratory efficiency

🌱 Environmental Recommendations
Short-Term:

Predictive pollution alerts

Traffic restrictions on high-AQI days

Medium-Term:

Industrial emission control

Urban greenery expansion

Long-Term:

Renewable energy transition

Smart city air quality monitoring systems

⚠ Limitations

AQI is a proxy (not regulatory standard)

City-level aggregation limits micro-analysis

No direct health outcome data

🛠 Tech Stack

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-Learn

Jupyter Notebook


📌 Project Highlights

Complete Data Science Workflow

Advanced Data Cleaning Strategy

Multiple ML Model Comparison

Strong Model Evaluation

Real-World Environmental Interpretation

Complex Computing Problem (CCP Attributes A2, A3, A8 Covered)
