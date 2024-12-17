# XGBoost Regression App

This project is an interactive web application for performing regression analysis using the XGBoost algorithm. Built with the Dash framework, this app allows users to upload a dataset, visualize correlations, analyze categorical variables, train a regression model, and make predictions based on user-provided input.

---

## Features

### 1. Upload Dataset
- Users can upload a CSV file to begin the analysis.
- The application detects categorical and numerical variables in the dataset.
- Provides clear instructions about the dataset and its contents.

### 2. Target Variable Selection
- Choose a target variable (dependent variable) for analysis.
- Automatically updates graphs and components based on the selected target.

### 3. Data Visualization
- **Correlation Graph**: Displays the correlation of numerical features with the selected target variable.
- **Categorical Analysis Graph**: Shows the average value of the target variable for each category in a selected categorical variable.

### 4. Model Training
- Allows users to select features to train an XGBoost regression model.
- Displays model performance metrics:
  - **R² Score**
  - **Root Mean Square Error (RMSE)**

### 5. Predictions
- Users can input feature values to make predictions using the trained model.
- Ensures the input matches the features used during model training.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/hagershahinn/Milestone-4-app.git
   cd Milestone-4-app
  
You can test the app using sample datasets such as:
tips.csv (contains columns like total_bill, tip, sex, smoker, etc.)
ifood_df.csv (marketing data with numerical and categorical variables).
