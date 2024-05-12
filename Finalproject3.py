#import necessary libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import missingno as msno
import dtale
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression

df = pd.read_csv("trains_insurance_data.csv")
columns_to_drop = ['id']

df_cleaned = df.drop(columns=columns_to_drop)
df = df_cleaned

df = df.dropna()
    
def EDA(df):
    # Launch d-tale for the DataFrame
    d = dtale.show(df)

    # To view the d-tale instance in a Jupyter notebook, you can use:
    d.open_browser()
    

def plot(df, column):
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Box plot
    sns.boxplot(data=df, x=column, ax=axes[0])
    axes[0].set_title(f'Box Plot for {column}')

    # Distribution plot
    sns.histplot(data=df, x=column, kde=True, bins=50, ax=axes[1])
    axes[1].set_title(f'Distribution Plot for {column}')

    # Violin plot
    sns.violinplot(data=df, x=column, ax=axes[2])
    axes[2].set_title(f'Violin Plot for {column}')

    # Display the figure
    st.pyplot(fig)

column_names = df.columns.tolist()

df_cat = df.select_dtypes(object)
column_Cat_names = df_cat.columns.tolist()

df_encoded = df.copy()

def encode_cat(df, column_name):
    from sklearn.preprocessing import LabelEncoder

    # Instantiate the LabelEncoder object
    le = LabelEncoder()

    # Fit and transform the specified column
    encoded_column = le.fit_transform(df[column_name])

    # Replace the original column with the encoded values
    df[column_name] = encoded_column

    # Return the modified DataFrame
    return df

for column_name in column_Cat_names:
    df_encoded = encode_cat(df_encoded, column_name)
    
outlier_column = ['age_in_days','application_underwriting_score']
df_outiler = df_encoded.copy()    
    
def outlier(df, column):
    iqr = df[column].quantile(0.75) - df[column].quantile(0.25)
    upper_threshold = df[column].quantile(0.75) + (1.5*iqr)
    lower_threshold = df[column].quantile(0.25) - (1.5*iqr)
    df[column] = df[column].clip(lower_threshold, upper_threshold)
    
for i in outlier_column:  
    outlier(df_outiler, i)

df_skewed = df_outiler.copy()

column_for_skew = ['perc_premium_paid_by_cash_credit','Count_3-6_months_late','Count_6-12_months_late','Count_more_than_12_months_late','application_underwriting_score']

def skewness(df, column):
    df_skewed = df.copy()  # Create a copy of the original DataFrame
    df_skewed[column] = np.sqrt(df_skewed[column])  # Apply logarithmic transformation
    return df_skewed  # Return the modified DataFrame

# Example usage
# Assuming df_skewed is your DataFrame and column_for_skew is the column name
df_skewed = skewness(df_skewed, column_for_skew) 

Features = df_skewed.drop(columns=["premium"])
Target = df_skewed["premium"]

X = Features
y = Target

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X = SS.fit_transform(X)

def machine_learning_premium_price(X, y, algorithm, cv=5):
    # Generate some synthetic data for demonstration
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    scores = cross_val_score(algorithm(), X, y, cv=cv, scoring='neg_mean_squared_error', error_score='raise')
    rmse_scores = np.sqrt(-scores)  # Calculate root mean squared error from negative MSE scores
    r2_scores = cross_val_score(algorithm(), X, y, cv=cv, scoring='r2', error_score='raise')  # Calculate R-squared scores

    metrics = {
        'Algorithm': str(algorithm).split("'")[1].split(".")[-1],
        'Cross-Validation RMSE Mean': rmse_scores.mean(),
        'Cross-Validation RMSE Std': rmse_scores.std(),
        'Cross-Validation R2 Mean': r2_scores.mean(),
        'Cross-Validation R2 Std': r2_scores.std()
    }

    return metrics

def prediction_model1(X, y, algorithm, cv=5):
    # Perform cross-validated prediction
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    from sklearn.model_selection import cross_val_predict

    # Define RandomForestRegressor
    algorithm = algorithm()

    # Perform cross-validation to obtain predicted values
    y_pred_rf_cv = cross_val_predict(algorithm, X, y, cv=5)
    
    # Plot true vs predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred_rf_cv, color='blue', alpha=0.5)
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    
    # Get the current figure handle and display it using st.pyplot()
    fig = plt.gcf()
    st.pyplot(fig)

# streamlit:
st.set_option('deprecation.showPyplotGlobalUse', False)
st.header(':blue[_Insurance Analytics and Prediction_]')
tab1, tab2, tab3 = st.tabs([":briefcase: Data collection","Data Modelling", "Data prediction"])

with tab1:
    if st.button("EDA"):
        EDA(df)
    if st.button("Untreated_Skew_Data"):
        for i in df:
            plot(df, i)
    if st.button("Outlier_Treatment"):
        for i in outlier_column:
            plot(df_outiler, i)
        view_data =  df_outiler.head(5)
        st.table(view_data)
    if st.button("Treated_Skew_Data"):
        for i in column_names:
            plot(df_skewed, i)

Data = []

with tab2:
   st.header("Data Modelling")

   Regression_algorithms = {
        "Extra Trees Regressor": ExtraTreesRegressor,
        "Random Forest Regressor": RandomForestRegressor,
        "AdaBoost Regressor": AdaBoostRegressor,
        "Gradient Boosting Regressor": GradientBoostingRegressor,
        "XGBoost Regressor": XGBRegressor,
        "Decision Tree Regressor": DecisionTreeRegressor,
        "Linear Regression": LinearRegression,
        "Ridge Regression": Ridge,
        "Lasso Regression": Lasso,
    }
   Regression_Selection = st.selectbox("Algorithm selection", list(Regression_algorithms.keys()))
   
   if st.button("Run Model_Reg"):
        Metrics = machine_learning_premium_price(X, y, Regression_algorithms[Regression_Selection])
        st.write("Results:", Metrics)
        Data.append(Metrics)
        
   if Data:
        st.header("Evaluation Metrics_Reg")
        st.table(Data)
   
# confusion matrix

  # Dropdown to select the regression algorithm for confusion matrix
   Regression_Selection_Confusion_Matrix = st.selectbox("Algorithm selection for Confusion Matrix", list(Regression_algorithms.keys()))

   if st.button("Confusion Matrix_Reg"):
        prediction_model1(X, y, Regression_algorithms[Regression_Selection_Confusion_Matrix])

with tab3:                
    # Define Regression algorithms
    Regression_algorithms = {
        "Extra Trees Regressor": ExtraTreesRegressor,
        "Random Forest Regressor": RandomForestRegressor,
        "AdaBoost Regressor": AdaBoostRegressor,
        "Gradient Boosting Regressor": GradientBoostingRegressor,
        "XGBoost Regressor": XGBRegressor,
        "Decision Tree Regressor": DecisionTreeRegressor,
        "Linear Regression": LinearRegression,
        "Ridge Regression": Ridge,
        "Lasso Regression": Lasso,
    }
    

    st.write("Enter feature values to predict the credit score:")
    input_features = []
    st.write('''For example - \n 
             perc_premium_paid_by_cash_credit-0.429    age_in_days-12058   Income-355060\n
             Count_3-6_months_late-0.0		Count_6-12_months_late-0.0\n	
             Count_more_than_12_months_late-0.0		  application_underwriting_score-99.02\n
             no_of_premiums_paid-13	      sourcing_channel-2	residence_area_type- 1''')

    for feature_name in ['perc_premium_paid_by_cash_credit',
                        'age_in_days',
                        'Income',
                        'Count_3-6_months_late',
                        'Count_6-12_months_late',
                        'Count_more_than_12_months_late',
                        'application_underwriting_score',
                        'no_of_premiums_paid',
                        'sourcing_channel',
                        'residence_area_type']:
        input_value = st.number_input(f"Enter {feature_name}:")
        input_features.append(input_value)
        
    if len(input_features) == 10:
            selected_model_name = st.selectbox("Select a model", list(Regression_algorithms.keys()))
            selected_model = Regression_algorithms[selected_model_name]()    
    else:
            st.error("Please enter all feature values.")  
    if st.button("Predict"):
            X = Features
            y = Target
            selected_model.fit(X, y)
            predicted_score = selected_model.predict([input_features])[0]
            st.write("Predicted credit score:", predicted_score)
        