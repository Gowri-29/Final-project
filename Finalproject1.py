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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Data Collection
df = pd.read_csv("train.csv")

# Assuming 'df' is your DataFrame
columns_to_drop = ['ID', 'Customer_ID', 'Name', 'Payment_of_Min_Amount']

df_cleaned = df.drop(columns=columns_to_drop)
df = df_cleaned

# Remove Duplicate
duplicate_rows = df[df.duplicated()]
df_unique = df.drop_duplicates()
df = df_unique
    
def EDA(df):
    # Launch d-tale for the DataFrame
    d = dtale.show(df)

    # To view the d-tale instance in a Jupyter notebook, you can use:
    d.open_browser()    

column_names = df.columns.tolist()

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

# Example usage:
# Assuming 'df' is your DataFrame and 'column' is the column name you want to plot
# plot(df, 'your_column_name')


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

df_for_Credit_Risk = df[['Age', 'Annual_Income', 'Num_Bank_Accounts', 'Num_Credit_Card', 
                    'Num_of_Loan', 'Delay_from_due_date', 'Outstanding_Debt', 'Credit_Utilization_Ratio', 
                    'Credit_History_Age', 'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance',
                    'Credit_Score']]

# Select features and target variable
features = df_for_Credit_Risk[['Age', 'Annual_Income', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Num_of_Loan', 'Delay_from_due_date', 'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age', 'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']]
target = df_for_Credit_Risk['Credit_Score'] # You may need to transform 'Credit_Score' into a binary variable based on a threshold

# Extract features and target from the DataFrame
X = features
y = target


def machine_learning_credit_score(X, y, algorithm):
    
    from sklearn.preprocessing import StandardScaler
    SS = StandardScaler()
    X = SS.fit_transform(X)
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
    
    # Initialize the model with specified algorithm
    model = algorithm()
    
    # Fit the model on the training data
    model.fit(X_train, y_train)
    
    # Calculate accuracy for training and testing sets
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    
    return train_accuracy, test_accuracy

def confusion(X, y, algorithm):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

    # Initialize the model with specified algorithm
    model = algorithm()

    # Fit the model on the training data
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(7, 5))
    sns.heatmap(conf_matrix, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title('Confusion Matrix')
    
    # Display the plot using Streamlit
    st.pyplot()

# streamlit:
st.set_option('deprecation.showPyplotGlobalUse', False)
st.header(':blue[_Comprehensive Banking Analytics_]')
tab1,tab2,tab3= st.tabs([":briefcase: Data collection","Data Modelling", "Data prediction"])

with tab1:
    if st.button("EDA"):
        EDA(df)
    if st.button("Missing values"):
        msno.matrix(df)
        st.pyplot()
    if st.button("Data view"):   
        for i in column_names:
            plot(df, i)
    st.write("Due to data sensitive, skewness and outlier is not performed")  
    
Data = []
with tab2:
    Classification_algorithm = {
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "KNeighborsClassifier": KNeighborsClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "LogisticRegression": LogisticRegression,
        "AdaBoostClassifier": AdaBoostClassifier,
        "SVC": SVC,
        "Ridge GaussianNB": GaussianNB
    }
    
    Classification_Selection = st.selectbox("Algorithm selection", list(Classification_algorithm.keys()))

    if st.button("Run Model_Class"):
        X_class, y_class = X, y
        Metrics = machine_learning_credit_score(X_class, y_class, Classification_algorithm[Classification_Selection])
        st.write("Results:", Metrics)
        Data.append(Metrics)
        
    if Data:
        st.header("Evaluation Metrics_Class")
        st.table(Data)
# confusion matrix

# Dropdown to select the regression algorithm for confusion matrix
    Classification_Selection_Confusion_Matrix = st.selectbox("Algorithm selection for Confusion Matrix", list(Classification_algorithm.keys()))

    if st.button("Confusion Matrix_Class"):
        X_class, y_class = X, y
        confusion(X_class, y_class, Classification_algorithm[Classification_Selection_Confusion_Matrix])

with tab3:                
    # Define Regression algorithms
    Classification_algorithms = {
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "KNeighborsClassifier": KNeighborsClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "LogisticRegression": LogisticRegression,
        "AdaBoostClassifier": AdaBoostClassifier,
        "SVC": SVC,
        "Ridge GaussianNB": GaussianNB
    }

    st.write("Enter feature values to predict the credit score:")
    input_features = []
    st.write("For example - Age - 23.0	Annual_Income-19114.12	Num_Bank_Accounts-3.0 Num_Credit_Card-4.0 Num_of_Loan-4.0	Delay_from_due_date -3.0	Outstanding_Debt-809.98	Credit_Utilization_Ratio-26.822620	Credit_History_Age-265.0	Total_EMI_per_month-49.574949	Amount_invested_monthly-21.465380	Monthly_Balance-312.494089	Output -Good")

    for feature_name in ['Age', 'Annual_Income', 'Num_Bank_Accounts', 'Num_Credit_Card', 
                         'Num_of_Loan', 'Delay_from_due_date', 'Outstanding_Debt', 
                         'Credit_Utilization_Ratio', 'Credit_History_Age', 'Total_EMI_per_month', 
                         'Amount_invested_monthly', 'Monthly_Balance']:
        input_value = st.number_input(f"Enter {feature_name}:")
        input_features.append(input_value)
    if len(input_features) == 12:
            selected_model_name = st.selectbox("Select a model", list(Classification_algorithms.keys()))
            selected_model = Classification_algorithms[selected_model_name]()
    else:
            st.error("Please enter all feature values.")          
    if st.button("Predict"):
            X = df_for_Credit_Risk[['Age', 'Annual_Income', 'Num_Bank_Accounts', 
                                    'Num_Credit_Card', 'Num_of_Loan', 'Delay_from_due_date', 
                                    'Outstanding_Debt', 'Credit_Utilization_Ratio', 
                                    'Credit_History_Age', 'Total_EMI_per_month', 
                                    'Amount_invested_monthly', 'Monthly_Balance']]
            y = df_for_Credit_Risk['Credit_Score']
            selected_model.fit(X, y)
            predicted_score = selected_model.predict([input_features])[0]
            st.write("Predicted credit score:", predicted_score)