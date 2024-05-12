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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
st.set_option('deprecation.showPyplotGlobalUse', False)

# Data Collection
df = pd.read_csv("updated-data.csv")
df = df.drop(columns=["Unnamed: 22"])

def missing_values(df):
    msno.matrix(df)
    st.pyplot()
    sns.heatmap(df.corr(), annot=True)
    st.pyplot()
    
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

columns_to_drop = ['Bank', 'Items']

df_cleaned = df.drop(columns=columns_to_drop)
df = df_cleaned

column_names = df.columns.tolist()

df_skewed = df.copy()

# Define a small constant value
epsilon = 1e-9

# Apply natural logarithm transformation to selected columns to reduce skewness
df_skewed['No. of employees'] = np.log(df_skewed['No. of employees'] + epsilon)
df_skewed['No. of offices'] = np.log(df_skewed['No. of offices'] + epsilon)
df_skewed['Deposits'] = np.log(df_skewed['Deposits'] + epsilon)
df_skewed['Investments'] = np.log(df_skewed['Investments'] + epsilon)
df_skewed['Advances'] = np.log(df_skewed['Advances'] + epsilon)
df_skewed['Interest income'] = np.log(df_skewed['Interest income'] + epsilon)
df_skewed['Other income'] = np.log(df_skewed['Other income'] + epsilon)
df_skewed['Interest expended'] = np.log(df_skewed['Interest expended'] + epsilon)
df_skewed['Operating expenses'] = np.log(df_skewed['Operating expenses'] + epsilon)
df_skewed['CRAR'] = np.log(df_skewed['CRAR'] + epsilon)
df_skewed['Net NPA ratio'] = np.log(df_skewed['Net NPA ratio'] + epsilon)

# Drop rows with missing values after transformation
df_skewed = df_skewed.dropna()

outlier_column = ['No. of employees','Business per employee','Profit per employee','Capital and Reserves & Surplus','Deposits','Investments','Advances','Interest income',
                'Other income','Net Interest Margin','Cost of Funds (CoF)','Return on advances adjusted to CoF','Wages as % to total expenses','Return on Equity','Return on Assets','CRAR','Net NPA ratio']
df_outiler = df_skewed.copy() 

def outlier(df, column):
    iqr = df[column].quantile(0.75) - df[column].quantile(0.25)
    upper_threshold = df[column].quantile(0.75) + (1.5*iqr)
    lower_threshold = df[column].quantile(0.25) - (1.5*iqr)
    df[column] = df[column].clip(lower_threshold, upper_threshold)
    

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_outiler)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_data)

# Add cluster labels to the DataFrame
df_outiler['Cluster'] = kmeans.labels_

# Extract features from the DataFrame
X = df_outiler.values

# Apply dimensionality reduction (PCA) to reduce the number of features
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
X_pca = pca.fit_transform(X)

# Assign cluster labels using K-means clustering
labels = kmeans.labels_

# Define the mapping from numerical cluster labels to corresponding labels
cluster_label_mapping = {0: 'Premium Customers', 1: 'Standard Customers', 2: 'Low-Value Customers'}

# Replace numerical cluster labels with corresponding labels
df_outiler["Cluster"] = df_outiler["Cluster"].replace(cluster_label_mapping)

Features = df_outiler.drop(columns=["Cluster"])
Target = df_outiler["Cluster"]

X = Features
y = Target
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X = SS.fit_transform(X)

def machine_learning_Banking_Performance(X, y, algorithm):
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

def predict_probability(input_features, model):
    # Convert input features to a 2D array
    input_data = [input_features]

    # Make prediction
    predicted_value = model.predict(input_data)
    
    return predicted_value[0]

def confusion(X,y,algorithm):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

    # Initialize the model with specified algorithm
    model = algorithm()

    # Fit the model on the training data
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)


    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix for Logistic Regression
    plt.figure(figsize=(7, 5))
    sns.heatmap(conf_matrix, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title('Confusion Matrix for Logistic Regression')
    st.pyplot()
    
    
# streamlit:
st.set_option('deprecation.showPyplotGlobalUse', False)
st.header(':blue[_Performance Prediction_]')
tab1, tab2, tab3 = st.tabs([":briefcase: Data collection","Data Modelling", "Data prediction"])

with tab1:
    if st.button("EDA"):
        EDA(df)
    if st.button("Untreated_Skew_Data"):
        for i in df:
            plot(df, i)
    if st.button("Treated_Skew_Data"):
        for i in column_names:
            plot(df_skewed, i)
    if st.button("Outlier_Treatment"):
        for i in outlier_column:
            plot(df_outiler, i)
        view_data =  df_outiler.head(5)
        st.table(view_data)
        # Visualize the clusters in reduced dimensions
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=50, cmap='viridis')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('K-means Clustering (PCA)')
        st.pyplot()
Data = []

with tab2:
   st.header("Data Modelling")

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
        Metrics = machine_learning_Banking_Performance(X_class, y_class, Classification_algorithm[Classification_Selection])
        st.write("Results:", Metrics)
        Data.append(Metrics)
        
   if Data:
        st.header("Evaluation Metrics_Class")
        st.table(Data)
   
# confusion matrix

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
    st.write("For example - 928,11439,55.54,0.36,20465,392244,109988,298507,38103,5771,27071,7874,2.52,6.66,4.22,13.19,21.46,0.92,14.52,0.85")

    for feature_name in ['No. of offices','No. of employees','Business per employee','Profit per employee','Capital and Reserves & Surplus','Deposits','Investments',
                        'Advances','Interest income','Other income','Interest expended','Operating expenses','Net Interest Margin',
                        'Cost of Funds (CoF)','Return on advances adjusted to CoF','Wages as % to total expenses','Return on Equity',
                        'Return on Assets','CRAR','Net NPA ratio']:
        input_value = st.number_input(f"Enter {feature_name}:")
        input_features.append(input_value)
    
    if len(input_features) == 20:
            selected_model_name = st.selectbox("Select a model", list(Classification_algorithms.keys()))
            selected_model = Classification_algorithms[selected_model_name]()
    else:
            st.error("Please enter all feature values.")
            
                    
    if st.button("Predict"):
            X = df_outiler[['No. of offices','No. of employees','Business per employee','Profit per employee','Capital and Reserves & Surplus','Deposits','Investments',
                        'Advances','Interest income','Other income','Interest expended','Operating expenses','Net Interest Margin',
                        'Cost of Funds (CoF)','Return on advances adjusted to CoF','Wages as % to total expenses','Return on Equity',
                        'Return on Assets','CRAR','Net NPA ratio']]
            y = df_outiler["Cluster"]
            selected_model.fit(X, y)
            predicted_score = selected_model.predict([input_features])[0]
            st.write("Predicted credit score:", predicted_score)
        