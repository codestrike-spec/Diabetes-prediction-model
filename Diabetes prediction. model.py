#!/usr/bin/env python
# coding: utf-8

# In[100]:


#importing libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt


# In[101]:


#data collection and data analysis
diabetes_dataset=pd.read_csv('diabetes.csv')


# In[102]:


diabetes_dataset.head()


# In[103]:


#number of rows and Columns in this dataset
diabetes_dataset.shape


# In[104]:


# getting the statistical measures of the data
diabetes_dataset.describe()


# In[105]:


diabetes_dataset['Outcome'].value_counts()


# In[106]:


# 0 --> Non-Diabetic

# 1 --> Diabetic

diabetes_dataset.groupby('Outcome').mean()


# In[113]:


df=diabetes_dataset
df.head()


# In[114]:


#let's analyze the skewness in the data
print('KDE Plots for all the columns')
for column in df.columns:
    sns.kdeplot(df[column], label=column, fill=True)
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.legend(title='Column Name')
    print(df[column].skew())
    plt.show()


# In[ ]:


#let's reduce the skewness





# In[115]:


#column with positive significant skewness
df['Insulin'] = np.log1p(df['Insulin'])
df['Age'] = np.log1p(df['Age'])
df['DiabetesPedigreeFunction'] = np.log1p(df['DiabetesPedigreeFunction'])
df['SkinThickness'] = np.log1p(df['SkinThickness'])


# In[116]:


#column with negativet skewness
df['BloodPressure'] = np.log1p(df['BloodPressure'].max() - df['BloodPressure'])
df['BMI'] = np.log1p(df['BMI'].max() - df['BMI'])


# In[117]:


#columns with moderate skewness
df['Pregnancies'] = np.sqrt(df['Pregnancies'])


# In[118]:


#let's analyze the skewness again after reducing it the data
print('KDE Plots for all the columns')
for column in df.columns:
    sns.kdeplot(df[column], label=column, fill=True)
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.legend(title='Column Name')
    print(df[column].skew())
    plt.show()


# In[137]:


# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']


# In[ ]:





# In[138]:


df.describe()


# In[139]:


#let's standardized the data 




# In[140]:


scaler = StandardScaler()


# In[141]:


scaler.fit(X)


# In[142]:


standardized_data = scaler.transform(X)


# In[143]:


print(standardized_data)


# In[144]:


standardized_data[0]


# In[145]:


#coverting numpy array of data into dataframe for easy handelling

# Convert array to DataFrame
standardized_df = pd.DataFrame(standardized_data, columns=['Pregnancies', 'Glucose', 'BloodPressure','SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age' ])


# In[146]:


df2=standardized_df
df2.head()


# In[147]:


print('KDE Plots for all the columns')
for column in df2.columns:
    sns.kdeplot(df2[column], label=column, fill=True)
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.legend(title='Column Name')
    print(df2[column].skew())
    plt.show()


# In[148]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)


# In[149]:


print(X.shape, X_train.shape, X_test.shape)


# In[150]:


classifier = svm.SVC(kernel='linear')


# In[151]:


#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)


# In[152]:


# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[153]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[154]:


# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[155]:


print('Accuracy score of the test data : ', test_data_accuracy)


# In[ ]:


#### have to do after this


# In[ ]:





# In[27]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the diabetes dataset
diabetes_dataset = pd.read_csv('diabetes.csv')

# Separate features and target variable
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[28]:


#linear regression 

from sklearn.linear_model import LinearRegression

# Initialize and train the model
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, Y_train)

# Make predictions and calculate accuracy
linear_regression_predictions = linear_regression_model.predict(X_test)
# Convert predictions to binary (0 or 1)
linear_regression_predictions_binary = [1 if pred >= 0.5 else 0 for pred in linear_regression_predictions]
linear_regression_accuracy = accuracy_score(Y_test, linear_regression_predictions_binary)
print(f"Linear Regression Model Accuracy: {linear_regression_accuracy * 100:.2f}%")


# In[29]:


#logistic regression

from sklearn.linear_model import LogisticRegression

# Initialize and train the model
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, Y_train)

# Make predictions and calculate accuracy
logistic_regression_predictions = logistic_regression_model.predict(X_test)
logistic_regression_accuracy = accuracy_score(Y_test, logistic_regression_predictions)
print(f"Logistic Regression Model Accuracy: {logistic_regression_accuracy * 100:.2f}%")


# In[30]:


#decision tree

from sklearn.tree import DecisionTreeClassifier

# Initialize and train the model
decision_tree_model = DecisionTreeClassifier(random_state=2)
decision_tree_model.fit(X_train, Y_train)

# Make predictions and calculate accuracy
decision_tree_predictions = decision_tree_model.predict(X_test)
decision_tree_accuracy = accuracy_score(Y_test, decision_tree_predictions)
print(f"Decision Tree Model Accuracy: {decision_tree_accuracy * 100:.2f}%")


# In[36]:


print("Model Comparison:")
print(f"Linear Regression Accuracy: {linear_regression_accuracy * 100:.2f}%")
print(f"Logistic Regression Accuracy: {logistic_regression_accuracy * 100:.2f}%")
print(f"Decision Tree Accuracy: {decision_tree_accuracy * 100:.2f}%")
print(f"SVM model accuracy : {test_data_accuracy  * 100:.2f}%")


# In[ ]:




