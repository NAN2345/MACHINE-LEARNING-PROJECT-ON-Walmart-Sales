#!/usr/bin/env python
# coding: utf-8

# In[59]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import math


# In[2]:


# supress warnings
import warnings
warnings.filterwarnings("ignore")


# #Data Collection and Processing

# In[3]:


#Loading the csv data to a Pandas Dataframe
df=pd.read_csv("Downloads/Walmart_sales.csv")


# In[4]:


df


# In[5]:


#print first 5 rows in the Dataframe
df.head()


# In[6]:


#print last 5 rows of the Dataframe
df.tail()


# In[7]:


#number of rows and columns
df.shape


# In[8]:


df.columns


# In[9]:


#getting some basic informations about the data
df.info()


# In[10]:


#checking the number of missing values
df.isnull().sum()


# In[11]:


#getting the statistical measures of the data
df.describe()


# In[12]:


df.Date.value_counts()


# In[13]:


#
df.Weekly_Sales.value_counts()


# In[14]:


df.Temperature.value_counts()


# In[15]:


df.Fuel_Price.value_counts()


# In[16]:


df.Date.unique()


# In[17]:


df.Weekly_Sales.unique()


# In[18]:


df.Temperature.unique()


# In[19]:


df.Fuel_Price.unique()


# # Visualizing Data

# In[20]:


plt.figure(figsize=(14, 10))

# Line plots for Weekly_Sales,Holiday_Flag,Temperature,Fuel_Price
plt.subplot(2, 2, 1)
sns.lineplot(data=df, x='Date', y='Weekly_Sales', marker='o', color='blue', label='Weekly sales')
plt.title('Date wise weekly sales')

plt.subplot(2, 2, 2)
sns.lineplot(data=df, x='Date', y='Holiday_Flag', marker='o', color='green', label='Holiday flag')
plt.title('Date wise holiday flag')

plt.subplot(2, 2, 3)
sns.lineplot(data=df, x='Date', y='Temperature', marker='o', color='red', label='Temparature')
plt.title('Weekly temparature')

plt.subplot(2, 2, 4)
sns.lineplot(data=df, x='Date', y='Fuel_Price', marker='o', color='purple', label='Fuel price')
plt.title('weekly fuel price')

# Adjusting the layout
plt.tight_layout()
plt.show()


# In[21]:


plt.figure(figsize=(8, 6))
sns.boxplot(x='Holiday_Flag', y='Weekly_Sales', data=df)
plt.title('Weekly Sales by Holiday Flag')
plt.xlabel('Holiday Flag')
plt.ylabel('Weekly Sales')
plt.show()


# In[22]:


dates = df['Date']
cpi_values = df['CPI']
unemployment_values = df['Unemployment']

# Creating the plot
plt.figure(figsize=(10, 6))

# Plotting CPI
sns.lineplot(x=dates, y=cpi_values, label='CPI', color='blue')

# Plotting Unemployment on a separate axis
plt.twinx()
sns.lineplot(x=dates, y=unemployment_values, label='Unemployment', color='red')

# Adding labels and titles
plt.title('CPI and Unemployment Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment')
plt.legend(loc='upper left')
plt.grid(False)

# Show plot
plt.show()


# In[23]:


numerical_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']

# Plot histograms using Seaborn
plt.figure(figsize=(12, 8))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 2, i)
    sns.histplot(data=df, x=feature, kde=True)
    plt.title(f'Histogram of {feature}')
plt.suptitle('Histograms of Numerical Features', y=1.02)
plt.tight_layout()
plt.show()


# In[24]:


plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()



# In[25]:


numerical_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
# Scatter plots for numerical features against Weekly Sales
plt.figure(figsize=(12, 8))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 2, i)
    sns.scatterplot(x=feature, y='Weekly_Sales', data=df)
    plt.title(f'{feature} vs. Weekly Sales')
plt.tight_layout()
plt.show()


# In[26]:


# Assuming your data is stored in a DataFrame named 'df'
X = df.drop(columns=['Weekly_Sales', 'Date'])  # Features
Y = df['Weekly_Sales']  # Target variable


# In[27]:


print(X)


# In[28]:


#importing the scaler
from sklearn.preprocessing import MinMaxScaler
mn= MinMaxScaler()
X=mn.fit_transform(X)


# In[29]:


# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



# In[30]:


print(X_train)


# In[31]:


print(X_test)


# In[32]:


from sklearn.metrics import r2_score, mean_squared_error


# In[33]:


# Build a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest model
r2_rf = r2_score(Y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(Y_test, y_pred_rf))

print(f'R2 Score (Random Forest): {r2_rf}')
print(f'RMSE (Random Forest): {rmse_rf}')


# In[34]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a DataFrame to hold the actual and predicted values
plot_data = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred_rf})

# Plot using Seaborn's lineplot
plt.figure(figsize=(10, 6))
sns.lineplot(data=plot_data, markers=True)
plt.title('Actual vs. Predicted Sales')
plt.xlabel('Data Points')
plt.ylabel('Sales')
plt.grid(True)
plt.show()


# In[60]:


model = DecisionTreeRegressor(random_state=42)


# In[61]:


model.fit(X_train, Y_train)


# In[62]:


Y_pred_1 = model.predict(X_test)


# In[39]:


print("Training Accuracy :", model.score(X_train, Y_train))
print("Testing Accuracy :", model.score(X_test, Y_test))


# In[40]:


mse = mean_squared_error(Y_test, Y_pred_1)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)


# In[41]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a DataFrame to hold the actual and predicted values
plot_data = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred_1})

# Plot using Seaborn's lineplot
plt.figure(figsize=(10, 6))
sns.lineplot(data=plot_data, markers=True)
plt.title('Actual vs. Predicted Sales')
plt.xlabel('Data Points')
plt.ylabel('Sales')
plt.grid(False)
plt.show()


# In[57]:


#calculating the R squared error
error_score=metrics.r2_score(Y_test,Y_pred_1)
print("R squared error:", error_score)


# In[43]:


#calculating errors
mse = metrics.mean_squared_error(Y_test, Y_pred_1)
print("Mean Squared Error:", mse)
mae= metrics.mean_absolute_error(Y_test,Y_pred_1)
print("Mean Absolute Error:", mae)


# In[44]:


pip install xgboost


# In[45]:


from xgboost import XGBRegressor

model_1 = XGBRegressor()
model_1.fit(X_train, Y_train)


# In[46]:


# Make predictions on the testing data
Y_pred_2= model_1.predict(X_test)


# In[47]:


# Evaluate the model
#Calculating the Accuracy
print("Training Accuracy :", model_1.score(X_train, Y_train))
print("Testing Accuracy :", model_1.score(X_test, Y_test))


# In[48]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a DataFrame to hold the actual and predicted values
plot_data = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred_2})

# Plot using Seaborn's lineplot
plt.figure(figsize=(10, 6))
sns.lineplot(data=plot_data, markers=True)
plt.title('Actual vs. Predicted Sales')
plt.xlabel('Data Points')
plt.ylabel('Sales')
plt.grid(False)
plt.show()


# In[49]:


mse = mean_squared_error(Y_test, Y_pred_2)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)


# In[50]:


get_ipython().system('pip install catboost')


# In[51]:


from catboost import CatBoostRegressor


# In[52]:


model_2 = CatBoostRegressor(verbose=False)
model_2.fit(X_train, Y_train)


# In[53]:


# Make predictions on the testing data
Y_pred_3 = model.predict(X_test)


# In[54]:


# Evaluate the model
#Calculating the Accuracy
print("Training Accuracy :", model_2.score(X_train, Y_train))
print("Testing Accuracy :", model_2.score(X_test, Y_test))


# In[55]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a DataFrame to hold the actual and predicted values
plot_data = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred_3})

# Plot using Seaborn's lineplot
plt.figure(figsize=(10, 6))
sns.lineplot(data=plot_data, markers=True)
plt.title('Actual vs. Predicted Sales')
plt.xlabel('Data Points')
plt.ylabel('Sales')
plt.grid(False)
plt.show()


# In[56]:


mse = mean_squared_error(Y_test, Y_pred_3)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)


# In[ ]:





# In[ ]:




