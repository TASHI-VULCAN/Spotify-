#!/usr/bin/env python
# coding: utf-8

# Spotify 2023 songs: Predective modelling 

# In[ ]:


import kagglehub
import pandas as pd
import numpy

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[ ]:


path = kagglehub.dataset_download("nelgiriyewithana/top-spotify-songs-2023")
print("Path to dataset files:", path)


# In[ ]:


df = pd.read_csv(r"C:\Users\tdawa\.cache\kagglehub\datasets\nelgiriyewithana\top-spotify-songs-2023\versions\1\spotify-2023.csv", encoding='ISO-8859-1')
df.head()


# In[ ]:


#Checking basic information (data types and null values)
df.info()
#summary of statistics for numerical columns
df.describe().round(1)


# In[ ]:


#auto adjusting the Dtype if required
df = df.infer_objects()
df.info()


# In[ ]:


missing_values = df.isnull().sum()
print(missing_values)


# In[ ]:


df = df.dropna()
df.info()


# In[ ]:


#Histogram of a numerical column 
df['bpm'].hist()
plt.title('Popularity Distribution')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


#Bar plot for a categorical column 
x = df['released_year'] #.value_counts()
y = df['in_spotify_charts']
plt.title('Charts by year realeased')
plt.xlabel('released_year')
plt.ylabel('in_spotify_charts')
plt.bar(x,y,align='center',width=0.8)

plt.show()


# In[ ]:


#Plotting two different scatter plots in the same graph

#plot 1
x = df['bpm']
y = df['in_spotify_charts']

print("Min: ",min(y), "Max: ",max(y))
plt.scatter(x, y, color='blue')
plt.title('Relation between bpm and in_spotify_charts')
plt.xlabel('bpm')
plt.ylabel('in_spotify_charts')

#plot 2
x = df['bpm']
y = df['energy_%']

print("Min: ",min(y), "Max: ",max(y))
plt.scatter(x, y, color= 'green')
plt.title('Relation between bpm and energy_%')
plt.xlabel('bpm')
plt.ylabel('energy_%')

plt.show()


# In[ ]:


#Correlation matrix

correlation_matrix = df['bpm','energy_%'].corr()

#Heatmap of the correlation matrix
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

#Need to clean the data and convert it into integers or floats first, 
# then we can compare which features have the best correleation to each other.


# In[ ]:


#identify outliers using boxplots
sns.boxenplot(df['bpm'])
plt.title('Boxplot of popularity')
plt.show()


# In[ ]:


#Scatter plot to explore replationships

sns.scatterplot(data=df[['streams','bpm']], x='streams', y='bpm')
plt.title('Streams vs Popularity')
plt.show()


# In[ ]:


#Data preparation for ML
#one hot encoding
df = pd.get_dummies(df, columns=['mode'], drop_first=True)
print(df[['mode']])


# In[ ]:


scaler = StandardScaler()
numerical_columns = ['streams', 'duration_ms', 'tempo']  # Example columns
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])


# In[ ]:


# ML model

y = df['streams'] #Target variable

X = df[['bpm','energy_%']]

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


# Initialize the model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)


# In[ ]:


# Make predictions on the test set
y_pred = model.predict(X_test)


# In[ ]:


# Calculate R² score to evaluate the model
r2 = r2_score(y_test, y_pred)
print(f'R² Score: {r2}')


# In[ ]:


# Calculate MSE and RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f'MSE: {mse}, RMSE: {rmse}')


# In[ ]:


# Scatter plot for actual vs predicted values
plt.scatter(y_test,y_pred, alpha=0.3,color='red')
plt.xlabel('Actual Popularity')
plt.ylabel('Predicted Popularity')
plt.title('Actual vs Predicted Popularity')
plt.show()


# In[ ]:


residuals = y_test - y_pred

# Plot residuals
plt.hist(residuals, bins=20, color='blue', edgecolor='black')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Residuals Distribution')
plt.show()

