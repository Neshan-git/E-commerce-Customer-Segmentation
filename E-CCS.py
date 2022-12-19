import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import random


# DATA COLLECTION & ANALYSIS :-

# Loading the data from csv file to a Pandas DataFrame :-
full_data = pd.read_csv('cust_data.csv')
print(full_data.columns)

# Finding the number of rows and columns
a = full_data.shape
print(a)

# Getting some information about the dataset
full_data.info()

# First 100 rows in the dataframe :-
x = full_data.head(100)

# Checking for missing values
b = full_data.isnull().sum()
print(b)

# GROPING METHOD :-
# Training the k-Means Clustering Model:-

a = input("Enter the brand name : ")

df = x.drop(['Gender', 'Orders'], axis=1)
kmeans = KMeans(n_clusters=3)
kmeans.fit(df)

plt.scatter(df['Cust_ID'], df[a], c=kmeans.labels_)
plt.title('customer Groups')
plt.xlabel('Cust_ID')
plt.ylabel('No of times visited')
plt.show()

# Plotting the L-blow Graph

g = []
h = []

for i in range(1, 11):
    km = KMeans(n_clusters=i)
    km.fit_predict(df)
    g.append(km.inertia_)
    h.append(i)
plt.plot(g, h)
plt.title('The Elbow Point Graph')
plt.xlabel('Cust_ID')
plt.ylabel('No of times visited')
plt.show()

