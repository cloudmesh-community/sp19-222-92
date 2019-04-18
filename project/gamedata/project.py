import pandas as pd
import numpy as np
import matplotlib.pylab as plt

# Used for normalization
from sklearn import preprocessing

# Used to shuffle data
import random

# Used to perform dimensionality reduction
from sklearn.decomposition import PCA

# Used for KMeans clustering
from sklearn.cluster import KMeans

# File name should probably be read in as an argument to a function
datafile = "total_game_data2.csv"

df = pd.read_csv(datafile)
df.head()

# Removes every row where a player had zero minutes played
df = df[df['Min Played'] != 0]

#Reset index for df
df = df.reset_index(drop=True)

#Drop unnecessary columns (one line?)
df = df[df.columns.drop(list(df.filter(regex='Time')))]
df = df[df.columns.drop(list(df.filter(regex='HR')))]
df = df[df.columns.drop(list(df.filter(regex='Calories')))]
df = df[df.columns.drop(list(df.filter(regex='load score')))]
df = df[df.columns.drop(list(df.filter(regex='Recovery')))]
df = df[df.columns.drop(list(df.filter(regex='Speed zone 1')))]

# Shuffle the data
[m,n] = df.shape
df = df.sample(frac=1)

labels = df['Class']
df = df[df.columns.drop(list(df.filter(regex='Class')))]

#Reset index for df
df = df.reset_index(drop=True)

# Column names to use in later loop
distance_columns = ('Distance in Speed zone 2 [yd] (0.10 - 2.59 mph)', 'Distance in Speed zone 3 [yd] (2.60 - 5.13 mph)', 'Distance in Speed zone 4 [yd] (5.14 - 8.38 mph)', 'Distance in Speed zone 5 [yd] (8.39- mph)')
accel_columns = ('Number of accelerations (-50.00 - -3.00 m/s)', 'Number of accelerations (-2.99 - -2.00 m/s)', 'Number of accelerations (-1.99 - -1.00 m/s)', 'Number of accelerations (-0.99 - -0.50 m/s)', 'Number of accelerations (0.50 - 0.99 m/s)', 'Number of accelerations (1.00 - 1.99 m/s)', 'Number of accelerations (2.00 - 2.99 m/s)', 'Number of accelerations (3.00 - 50.00 m/s)')

# Create new column for total accelerations
df['Total accelerations'] = pd.Series(np.random.randn(len(df)), index=df.index)

# Convert column dtypes to floats for percentage values
df['Sprints'] = df['Sprints'].apply(np.float64)
for i in distance_columns:
    df[i] = df[i].apply(np.float64)
for i in accel_columns:
    df[i] = df[i].apply(np.float64)

# Adjust sprints for minutes played, adjust speed zones for total distance, and get total accelerations
for index, row in df.iterrows():
    # Initialize total acceleration value
    total_accels = 0
    # Change # of sprints to sprints per min
    df.at[index, 'Sprints'] = row['Sprints'] / row['Min Played']
    # Divide distance in speed zone by total distance, save back in speed zone column
    for i in distance_columns:
        df.at[index, i] = (row[i]) / int(row['Total distance [yd]'])
    # Calculate total # of accelerations and save in new column
    for i in accel_columns:
        total_accels += row[i]
    df.at[index, 'Total accelerations'] = total_accels

# Second loop to divide each acceleration column by the total accelerations, save back in acceleration columns
for index, row in df.iterrows():
    for i in accel_columns:
        df.at[index, i] = (row[i]) / int(row['Total accelerations'])

# Shows all remaining columns and whether they were dropped or not (tested to maximize scores at the end)
df = df[df.columns.drop(list(df.filter(regex='Total distance')))]
df = df[df.columns.drop(list(df.filter(regex='Distance / min')))]
df = df[df.columns.drop(list(df.filter(regex='Average')))]
df = df[df.columns.drop(list(df.filter(regex='Max')))]
df = df[df.columns.drop(list(df.filter(regex='Sprints')))]
#df = df[df.columns.drop(list(df.filter(regex='zone 2')))]
#df = df[df.columns.drop(list(df.filter(regex='zone 3')))]
df = df[df.columns.drop(list(df.filter(regex='zone 4')))]
#df = df[df.columns.drop(list(df.filter(regex='zone 5')))]
#df = df[df.columns.drop(list(df.filter(regex='-50.00 - -3.00')))]
#df = df[df.columns.drop(list(df.filter(regex='-2.99 - -2.00')))]
#df = df[df.columns.drop(list(df.filter(regex='-1.99 - -1.00')))] ####
df = df[df.columns.drop(list(df.filter(regex='-0.99 - -0.50')))]
df = df[df.columns.drop(list(df.filter(regex='0.50 - 0.99')))]
#df = df[df.columns.drop(list(df.filter(regex='1.00 - 1.99')))] 
#df = df[df.columns.drop(list(df.filter(regex='2.00 - 2.99')))] 
#df = df[df.columns.drop(list(df.filter(regex='3.00 - 50.00')))]
df = df[df.columns.drop(list(df.filter(regex='Min Played')))]
df = df[df.columns.drop(list(df.filter(regex='Total accelerations')))]

df.head()

# Normalize
x = df.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

# Test effects of weighing certain columns
#df[column] = df[column]apply(lambda x: x*weight)

# Save changes in new csv to view later
df.to_csv("altered_total.csv")

# Reduce to two dimensions (uses data from every column, are they all really needed / does this overfit???)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

# Add the labels back to the df
finalDf = pd.concat([principalDf, labels], axis = 1)

# Store dimensions of dataframe and convert to an array
[m,n] = finalDf.shape
df_array = finalDf.values

# Columns to be used for clustering
ind1 = 0; ind2 = 1

X = np.zeros((m,2))
X[:,0] = df_array[:,ind1]
X[:,1] = df_array[:,ind2]

plt.scatter( X[:,0],X[:,1], alpha=0.25 )

def shape_assign(labels):
    shape_labels = []
    for i in labels:
        if(i == 'Forward'):
            shape_labels.append('x')
        elif(i == 'Midfielder'):
            shape_labels.append('^')
        elif(i == 'Defender'):
            shape_labels.append('.')
    return shape_labels

markers = shape_assign(labels)

# Scatter plot customization
plt.rcParams['figure.figsize'] = (15,15) 
plt.rcParams['font.size'] = 25 
plt.rcParams['lines.markersize'] = 7

# KMeans clustering code
kmeans = KMeans( n_clusters=3 ).fit(X)
x = X[:,0]
y = X[:,1]

# Function that lists color values for identifying the kmeans clusters
def assign_color(labels):
    colors = []
    for i in labels:
        if i == 0:
            colors.append('r')
        elif i == 1:
            colors.append('b')
        elif i == 2:
            colors.append('k')
    return colors

# Call function on kmeans labels
col = assign_color(kmeans.labels_)

# Plot each point one by one 
for i in range(len(X)):
    plt.scatter(x[i], y[i], c=col[i], marker=markers[i], alpha=0.5)
    
plt.savefig('cluster_plot.png')

# Function to convert class to a numeric value - 0, 1, or 2
def labels_to_numeric(labels, x, y, z):
    numeric_labels = []
    for i in labels:
        if i == 'Forward':
            numeric_labels.append(x)
        elif i == 'Midfielder':
            numeric_labels.append(y)
        else:
            numeric_labels.append(z)
    return(numeric_labels)

# Convert labels to numeric value
numeric_labels = labels_to_numeric(labels, 0, 1, 2)

from sklearn.metrics.cluster import v_measure_score, completeness_score, homogeneity_score
print("Homogeneity: %0.3f" % homogeneity_score(numeric_labels, kmeans.labels_))
print("Completeness: %0.3f" % completeness_score(numeric_labels, kmeans.labels_))
print("V-measure: %0.3f" % v_measure_score(numeric_labels, kmeans.labels_))