#cd Desktop/sp19-222-92/project/gamedata

import pandas as pd
import numpy as np

data = pd.read_csv("total_game_data2.csv")
data.head()

# Removes every row where a player had zero minutes played
data = data[data['Min Played'] != 0]

# Removes columns I don't need
data = data[data.columns.drop(list(data.filter(regex='Time')))]
data = data[data.columns.drop(list(data.filter(regex='HR')))]
data = data[data.columns.drop(list(data.filter(regex='Calories')))]
data = data[data.columns.drop(list(data.filter(regex='load score')))]
data = data[data.columns.drop(list(data.filter(regex='Recovery')))]
data = data[data.columns.drop(list(data.filter(regex='Speed zone 1')))]

data.head()


#data['Sprints'].apply(np.float64)
data['Sprints'] = data['Sprints'].apply(np.float64)
data.info()
# Adjust sprints for minutes played
for index, row in data.iterrows():
    print(index)
    # Change # of sprints to sprints per min
    data.at[index, 'Sprints'] = row['Sprints'] / row['Min Played']
    for i in range(5,9):
        data.at[index, i] = (row[i]) / int(row[0])
        
data.head()
data.to_csv("altered_total.csv")

# Normalize
from sklearn import preprocessing

x = data.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data = pd.DataFrame(x_scaled)