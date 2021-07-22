
# Packages / libraries
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
from math import sqrt

import pathlib


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers




dataset_path = keras.utils.get_file("auto-mpg.data", 
                                    "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin'] 
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()

dataset = dataset.dropna()

dataset = dataset[['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
       'Acceleration', 'Model Year']]

dataset['Weight [kg]'] = dataset['Weight']*0.453592

dataset['Displacement [l]'] = dataset['Displacement']*0.0163871

dataset['Fuel consumption [km/l]'] = dataset['MPG']*0.425144

dataset = dataset.drop(['MPG'], axis=1)
dataset = dataset.drop(['Displacement'], axis=1)
dataset = dataset.drop(['Weight'], axis=1)

for column in dataset:
    unique_vals = np.unique(dataset[column])
    nr_values = len(unique_vals)
    if nr_values < 10:
        print('The number of values for feature {} :{} -- {}'.format(
            column, nr_values,unique_vals))
    else:
        print('The number of values for feature {} :{}'.format(
            column, nr_values))

# Running Correlation

hm = dataset.corr()

# Can be great to plot only a half matrix
# Generate a mask for the upper triangle
mask = np.zeros_like(hm)
mask[np.triu_indices_from(mask)] = True

# Draw the heatmap with the mask
plt.figure('Heatmap MPG')
plt.clf()
plt.yticks(va="center")
plt.xticks(rotation=360)
#Visualizing Correlation with a Heatmap
sns.heatmap(hm, mask=mask, square=True,
            annot = True, annot_kws={'size':20}, cmap="Blues")

# Split the data into X & y

X = dataset.drop('Fuel consumption [km/l]', axis = 1).values
y = dataset['Fuel consumption [km/l]']

# Hold-out validation

# first one
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size = 0.7,
                                                    test_size=0.3, random_state=10)

# Second one
X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                      y_train,
                                                      train_size = 0.9,
                                                      test_size=0.1, random_state=10)


forest_reg = RandomForestRegressor(random_state=10)

forest_reg.fit(X_train, y_train)

print("Training accuracy: ", forest_reg.score(X_train, y_train))

y_pred = forest_reg.predict(X_test)

print("Testing accuracy: ", forest_reg.score(X_test, y_test))

forest_rmse = np.sqrt(mean_squared_error(y_pred, y_test))
print("RMSE: ",forest_rmse)
print('\n')


lm = LinearRegression(fit_intercept = True)
lm.fit(X_train, y_train)

y_pred = forest_reg.predict(X_train)
y_pred_val = forest_reg.predict(X_valid)

# Model Accuracy on training dataset

print('The Accuracy on the training dataset is: ', forest_reg.score(X_train, y_train) )
print('The Accuracy n2  on the training dataset is: ',r2_score(y_train,y_pred) )   



print("")
# Model Accuracy on testing dataset
print('The Accuracy on the testing dataset is: ', forest_reg.score(X_test, y_test) )
print('The Accuracy on the validation dataset is: ', forest_reg.score(X_valid, y_valid) )

print("")
# The Root Mean Squared Error (RMSE)
print('The RMSE on the training dataset is: ',
      sqrt(mean_squared_error(y_train,y_pred)))
print('The RMSE on the testing dataset is: ',
      sqrt(mean_squared_error(y_test,forest_reg.predict(X_test))))
print('The RMSE on the validation dataset is: ',
      sqrt(mean_squared_error(y_valid,forest_reg.predict(X_valid))))

print("")
# The Mean Absolute Error (MAE)
print('The MAE on the training dataset is: ',
      mean_absolute_error(y_train,y_pred))
print('The MAE on the testing dataset is: ',
      mean_absolute_error(y_test,forest_reg.predict(X_test)))
print('The MAE on the validation dataset is: ',
      mean_absolute_error(y_test,forest_reg.predict(X_test)))


