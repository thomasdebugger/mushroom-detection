import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns


import os

BATCH_SIZE  = 32
EPOCHS      = 200

data_dir = '/Users/cbml5653/Documents/Cours_energie/mushrooms_analysis/mushroom-detection/data/mushrooms.csv'
mushrooms_df = pd.read_csv(data_dir, delimiter=',', encoding='UTF-8')

target_column = 'class'
feature_columns = mushrooms_df.columns.drop([target_column])

data_ = preprocessing.LabelEncoder()
for column in feature_columns :
    data_unique = mushrooms_df[column].unique()
    data_.fit(data_unique)
    mushrooms_df[column] = data_.transform(mushrooms_df[column] )

data_ = preprocessing.LabelEncoder()
data_.fit([ 'e', 'p'])
mushrooms_df['class']=data_.transform(mushrooms_df['class'])

data_train, data_test = train_test_split(mushrooms_df, test_size=0.2) #0.8 train

#Split in test/training set and  normalize data
x_train = data_train[feature_columns]

y_train = np.array(data_train[target_column])

x_test = np.array(data_test[feature_columns])
#x_test = np.array(valid_data[feature_column])
y_test = np.array(data_test[target_column])

normalizer = preprocessing.Normalizer()

x_train_norm = normalizer.fit_transform(x_train)
x_test_norm = normalizer.transform(x_test)

def build_model(feature_keys_len):
    learning_rate: float = 0.001
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[feature_keys_len]),
        layers.Dense(1, activation='sigmoid')
    ])
    #optimizer = keras.optimizers.RMSprop(0.001)
    adam_opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon = 1e-08, decay=0.0, amsgrad=False) #à reflechir
    model.compile(loss='mean_squared_error',
                  optimizer=adam_opt,
                  metrics=['mean_absolute_error', 'mean_squared_error'])

    return model

early_stop = keras.callbacks.EarlyStopping(monitor='mean_absolute_error', patience=100)

model = build_model(len(x_train[feature_columns].keys()))

fit_model = model.fit(
  x=x_train,
  y=y_train,
  batch_size=BATCH_SIZE,
  epochs=EPOCHS,
  verbose=0,
  use_multiprocessing=True,
  workers=-1)



print('\n# Generate predictions for 3 samples')
print(x_test)
predictions = model.predict(x_test)

y_pred = predictions.astype(int)
print('predictions shape:', y_pred)
print('normal attente :', y_test)
print('diff', y_pred - y_test)

compteur_reussite = 0
for y in y_pred:
    if y == 0:
        compteur_reussite += 1

print("Taux de résussite: ", compteur_reussite / len(y_pred))
matrix = sk.metrics.confusion_matrix(y_test, y_pred)
print(matrix)

cm = sk.metrics.confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm,
                     index = ['e','p'],
                     columns = ['e','p'])

sns.heatmap(cm_df, annot=True)
plt.ylabel('prédit')
plt.xlabel('reelle')
plt.show()