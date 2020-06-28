import numpy as np
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data_dir = '../data/mushrooms.csv'
mushrooms_df = pd.read_csv(data_dir, delimiter=',', encoding='UTF-8')
columns = list(mushrooms_df.columns.values)

sns.pairplot(mushrooms_df, x_vars=columns, y_vars= mushrooms_df['class'], height=10, aspect=0.6)
