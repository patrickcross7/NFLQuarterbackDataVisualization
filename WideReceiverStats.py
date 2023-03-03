import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

df = pd.read_csv("2022WideReceivers.csv")
x = df.drop(columns=["Yards"])
y = df["Yards"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(64, input_shape = x_train.shape, activation = 'sigmoid'))
model.add(tf.keras.layers.Dense(64, activation = 'sigmoid'))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

