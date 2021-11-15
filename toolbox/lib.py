# library of interesting functions


#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
from tensorflow.keras import models, layers


def init_binary_model(input_dim=10):
    """instantiate a basic binary classification neural network model
       accepts an integer, input_dim, used to create first dense layer"""
    model = models.Sequential()
    model.add(layers.Dense(20, activation='relu', input_dim=input_dim))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

if __name__ == '__main__':
    model = init_binary_model()
    print(type(model))
    model.summary()
