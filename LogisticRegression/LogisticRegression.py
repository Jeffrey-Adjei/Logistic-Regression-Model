#Let's build a diabetes prediction model using Logistic Regression

import numpy as np
import	pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def load_dataset():
	data = pd.read_csv('diabetes.csv')
	return data

def define_variables(data):
	x = data.drop('Outcome', axis='columns')
	y = data['Outcome']
	return x, y

def split_dataset(x, y):
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=16)
	return x_train, y_train

def train_logistic_regression_model(x_train, y_train):
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    return logreg

def model_predictions(logreg, x_test):
    y_pred = logreg.predict(x_test)
    return y_pred

def confusion_matrix_evaluation(y_test, y_pred):
    pass





