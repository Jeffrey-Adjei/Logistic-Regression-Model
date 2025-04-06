#Let's build a diabetes prediction model using Logistic Regression

import numpy as np
import	pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

def load_dataset():
	data = pd.read_csv('diabetes.csv')
	return data

def redundant_data_removal(data):
	data.isna().sum()

def define_variables(data):
	x = np.array(data.drop('Outcome', axis='columns'))
	y = np.array(data['Outcome'])
	x_scaled = StandardScaler().fit_transform(x)
	return x_scaled, y

def split_dataset(x, y):
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=16)
	return x_train, x_test, y_train, y_test

def train_logistic_regression_model(x_train, y_train):
	logreg = LogisticRegression()
	logreg.fit(x_train, y_train)
	return logreg

def model_predictions(logreg, x_test):
	y_pred = logreg.predict(x_test)
	return y_pred

def classification_evaluation_report(y_test, y_pred):
	return classification_report(y_test, y_pred)

def confusion_matrix_evaluation(y_test, y_pred):
	conf_matrix = confusion_matrix(y_test, y_pred)
	display = ConfusionMatrixDisplay(conf_matrix, display_labels=["0", "1"])
	plt.figure(figsize= (8, 6))
	display.plot(cmap= plt.cm.Blues)
	plt.title("Confusion Matrix")
	plt.show()

def roc_curve(x_test, y_test, logreg):
	pass

if __name__ == "__main__":
	data = load_dataset()
	x, y = define_variables(data)
	x_train, x_test, y_train, y_test = split_dataset(x, y)
	logreg = train_logistic_regression_model(x_train, y_train)
	y_pred = model_predictions(logreg, x_test)
	print(classification_report(y_test, y_pred))
	confusion_matrix_evaluation(y_test, y_pred)

