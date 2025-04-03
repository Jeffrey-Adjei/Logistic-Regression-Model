#Let's build a diabetes prediction model using Logistic Regression

import numpy as np
import	pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report

def load_dataset():
	data = pd.read_csv('diabetes.csv')
	return data

def define_variables(data):
	x = data.drop('Outcome', axis='columns')
	y = data['Outcome']
	return x, y

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
	cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
	target_class =[0,1]
	fig, ax = plt.subplots()
	tick_marks = np.arange(len(target_class))
	plt.xticks(tick_marks, target_class)
	plt.yticks(tick_marks, target_class)

	#plot heatmap
	sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="crest")
	ax.xaxis.set_label_position("top")
	plt.tight_layout()
	plt.title('Confusion matrix')
	plt.ylabel('Actual label')
	plt.xlabel('Predicted label')
	plt.show()

def roc_curve(x_test, y_test, logreg):
	y_pred_probability = logreg.predict(x_test)[::1]
	fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_probability)
	auc = metrics.roc_auc_score(y_test, y_pred_probability)
	plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
	plt.legend(loc=4)
	plt.show()

if __name__ == "__main__":
	data = load_dataset()
	x, y = define_variables(data)
	x_train, x_test, y_train, y_test = split_dataset(x, y)
	logreg = train_logistic_regression_model(x_train, y_train)
	y_pred = model_predictions(logreg, x_test)
	print(classification_report(y_test, y_pred))
	confusion_matrix_evaluation(y_test, y_pred)

