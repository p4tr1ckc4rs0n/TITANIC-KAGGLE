#!/usr/bin/env python

import pandas as pd
import scipy
from sklearn import linear_model
import csv

def load_data(input_data):
    
	# load data
	in_data = pd.read_csv(input_data)

	# define features
	f_ = ['Survived','Pclass','Sex','Age','PassengerId']

	# select features
	if "train" in input_data:
		data = in_data[f_[0:4]]
	else:
		data = in_data[f_[1:5]]
        
	# change male to 1 and female to 0
	data['Sex'] = data['Sex'].apply(lambda sex: 1 if sex=='male' else 0)

	# fill nan value with mean age
	data['Age'] = data['Age'].fillna(data_train['Age'].mean())

	return data

def logit_model(X_train,Y_train,test_data):
    
	# instantiate model
	logreg = linear_model.LogisticRegression()

	# train and fit model
	logreg.fit(X_train.drop(['Survived'],axis=1),Y_train)

	# apply model to test data
	predicted = logreg.predict(test_data.drop(['PassengerId'],axis=1))

	# write predicted results to csv file
	with open('results.csv', 'wb') as csvfile:
		outfile = csv.writer(csvfile)
		outfile.writerow(('PassengerId','Survived'))
		for i,j in zip(test_data['PassengerId'],predicted):
			outfile = csv.writer(csvfile)
			outfile.writerow((i,j))

def main():
	# load training data
	titanic_train = load_data(args.train)
	# load test data
	titanic_test = load_data(args.test)
	# fit model
	logit_model(titanic_train,titanic_train['Survived'],titanic_test)