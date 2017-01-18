#!/usr/bin/env python

import pandas as pd
import argparse
from sklearn import linear_model
import csv

parser = argparse.ArgumentParser(description='dataset inputs for titanic kaggle comp')
parser.add_argument('-tr','--train',dest='train_data',help='path to training data')
parser.add_argument('-te','--test',dest='test_data',help='path to test data')

args = parser.parse_args()

def load_data(input_data):
	# load data
	in_data = pd.read_csv(input_data)

	# define features
	f_ = ['Survived','Pclass','Sex','Age','Embarked','SibSp','PassengerId']

	# select features
	if "train" in input_data:
		data = in_data[f_[0:6]]
	else:
		data = in_data[f_[1:7]]
        
	# change male to 1 and female to 0
	data.loc[data['Sex'] == 'male', 'Sex'] = 1
	data.loc[data['Sex'] == 'female', 'Sex'] = 0

	# change embarked (S,C,Q) to (0,1,2)
	data['Embarked'] = data['Embarked'].fillna('S')
	data.loc[data['Embarked'] == 'S', 'Embarked'] = 0
	data.loc[data['Embarked'] == 'C', 'Embarked'] = 1
	data.loc[data['Embarked'] == 'Q', 'Embarked'] = 2

	# fill nan value with mean age
	data['Age'] = data['Age'].fillna(data['Age'].median())

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
	titanic_train = load_data(args.train_data)
	# load test data
	titanic_test = load_data(args.test_data)
	# fit and test model
	logit_model(titanic_train,titanic_train['Survived'],titanic_test)

	print "###"
	print "Model trained and tested. See results.csv"
	print "###"

if __name__ == "__main__":
    main()