# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
nltk.download('stopwords')


#loading our csv
data = pd.read_csv('spam.csv',encoding='latin1')
del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']

data['v1'] = data['v1'].replace(['ham','spam'],[0,1])
Y_text = data['v1'].as_matrix()
X_text = data['v2'].as_matrix()


# commonly used word (such as “the”, “a”, “an”, “in”) will now be ignored
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
sw = stopwords.words("english")
cv = CountVectorizer(stop_words =sw) ##converting into array. Convert a collection of text documents to a matrix of token counts
tcv = cv.fit_transform(X_text).toarray()##fitting all the words in



from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words=sw,lowercase=True) ##converting our document into matrix again
X = vectorizer.fit_transform(X_text).toarray() ##fitting all the words


#startung analysis for multiple techniques and getting different scores
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y_text, test_size=0.202, random_state=42)##splitting our arrray into random testing and training portion


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
clf = LogisticRegression()
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
print("Logistic Regression",accuracy_score(y_test,pred))


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
print("Naive Bayes",accuracy_score(y_test,pred))


from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(500,500))
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
print("Nueral network",accuracy_score(y_test,pred))


from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
print("DecisionTreeClassifier",accuracy_score(y_test,pred))


from sklearn.svm import SVC
clf = SVC(gamma=0.1,C=1,kernel='rbf')
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
print("SVC",accuracy_score(y_test,pred))
