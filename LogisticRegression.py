### Logistic Regression implementation on the dataset
"""
Breast cancer Wisconsin dataset 
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score,accuracy_score

'''
1.import data
'''


names=['ID', 'ClumpTkns', 'UnofCSize', 'UnofCShape', 'MargAdh', 
'SngEpiCSize', 'BareNuc', 'BlandCrmtn', 'NrmlNuc', 'Mitoses', 'Class' ]
Data=pd.read_csv('WBCD.csv', names=names)

 


'''
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                 test_size=0.3, random_state=42)
'''

'''
train['BareNuc'].replace(?, np.nan, inplace=true)

'''


Data['BareNuc']=Data['BareNuc'].fillna(Data["BareNuc"].median())

'''
3.data precessing
'''


#select features and Normalization
x=Data.loc[:,['ClumpTkns', 'UnofCSize', 'UnofCShape', 'MargAdh', 'SngEpiCSize', 'BareNuc', 'BlandCrmtn', 'NrmlNuc', 'Mitoses']]
y=Data['Class']


min_max_scaler = preprocessing.MaxAbsScaler()
x_minmax = min_max_scaler.fit_transform(x)



#print(x_minmax)

'''
4.train model and performed testing using logistic regression
'''

# using Logistic regression
print("Logistic regression")

clf=LogisticRegression(random_state=1,solver='liblinear')

#print("AdaBoostClassifier")
#clf=AdaBoostClassifier()

#clf.fit(x_minmax,y)


scores = cross_val_score(clf, x, y, cv=7)
print(scores)
print("mean = ",np.mean(scores))


'''
print("Train-Accuracy=",accuracy_score(train_y,train_predict))

#print("AUC=",roc_auc_score(test_y, test_predict))

'''
#submission

submission = pd.DataFrame({
        "ID": testData["ID"],
        "Class": test_predict
    })
submission.to_csv('wresult.csv', index=False)
