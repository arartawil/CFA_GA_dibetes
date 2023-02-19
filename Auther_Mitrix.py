
import numpy as np
import pandas as pd
import random


data="C:/Users/omen/Desktop/CuttelFish/CFA_GA_dibetes/Data/dataset_2.csv"


def rd_2(lst,test_data):
    # Importing the dataset
    dataset = pd.read_csv(data)
    X = dataset.iloc[:, lst].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_data, random_state = 0)

    # Encoding categorical data
    # Encoding the Independent Variable
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_y = LabelEncoder()
    y_train = labelencoder_y.fit_transform(y_train)
    y_test = labelencoder_y.fit_transform(y_test)

    # Fitting Random Forest Classification to the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=0)
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    met=[]
    from sklearn.metrics import mean_absolute_error,cohen_kappa_score
    met.append(mean_absolute_error(y_test, y_pred))
    met.append(cohen_kappa_score(y_test, y_pred))
    return met


def knn_2(lst,test_data):
    # Importing the dataset
    dataset = pd.read_csv(data)
    X = dataset.iloc[:, lst].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_data, random_state = 0)

    # Encoding categorical data
    # Encoding the Independent Variable
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_y = LabelEncoder()
    y_train = labelencoder_y.fit_transform(y_train)
    y_test = labelencoder_y.fit_transform(y_test)

    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    met = []
    from sklearn.metrics import mean_absolute_error, cohen_kappa_score
    met.append(mean_absolute_error(y_test, y_pred))
    met.append(cohen_kappa_score(y_test, y_pred))
    return met

def Svm(lst,test_data):
    # Importing the dataset
    dataset = pd.read_csv(data)
    X = dataset.iloc[:, lst].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_data, random_state = 0)

    # Encoding categorical data
    # Encoding the Independent Variable
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_y = LabelEncoder()
    y_train = labelencoder_y.fit_transform(y_train)
    y_test = labelencoder_y.fit_transform(y_test)

    # Fitting SVM to the Training set
    from sklearn.svm import SVC
    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    met = []
    from sklearn.metrics import mean_absolute_error, cohen_kappa_score
    met.append(mean_absolute_error(y_test, y_pred))
    met.append(cohen_kappa_score(y_test, y_pred))
    return met



def nv(lst,test_data):
    # Importing the dataset
    dataset = pd.read_csv(data)
    X = dataset.iloc[:, lst].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_data, random_state = 0)

    # Encoding categorical data
    # Encoding the Independent Variable
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_y = LabelEncoder()
    y_train = labelencoder_y.fit_transform(y_train)
    y_test = labelencoder_y.fit_transform(y_test)

    # Fitting Naive Bayes to the Training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    met = []
    from sklearn.metrics import mean_absolute_error, cohen_kappa_score
    met.append(mean_absolute_error(y_test, y_pred))
    met.append(cohen_kappa_score(y_test, y_pred))
    return met


def Dt(lst,test_data):
    # Importing the dataset
    dataset = pd.read_csv(data)
    X = dataset.iloc[:, lst].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_data, random_state = 0)

    # Encoding categorical data
    # Encoding the Independent Variable
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_y = LabelEncoder()
    y_train = labelencoder_y.fit_transform(y_train)
    y_test = labelencoder_y.fit_transform(y_test)

    # Fitting Decision Tree Classification to the Training set
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    met = []
    from sklearn.metrics import mean_absolute_error, cohen_kappa_score
    met.append(mean_absolute_error(y_test, y_pred))
    met.append(cohen_kappa_score(y_test, y_pred))
    return met





def logistic_2(lst,test_data):

    import pandas as pd

    # Importing the dataset
    dataset = pd.read_csv(data)
    X = dataset.iloc[:, lst].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_data, random_state = 0)

    # Fitting Logistic Regression to the Training set
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0,penalty="l2")
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    met = []
    from sklearn.metrics import mean_absolute_error, cohen_kappa_score
    met.append(mean_absolute_error(y_test, y_pred))
    met.append(cohen_kappa_score(y_test, y_pred))
    return met




def matrix_metrix(real_values,pred_values,beta):
   from sklearn.metrics import confusion_matrix,accuracy_score
   import math
   CM = confusion_matrix(real_values,pred_values)
   TN = CM[0][0]
   FN = CM[1][0]
   TP = CM[1][1]
   FP = CM[0][1]
   Population = TN+FN+TP+FP
   Prevalence = round( (TP+FP) / Population,2)
   Accuracy   = round( (TP+TN) / Population,4)
   Precision  = round( TP / (TP+FP),4 )
   NPV        = round( TN / (TN+FN),4 )
   FDR        = round( FP / (TP+FP),4 )
   FOR        = round( FN / (TN+FN),4 )
   check_Pos  = Precision + FDR
   check_Neg  = NPV + FOR
   Recall     = round( TP / (TP+FN),4 )
   FPR        = round( FP / (TN+FP),4 )


lst=[0,1,4,7]
print(lst)
for i in [0.5,0.4,0.3,0.2,0.1]:
    print("G+P   GA "+str(i))
    print("RF " + str(rd_2(lst,i)))
    print("KNN " + str(knn_2(lst,i)))
    print("SVM " + str(Svm(lst,i)))
    print("Nv " + str(nv(lst,i)))
    print("DT " + str(Dt(lst,i)))
    print("Log " + str(logistic_2(lst,i)))
    print("\n")