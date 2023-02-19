
import numpy as np
import pandas as pd
import random


data="C:/Users/omen/Desktop/CuttelFish/CFA_GA_dibetes/Data/diabetes.csv"




def rd_2(lst):
    # Importing the dataset
    dataset = pd.read_csv(data)
    X = dataset.iloc[:, lst].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

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
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_test, y_pred)


def knn_2(lst):
    # Importing the dataset
    dataset = pd.read_csv(data)
    X = dataset.iloc[:, lst].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

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
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_test, y_pred)

def Svm(lst):
    # Importing the dataset
    dataset = pd.read_csv(data)
    X = dataset.iloc[:, lst].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

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
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_test, y_pred)



def nv(lst):
    # Importing the dataset
    dataset = pd.read_csv(data)
    X = dataset.iloc[:, lst].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

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
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_test, y_pred)


def Dt(lst):
    # Importing the dataset
    dataset = pd.read_csv(data)
    X = dataset.iloc[:, lst].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

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
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_test, y_pred)




def rd_all():
    #lst = [random.randint(element) for element in lst
    dataset = pd.read_csv(data)
    y = dataset.iloc[:, 0].values
    dataset=dataset.drop("Class",axis=1)
    X = dataset.iloc[:].values
    print(X)


    # Encoding categorical data
    # Encoding the Independent Variable
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)

    # Taking care of missing data
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imp.fit(X)
    X = imputer.transform(X)


    # Fitting Random Forest Classification to the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=0)

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(classifier, X, y, cv=5)
    return scores.mean()
    #print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


def logistic_2(lst):

    import pandas as pd

    # Importing the dataset
    dataset = pd.read_csv(data)
    X = dataset.iloc[:, lst].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    # Fitting Logistic Regression to the Training set
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0,penalty="l2")
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import accuracy_score
    return  accuracy_score(y_test, y_pred)


def MLP(lst):

    import pandas as pd

    # Importing the dataset
    dataset = pd.read_csv('C:/Users/omen/Desktop/CuttelFish/CFA_GA_dibetes/Data/diabetes.csv')
    X = dataset.iloc[:, lst].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    # Fitting Logistic Regression to the Training set
    from sklearn.neural_network import MLPClassifier
    classifier = MLPClassifier(solver='sgd',max_iter=500, random_state=1)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import accuracy_score
    return  accuracy_score(y_test, y_pred)


def logistic(lst):

    import pandas as pd

    # Importing the dataset
    dataset = pd.read_csv(data)
    X = dataset.iloc[:, lst].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Fitting Logistic Regression to the Training set
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import accuracy_score
    return  round(1- accuracy_score(y_test, y_pred),2)


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
