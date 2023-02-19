#importing libraries
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def corr(lst):
    count = lst.count(1)
    if(count==4):
        df = pd.read_csv('C:/Users/omen/Desktop/CuttelFish/CFA_GA_dibetes/Data/diabetes.csv')
        X = df.drop("Outcome",axis=1)   #Feature Matrix
        y = df["Outcome"]
        lst_index=[]
        for i in range(len(lst)):
            if lst[i] == 1:
                lst_index.append(i)

        lst_index.append(8)

        df = df.iloc[:, lst_index]
        cor = df.corr()
        #Correlation with output variable
        cor_target = abs(cor["Outcome"])#Selecting highly correlated features

        sum=0
        for i in cor_target:
            sum+=round(i,4)

        return 1-(sum/100)
    else:
        return 10000