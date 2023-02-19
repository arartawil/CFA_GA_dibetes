from CFA import CFA
from CFA_Modify import CFA_Modify
from Evo import knn_2,rd_2,Svm,logistic,nv,Dt,logistic_2,MLP
import random

lst_dt=[]
lst_rf=[]
lst_svm=[]
lst_log=[]
lst_knn=[]
lst_nv=[]
popsize=[90]
import time
now = time.time()
for j in range(len(popsize)):
    print("Popsize "+str(popsize[j]))
    lst = []
    lst_2 = []
    for t in [10,20,30,40,50,60,70]:
        sum = 0
        lst_log=[]
        print("Pop :"+str( popsize[j])+" Itr "+str(t))
        for i in range(30) :
            print("Run " + str(i))
            cfs=CFA(
                    popsize[j],
                    4,
                    7,
                    0,
                    1,
                    0.02,
                    t,
                  -1,
                   1,
                    -5,
                   5
                )
            lst=(cfs.run())
            #result=logistic_2(lst)
            #lst_2.append(result)
            #print(lst)
            #lst_rf.append(rd_2(lst))
            #lst_knn.append(knn_2(lst))
            #lst_svm.append(Svm(lst))
            #lst_nv.append(nv(lst))
            #lst_dt.append(Dt(lst))
            lst_log.append(logistic_2(lst))

        sum=0
        for i in range(len(lst_log)):
            sum=sum+ lst_log[i]
        print(sum/30)
    #print(lst_2)
    #print(max(lst_2))
    #print(str(min(lst_svm)) +" "+str(max(lst_svm)))
#print(logistic([2,1,7,5]))
#later = time.time()
#difference = int(later - now)
#print(difference)

"""
print("DT Max :"+str(max(lst_dt))+"Min :"+str(min(lst_dt)))
print("RF Max :"+str(max(lst_rf))+"Min :"+str(min(lst_rf)))
print("SVM Max :"+str(max(lst_svm))+"Min :"+str(min(lst_svm)))
print("NB Max :"+str(max(lst_nv))+"Min :"+str(min(lst_nv)))
print("KNN Max :"+str(max(lst_knn))+"Min :"+str(min(lst_knn)))
print("log Max :"+str(max(lst_log))+"Min :"+str(min(lst_log)))
print("Time = :" +str(difference))


lst=[0,1,2,3,4,5,6,7]

print("RF " + str(rd_2(lst)))


print("KNN " + str(knn_2(lst)))



print("SVM " + str(Svm(lst)))



print("Nv " + str(nv(lst)))


print("DT " + str(Dt(lst)))


print("logistic " + str(logistic_2(lst)))
"""

