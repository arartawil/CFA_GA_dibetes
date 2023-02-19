# genetic algorithm search of the one max optimization problem
from numpy.random import randint
from numpy.random import rand
from Correlation import corr
from Evo import knn_2,rd_2,Svm,logistic_2,nv,Dt
# objective function
def onemax(x):
    return -sum(x)




# tournament selection
def selection(pop, scores, k=5):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:

        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1) - 2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


# mutation operator
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # check for a mutation
        if rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]


# genetic algorithm
def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut):
    # initial population of random bitstring
    #pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]

    pop = [[0]*n_bits]*n_pop

    for i in range(n_pop):
        for j in range(4):
            index = randint(0,8)
            pop[i][j]=1
    # keep track of best solution
    best, best_eval = 0, objective(pop[0])
    # enumerate generations
    for gen in range(n_iter):

        # evaluate all candidates in the population
        scores = [objective(c) for c in pop]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %.3f" % (gen, pop[i], scores[i]))
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation

        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i + 1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
    return [best, best_eval]

3.88

lst_dt=[]
lst_rf=[]
lst_svm=[]
lst_log=[]
lst_knn=[]
lst_nv=[]
import time
now = time.time()

sum=0
for i in range(30):
    print("Run "+str(i))
    # define the total iterations
    n_iter = 50
    # bits
    n_bits = 8
    # define the population size
    n_pop = 100
    # crossover rate
    r_cross = 0.9
    # mutation rate
    r_mut =0.01
    # perform the genetic algorithm search
    best, score = genetic_algorithm(corr, n_bits, n_iter, n_pop, r_cross, r_mut)
    #print('Done!')
    #print('f(%s) = %f' % (best, score))

    lst_index = []
    if(best !=0):
        for j in range(len(best)):
            if best[j] == 1:
                lst_index.append(j)

        #lst_rf.append(rd_2(lst_index))
        #lst_knn.append(knn_2(lst_index))
        #lst_svm.append(Svm(lst_index))
        #lst_nv.append(nv(lst_index))
        #lst_dt.append(Dt(lst_index))
        lst_log.append(logistic_2(lst_index))
        #sum+=Svm(lst_index)
        #print(Svm(lst_index))

later = time.time()
difference = int(later - now)

sum=0
for i in range(len(lst_log)):
    sum=sum+ lst_log[i]
    print(sum/30)


"""print("DT Max :"+str(max(lst_dt))+"Min :"+str(min(lst_dt)))
print("RF Max :"+str(max(lst_rf))+"Min :"+str(min(lst_rf)))
print("SVM Max :"+str(max(lst_svm))+"Min :"+str(min(lst_svm)))
print("NB Max :"+str(max(lst_nv))+"Min :"+str(min(lst_nv)))
print("KNN Max :"+str(max(lst_knn))+"Min :"+str(min(lst_knn)))
print("log Max :"+str(max(lst_log))+"Min :"+str(min(lst_log)))
print("Time = :" +str(difference))

print("RF " + str(rd_2([5,1,6,7])))
print("KNN " + str(knn_2([5,1,6,7])))
print("SVM " + str(Svm([5,1,6,7])))
print("Nv " + str(nv([5,1,6,7])))
print("DT " + str(Dt([5,1,6,7])))
print("Log " + str(logistic_2([5,1,6,7])))
"""
