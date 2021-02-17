import numpy as np
from random import random,randint,choice,uniform,shuffle,randrange
from operator import itemgetter
from collections import deque

def gnp(n,p):
    a=[[0 for i in range(n)] for j in range(n)]
    for i in range(1,n):
        for j in range(i):
            if(random()<=p):
                a[i][j]=a[j][i]=1
    return a

def loswektor(n):
    return [randint(1,n) for _ in range(n)]

def check(graf,kolorowanie):
    n=len(kolorowanie)
    bledy=0
    for i in range(n):
        for j in range(i+1,n):
            if (graf[i][j]==1 and kolorowanie[i]==kolorowanie[j]):
                bledy+=1
    return bledy

def checkTABU(graf,kolorowanie):
    n=len(graf)
    conflict_count = 0
    move_candidates = set()
    for i in range(n):
        for j in range(i+1,n):
            if (graf[i][j]>0 and kolorowanie[i]==kolorowanie[j]):
                conflict_count+=1
                move_candidates.add(i)
                move_candidates.add(j)
    return conflict_count,list(move_candidates)

def tabucol(osobnik,k,graph):

    #############################################
    ####         PARAMETRY TABU SEARCH       ####
    tabu_size=7
    reps=100
    max_iterations=10000
    #############################################

    #solution=osobnik[0].copy()
    

    
    #lista mozliwych kolorow do uzycia
    colors=list(range(k))
    iterations=0

    tabu=deque()
    aspiration_level=dict()
    solution=dict()

    for i in range(len(graph)):
        solution[i]=colors[randrange(0,len(colors))]
    
    while iterations < max_iterations:
        conflict_count , move_candidates = checkTABU(graph, solution)
        
        if conflict_count == 0: 
            #znaleziono prawidlowe 
            break
        
        new_solution = None
        new_conflicts=0
        for _ in range(reps):
            node = move_candidates[randrange(0,len(move_candidates))] #Wybierz wierzcholek do zmienienia
            new_color = colors[randrange(0,len(colors)-1)] #inny kolor niz aktualny, if i != solution[node]]
            if solution[node] == new_color:
                new_color=colors[-1]

            new_solution = solution.copy()
            new_solution[node] = new_color
            #new_conflicts, _ = checkTABU(graph, new_solution)
            for i in range(len(graph)):
                for j in range(i+1,len(graph)):
                    if graph[i][j] and new_solution[i]==new_solution[j]:
                        new_conflicts += 1
            if new_conflicts < conflict_count: #znaleziono udoskonalenie
                if new_conflicts <= aspiration_level.setdefault(conflict_count, conflict_count - 1):
                    aspiration_level[conflict_count]= new_conflicts - 1
                    if (node,new_color) in tabu:
                        tabu.remove((node, new_color))
                        break

                else:
                    if (node,new_color) in tabu:
                        #zly ruch
                        continue
                break

            tabu.append((node,solution[node]))
            if len(tabu) > tabu_size: #przepelniona kolejka
                tabu.popleft()#usun najstarszy ruch

            solution = new_solution
            iterations += 1
    
    if conflict_count != 0:
        print("Nie znaleziono")
        return None
    else:
        solution=list(solution.values())
        [c_num+1 for c_num in solution]
        print("Znaleziono kolorowanie: ", len(set(solution)))
        return solution

###########################################################################################

def check_conflicts(matrix, colouring):
    conflict_neighbour = set()
    conflicts = 0
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            if matrix[i][j] > 0:
                if colouring[i] == colouring[j]:
                    conflict_neighbour.add(i)
                    conflict_neighbour.add(j)
                    conflicts += 1
    conflict_neighbour = list(conflict_neighbour)
    return conflict_neighbour, conflicts
 
 
def tabu_search_colouring(matrix, number_of_colors):
    colors = list(range(number_of_colors))
    count_iteration = 0
    tabu = deque()
    colouring = aspiration = dict()
 
    for i in range(len(matrix)):
        colouring[i] = colors[randrange(0, len(colors))]
 
    while count_iteration < 10000:
        conflict_neighbour, conflicts = check_conflicts(matrix, colouring)
        
 
        if conflicts == 0:
            break
 
        new_colouring = None
        new_conflicts = 0
 
        for _ in range(100):
            node = conflict_neighbour[randrange(0, len(conflict_neighbour))]
            color = colors[randrange(0, len(colors) - 1)]
            if colouring[node] == color:
                color = colors[-1]
            new_colouring = colouring.copy()
            new_colouring[node] = color
            for i in range(len(matrix)):
                for j in range(i + 1, len(matrix)):
                    if matrix[i][j] > 0 and new_colouring[i] == new_colouring[j]:
                        new_conflicts += 1
            if new_conflicts < conflicts:
                if new_conflicts <= aspiration.setdefault(conflicts, conflicts - 1):
                    aspiration[conflicts] = new_conflicts - 1
                    if (node, color) in tabu:
                        tabu.remove((node, color))
                        break
                else:
                    if (node, color) in tabu:
                        continue
                break
        tabu.append((node, colouring[node]))
        if len(tabu) > 7:
            tabu.popleft()
        colouring = new_colouring
        count_iteration += 1
    if conflicts != 0:
        print("Nie można pokolorować {} kolorami".format(number_of_colors))
        return None
    else:
        print("Znaleziono pokolorowanie {} kolorami\n".format(number_of_colors))
        print(list(colouring.values()),len(set(list(colouring.values()))))
 

"""
graph =  gnp(50,0.4)
y=loswektor(50)

print("Init wektor 1: ",y,len(set(y)))
k=len(set(y))-1
tabu_search_colouring(graph,k)
z=loswektor(50)
print("Init wektor 2: ",z,len(set(z)))
l=len(set(z))-1
sol=tabucol([z,None,None],l,graph)
print(sol,len(set(sol)),check(graph,sol))
"""

"""
w=[]
c=   w[randrange(0, len(w))]
print(c)
"""
w1=[]
c1= choice([ i for i in w1 if w1 != []])
print(c1)