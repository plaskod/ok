#!/usr/bin/env python3

import numpy as np
from random import random,randint,choice

#"Model G(n,p)"
def gnp(n,p):
    a=[[0 for i in range(n)] for j in range(n)]
    for i in range(1,n):
        for j in range(i):
            if(random()<=p):
                a[i][j]=a[j][i]=1
    return a

def check(graf,kolorowanie):
    n=len(kolorowanie)
    bledy=0
    for i in range(n):
        for j in range(i+1,n):
            if (graf[i][j]==1 and kolorowanie[i]==kolorowanie[j]):
                bledy+=1
    return bledy

def greedyColoring(graf):
    n=len(graf)
    kolorowanie=[None for _ in range(n)]
    kolorowanie[0]=1
    for i in range(1,n):
        for k in range(1,n+1):
            dobry=True
            for j in range(0,i+1):
                if(graf[i][j] and kolorowanie[j]==k):
                    dobry=False
                    break
            if(dobry):
                kolorowanie[i]=k
                break
    return kolorowanie

def printM(matrix):
    print(np.array(matrix))

def loswektor(n):
    return [randint(1,n) for _ in range(n)]

def inicjalizacjaPopulacji(pop_size, n):
    P=[]
    for _ in range(pop_size):
        P.append(loswektor(n))
    return P

def funkcja_oceny(chromosom):
    return len(set(chromosom))

def selekcja_top5(populacja):
    return populacja[:5]

def crossover(parent1,parent2):
    n=len(parent1)
    point=randint(1,n-1)
    child1=parent1[:point]+parent2[point:]
    child2=parent2[:point]+parent1[point:]
    return child1,child2

def mating(best):
    children=[]
    for i in range(len(best)):
        for j in range(i+1):
            child1,child2=crossover(best[i],best[j])
            children.append(child1)
            children.append(child2)
    return children

def replace(populacja,children):
    p=len(populacja)-len(children)
    j=0
    for i in range(p,len(populacja)):
        populacja[i]=children[j]
        j+=1

def mutacja(chromosom,prob):
    if prob > random.uniform(0.0,1.0):
        point=randint(0,len(chromosom)-1)
        mut=choice([i for i in range(1,len(chromosom)+1) if i!=chromosom[point]])
        chromosom[point]=mut

def mutuj_populacje(populacja,prob):
    for i in range(len(populacja)):
        mutacja(populacja[i],prob)

def algorytm_genetyczny(graf,n):
    pop_size=4
    Populacja=inicjalizacjaPopulacji(pop_size,n)
    printM(Populacja)
    t=5
    while(t):
        t-=1
        Populacja=sorted(Populacja,key=lambda x: funkcja_oceny(x))
        printM(Populacja)
        children=mating(selekcja_top5(Populacja))
        replace(populacja,children)

n=7
p=0.5
graf=gnp(n,p)
#algorytm_genetyczny(graf,n)
a=[[1,2,3,4],
  [5,6,7,8],
  [9,10,11,12],
  ["a","b","c","d"],
  ["e","f","g","h"],
  ["x","x","x","x"]]
for i in range(25):
    chromosom=[8,8,8,8,8]
    chromosom[point]=mut
    printM(chromosom)
