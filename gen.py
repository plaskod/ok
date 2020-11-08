#!/usr/bin/env python3

import numpy as np
from random import random,randint,choice,uniform

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
    if prob > uniform(0.0,1.0):
        point=randint(0,len(chromosom)-1)
        mut=choice([i for i in range(1,len(chromosom)+1) if i!=chromosom[point]])
        chromosom[point]=mut

def mutuj_populacje(populacja,prob):
    for i in range(len(populacja)):
        mutacja(populacja[i],prob)

def algorytm_genetyczny(graf,n):
    pop_size=100
    prob_mut=0.05
    Populacja=inicjalizacjaPopulacji(pop_size,n)
    najlepszy_chromosom=[i for i in range(1,n+1)]
    min_liczba_kolorow=len(set(najlepszy_chromosom))
    t=1000
    while(t):
        t-=1
        Populacja=sorted(Populacja,key=lambda x:funkcja_oceny(x))
        children=mating(selekcja_top5(Populacja))
        replace(Populacja,children)
        mutuj_populacje(Populacja,prob_mut)
        
        for i in range(5):
            liczba_kolorow=len(set(Populacja[i]))
            if liczba_kolorow<=min_liczba_kolorow and check(graf,Populacja[i])==0:
                min_liczba_kolorow=liczba_kolorow
                najlepszy_chromosom=Populacja[i]
    return najlepszy_chromosom
n=7
p=0.5
graf=gnp(n,p)
nc=algorytm_genetyczny(graf,n)
print("SOL: ")
printM(nc)

