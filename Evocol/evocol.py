#!/usr/bin/env python3

import numpy as np
from random import random,randint,choice,uniform
from operator import itemgetter

################################################################################################################################
###########################
######### PLIKI ###########
###########################
def load_graph(path):
    """Load graph from file."""
    edges=[]
    with open(path, mode='r') as f:
        for line in f:
            if line.split()[0] == 'e':
                _, w1, w2= line.split()
                if w1 != w2:
                    edges.append((int(w1), int(w2)))

            elif line.split()[0] == 'p':
                _,_, num_of_vertices, num_of_edges=line.split()

    return edges, int(num_of_vertices), int(num_of_edges)

def edges2matrix(edges,n):
    a=[[0 for i in range(n)] for j in range(n)]
    printM(a)

    for edge in edges:
        #print(edge[0],edge[1])
        a[edge[0]-1][edge[1]-1]=1
        #a[edge[1]-1][edge[0]-1]=1

    return a

################################################################################################################################

#"Model G(n,p)"
def gnp(n,p):
    a=[[0 for i in range(n)] for j in range(n)]
    for i in range(1,n):
        for j in range(i):
            if(random()<=p):
                a[i][j]=a[j][i]=1
    return a

def greedyColoring(graf):
    n=len(graf)
    colors=[None]*n
    colors[0]=1
    for i in range(1,n):
        for k in range(1,n+1):
            dobry=True
            for j in range(0,i):
                if graf[i][j] and colors[j]==k:
                    dobry=False
                    break
            if dobry:
                colors[i]=k
    return colors

def check(graf,kolorowanie):
    n=len(kolorowanie)
    bledy=0
    for i in range(n):
        for j in range(i+1,n):
            if (graf[i][j]==1 and kolorowanie[i]==kolorowanie[j]):
                bledy+=1
    return bledy

def check2(graf,kolorowanie):
    n=len(kolorowanie)
    bledy = 0
    indeksy_bledow = [0]*len(kolorowanie)
    for i in range(n):
        for j in range(i+1,n):
            if (graf[i][j]==1 and kolorowanie[i]==kolorowanie[j]):
                bledy+=1
                indeksy_bledow[i] = 1
                indeksy_bledow[j] = 1
    return bledy,indeksy_bledow

def printM(matrix):
    print(np.array(matrix))

def loswektor(n):
    return [randint(1,n) for _ in range(n)]

def inicjalizacjaPopulacji(pop_size, n):
    P=[]
    for _ in range(pop_size):
        x=loswektor(n)
        tmp=(x,None,None)
        ilosc_bledow,w=funkcja_oceny(graf,tmp) #(kolorowanie, funkcja_oceny, wektor_bledow)
        P.append(tmp)
    return P


def funkcja_oceny(chromosom,graf): #chromosom to trojka: (kolorowanie, funkcja_oceny, wektor_bledow)
    kolorowanie= chromosom[0]
    ilosc_bledow, w = check2(graf,kolorowanie)
    chromosom[1]=len(set(kolorowanie))+len(graf)*ilosc_bledow
    chromosom[2]=w
    

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
        #mut=choice([i for i in range(1,len(chromosom)+1) if i!=chromosom[point]])
        mut=choice([i for i in set(chromosom)])
        chromosom[point]=mut

def mutacja2(chromosom,prob):
    if prob > uniform(0.0,1.0):

        point=randint(0,len(chromosom)-1)
        #mut=choice([i for i in range(1,len(chromosom)+1) if i!=chromosom[point]])
        mut=choice([i for i in set(chromosom)])
        chromosom[point]=mut

def mutuj_populacje(populacja,prob):
    for i in range(len(populacja)):
        mutacja(populacja[i],prob)

def rating(Populacja):
    for p in Populacja:
        funkcja_oceny(p)


def algorytm_genetyczny(graf,n):
    pop_size=40
    prob_mut=0.7
    print("Inicjalizuje...")
    Populacja= inicjalizacjaPopulacji(pop_size,n) #zmieniona na trojke
    print("Koniec incijalizacji")
    best_kolorowanie=[i for i in range(1,n+1)]
    min_liczba_kolorow=len(set(najlepszy_chromosom))+1
    best_osobnik=(best_kolorowanie, min_liczba_kolorow, [0]*n)  #zmieniona na trojke
    t=700
    while(t):
        print("Iteracja: ", t, "Najlepszy kolor: ", len(set(Populacja[0])))
        t-=1
        #Populacja=sorted(Populacja,key=lambda x:funkcja_oceny(x,graf))
        rating(Populacja)
        children=mating(selekcja_top5(Populacja))
        replace(Populacja,children)
        mutuj_populacje(Populacja,prob_mut)
        
        for i in range(3):
            liczba_kolorow=len(set(Populacja[i]))
            if liczba_kolorow<=min_liczba_kolorow and check(graf,Populacja[i])==0:
                min_liczba_kolorow=liczba_kolorow
                najlepszy_chromosom=Populacja[i]
    return najlepszy_chromosom



P=[["foo",9], ["bar",2], ["baz",1]]
ind=P.index(min(P,key=itemgetter(1)))
print(ind)
"""
graf1_edges,num_of_v,num_of_e=load_graph("DSJC500.1.txt")
graf1=edges2matrix(graf1_edges,num_of_v)

sol=algorytm_genetyczny(graf1,num_of_v)
printM(sol)
print("Legalne?: ",check(graf1,sol))
print("Ilosc kolorow: ", len(set(sol)))
"""
#########################################################################
"""
Instancje:
DSJC500.1 - BEST = 12, OPT nie znane 
DSJR500.1 - OPT = 86 kolory

miles750 - OPT=31 kolory - znajduje prawidlowe kolorowanie = 35 dla parametrow: pop_Size= 400, prob_mut = 0.7, T=700
miles1000 - OPT=42 kolory - not even close
miles1500 - OPT=73 kolory -  nie zwraca prawidlowego

queen7_7 - OPT= 7 kolorow - znajduje prawidlowe kolorowanie = 11 dla parametrow: pop_Size= 400, prob_mut = 0.7, T=700
queen11_11 - OPT=11 kolorow - WTF?

myciel3 - OPT=4 kolory - zaliczone
myciel4 - OPT=5 kolorow - zaliczone 
myciel5 - OPT=6 kolorow - znajduje prawidlowe kolorowanie = 6/7/8 dla parametrow: pop_Size= 400, prob_mut = 0.7, T= 700
myciel6 - OPT=7 kolorow - znajduje prawidlowe kolorowanie = 15 dla parametrow: pop_Size= 400, prob_mut = 0.7, T=700
"""






"""
n=50
p=0.63
graf=gnp(n,p)
nc=algorytm_genetyczny(graf,n)
printM(graf)
greedy=greedyColoring(graf)
print("Legalny greedy: ", check(graf,greedy))
print("Greedy: ",len(set(greedy)))
printM(greedy)
print("legalny SOL: ", check(graf,nc))
printM(nc)
print("Ilosc kolorow: ", len(set(nc)))
"""