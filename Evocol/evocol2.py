#!/usr/bin/env python3

import numpy as np
from random import random,randint,choice,uniform,shuffle
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


def printM(matrix):
    print(np.array(matrix))

def loswektor(n):
    return [randint(1,n) for _ in range(n)]

def inicjalizacjaPopulacji(pop_size, n,graf):
    P=[]
    kolorowanie_greedy=greedyColoring(graf)
    greedy_osobnik=[kolorowanie_greedy,None,None]
    funkcja_oceny(greedy_osobnik,graf)
    P.append(greedy_osobnik)
    for _ in range(pop_size-1):
        x=loswektor(n)
        tmp=[x,None,None]
        funkcja_oceny(tmp,graf) #[kolorowanie, funkcja_oceny, wektor_bledow]
        P.append(tmp)
    return P

def check2(graf,chromosom):
    kolorowanie = chromosom[0]
    n=len(kolorowanie)
    ilosc_blednych_krawedzi = 0
    indeksy_bledow = [0]*len(graf)
    for i in range(n):
        for j in range(i+1,n):
            if (graf[i][j]==1 and kolorowanie[i]==kolorowanie[j]):
                ilosc_blednych_krawedzi+=1
                indeksy_bledow[i] = 1
                indeksy_bledow[j] = 1
    return ilosc_blednych_krawedzi,indeksy_bledow

def funkcja_oceny(chromosom,graf): #chromosom to trojka: (kolorowanie, funkcja_oceny, wektor_bledow)
    kolorowanie = chromosom[0]
    ilosc_bledow, w = check2(graf,chromosom)
    #suma=sum(w)
    chromosom[1]=len(set(kolorowanie))+ilosc_bledow
    chromosom[2]=w

def selekcja_top5(populacja,graf):
    pop_tmp = populacja.copy()
    najlepsi = []
    #najlepszy = pop_tmp[0]
    for _ in range(round(len(populacja)/20)):
        tmp = min(pop_tmp,key=itemgetter(1))
        pop_tmp.remove(tmp)
        najlepsi.append(tmp)
    dzieci = mating(najlepsi,graf)
    return dzieci


def crossover(parent1,parent2,graf):
    n=len(parent1)
    point=randint(1,n-1)
    kolorowanie1 = parent1[0][:point]+parent2[0][point:]
    kolorowanie2 = parent2[0][:point]+parent1[0][point:]
    child1 = [kolorowanie1,None,None]
    child2 = [kolorowanie1,None,None]
    funkcja_oceny(child1, graf)
    funkcja_oceny(child2, graf)
    return child1,child2

def tournament_crossing(populacja,graf):
    n = len(populacja[0][0])
    best_parent_colors = len(populacja[0][0]) ##Maksymalna ilosc kolorow coś jak int mini = INT_MAX zeby kazda ilosc kolorw byla mniejsza
    best_parent = None
    best_parent_index = 0

    ### LINEAR SEARCH FOR BEST COLORING

    for i in range(len(populacja)):
        if(i[1] < best_parent_colors):
            best_parent = populacja[i]
            index = i

    ### DEFINING 2 RANDOM PARENTS WHO WILL FIGHT FOR MATING DEPENDING OF GRADE RATE
    
    random_parent_1_index = choice(i for i in range(n) if i != best_parent_index)
    random_parent_2_index = choice(i for i in range(n) if (i != best_parent_index and i != random_parent_1_index))
    random_parent_1 = populacja[random_parent_1_index]
    random_parent_2 = populacja[random_parent_2_index]

    ### FIGHT FOR MATING

    best_parent_2 = random_parent_1
    best_parent_2_index = random_parent_1_index ###Linia 163
    if(random_parent_2(1) < random_parent_1(1)):
        best_parent_2 = random_parent_2
        best_parent_2_index = random_parent_2_index ###Linia 163
    
    ### MATING

    return crossover(best_parent,best_parent_2,graf)


def mating(best,graf):
    children=[]
    for i in range(len(best)):
        for j in range(i+1):
            child1,child2=crossover(best[i],best[j],graf)
            children.append(child1)
            children.append(child2)
    return children


def replace(populacja,children):
    p=len(populacja)-len(children) ## P - 2
    j=0
    for i in range(p,len(populacja)):
        populacja[i]=children[j]
        j+=1

def new_replace(populacja,children):
    for c in children:
        ind=populacja.index(max(populacja,key=itemgetter(1)))
        populacja[ind] = c
    return populacja


def mutacja(chromosom,prob):
    if prob > uniform(0.0,1.0):
        kolorowanie = chromosom[0]
        point=randint(0,len(kolorowanie)-1)
        #mut=choice([i for i in range(1,len(chromosom)+1) if i!=chromosom[point]])
        mut=choice([i for i in set(kolorowanie)])
        chromosom[0][point]=mut

def mutacja2(chromosom,prob,graf):
    if prob > uniform(0.0,1.0):
        kolorowanie = chromosom[0]
        wektor_bledow = chromosom[2]
        indeksy=[]
        for i in range (len(wektor_bledow)):
            if(wektor_bledow):
                indeksy.append(i)

        point=choice(indeksy)
        col_zakazane=[]
        for i in range (len(graf)):
            if(graf[i][point] or graf[point][i]):
                col_zakazane.append(i)
        col_zakazane=set(col_zakazane)
        if(len(col_zakazane)<len(set(kolorowanie))):
            mut=choice([i for i in set(kolorowanie) if i not in col_zakazane])
        else:
            mut=choice([i for i in [i for i in range(1,len(graf)+1)] if i not in col_zakazane])

        chromosom[0][point]=mut

def mutuj_populacje2(populacja,prob,graf):
    for i in range(len(populacja)):
        mutacja2(populacja[i],prob,graf)
####
#Czy uzywać funkcji oceny w mutacji                 NA POZNIEJ
##########################################


def mutuj_populacje(populacja,prob):
    for i in range(len(populacja)):
        mutacja(populacja[i],prob)

def rating(Populacja,graf):
    for p in Populacja:
        funkcja_oceny(p,graf)

def znajdz_najlepszego_osobnika(populacja,aktualny_najlepszy):
#maxi=min(lista,key=itemgetter(1))
    for p in populacja:
        if(p[1] < aktualny_najlepszy[1] and sum(p[2]) == 0):
            aktualny_najlepszy = p
    return aktualny_najlepszy



def algorytm_genetyczny(graf,n):
    pop_size=50
    prob_mut=0.7
    print("Inicjalizuje...")
    Populacja= inicjalizacjaPopulacji(pop_size,n,graf) #zmieniona na trojke
    print("Koniec incijalizacji")
    best_kolorowanie=[i for i in range(1,n+1)]
    min_liczba_kolorow=len(set(best_kolorowanie))+1
    best_osobnik=[best_kolorowanie, min_liczba_kolorow, [0]*n]  #zmieniona na trojke
    t=1000
    while(t):
        print("Iteracja: ", t, "Najlepszy kolor: ", len(set(best_osobnik[0])), "Ilosc błędów: ", sum(best_osobnik[2]))
        t-=1
        #Populacja=sorted(Populacja,key=lambda x:funkcja_oceny(x,graf))
        rating(Populacja,graf)
        dzieci = selekcja_top5(Populacja,graf)
        #dzieci = tournament(Populacja,graf)
        Populacja = new_replace(Populacja,dzieci)
        best_osobnik = znajdz_najlepszego_osobnika(Populacja,best_osobnik).copy()
        #mutuj_populacje(Populacja,prob_mut)
        mutuj_populacje2(Populacja,prob_mut,graf)
        

        """
        for i in range(3):
            liczba_kolorow=len(set(Populacja[i]))
            if liczba_kolorow<=min_liczba_kolorow and check(graf,Populacja[i])==0:
                min_liczba_kolorow=liczba_kolorow
                najlepszy_chromosom=Populacja[i]
        
    return najlepszy_chromosom
        """
    return best_osobnik




graf1_edges, num_of_v, num_of_e = load_graph("myciel5.txt")
graf1=edges2matrix(graf1_edges,num_of_v)
printM(graf1)
sol=algorytm_genetyczny(graf1,num_of_v)
printM(sol)
print("Legalne?: ",sum(sol[2]))
print("Ilosc kolorow: ", len(set(sol[0])))
"""
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

