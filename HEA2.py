#!/usr/bin/env python3

import numpy as np
from random import random,randint,choice,uniform,shuffle,randrange
from operator import itemgetter
from collections import deque

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

##################################################################
######                     GNP
##################################################################
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

"""
def check(graf,kolorowanie):
    n=len(kolorowanie)
    bledy=0
    for i in range(n):
        for j in range(i+1,n):
            if (graf[i][j]==1 and kolorowanie[i]==kolorowanie[j]):
                bledy+=1
    return bledy
"""

def printM(matrix):
    print(np.array(matrix))

def loswektor(n):
    return [randint(1,n) for _ in range(n)]


def inicjalizacjaPopulacji(pop_size, n,graf,best_osobnik):
    P=[]
    kolor_greedy=greedyColoring(graf)
    greedy_osobnik=[kolor_greedy,None,None]
    funkcja_oceny(greedy_osobnik,graf,best_osobnik)
    P.append(greedy_osobnik)
    for _ in range(pop_size-1):
        x=loswektor(n)
        tmp=[x,None,None]
        funkcja_oceny(tmp,graf,best_osobnik) #[kolorowanie, funkcja_oceny, wektor_bledow]
        P.append(tmp)
    return P


##################################################################################################################################################


##################################################################
######                   CHECK2
##################################################################

def check2(graf,chromosom):
    kolorowanie = chromosom[0]
    n=len(kolorowanie)
    liczba_blednych_krawedzi = 0
    indeksy_bledow = [0]*len(graf)
    for i in range(n):
        for j in range(i+1,n):
            if (graf[i][j]==1 and kolorowanie[i]==kolorowanie[j]):
                liczba_blednych_krawedzi+=1
                indeksy_bledow[i] = 1
                indeksy_bledow[j] = 1
    return liczba_blednych_krawedzi,indeksy_bledow


##################################################################################################################################################


##################################################################
######                   FUNKCJA OCENY
##################################################################

def funkcja_oceny(chromosom,graf,best_osobnik): 
    kolorowanie = chromosom[0]
    liczba_kolorow=len(set(kolorowanie))
    ilosc_bledow, w = check2(graf,chromosom)
    """
    q=ilosc_bledow*2
    if(ilosc_bledow==0):
        d=0
    else:
        d=1
    k=len(set(kolorowanie))
    chromosom[1]=q + d + k
    chromosom[2]=w
    
    """
    if(ilosc_bledow==0):
        aktualna_min_liczba_kol = len(set(best_osobnik[0]))
        if(liczba_kolorow < aktualna_min_liczba_kol):
            p=0
        else:
            p= 2*(liczba_kolorow - aktualna_min_liczba_kol)
        
        chromosom[1]=liczba_kolorow + p
    else:
        chromosom[1]=ilosc_bledow * liczba_kolorow
    chromosom[2]=w
    
##################################################################################################################################################    


def selekcja_top5(populacja,graf,best_osobnik):
    pop_tmp = populacja.copy()
    najlepsi = []
    #najlepszy = pop_tmp[0]
    for _ in range(10): #round(len(populacja)/40
        tmp = min(pop_tmp,key=itemgetter(1))
        pop_tmp.remove(tmp)
        najlepsi.append(tmp)
    return matingGPX(najlepsi,graf,best_osobnik)


def crossover(parent1,parent2,graf,best_osobnik):
    n=len(parent1)
    point=randint(1,n-1)
    kolorowanie1 = parent1[0][:point]+parent2[0][point:]
    kolorowanie2 = parent2[0][:point]+parent1[0][point:]
    child1 = [kolorowanie1,None,None]
    child2 = [kolorowanie1,None,None]
    funkcja_oceny(child1, graf,best_osobnik)
    funkcja_oceny(child2, graf,best_osobnik)
    return child1,child2

def mating(best,graf):
    children=[]
    for i in range(len(best)):
        for j in range(i+1):
            child1,child2=crossover(best[i],best[j],graf)
            children.append(child1)
            children.append(child2)
    return children


  


##################################################################################################################################################
##################################################################
######             CONFLICT ELIMINATION CROSSOVER
##################################################################


def CEX(parent1,parent2,graf,best_osobnik): #Conflict elimination crossover
    n=len(graf)
    konflikty1=parent1[2]
    konflikty2=parent2[2]
    child1_kolorowanie=[0]*n
    child2_kolorowanie=[0]*n
    
    for i in range(n):
        if(konflikty1[i]):
            child1_kolorowanie[i]=parent2[0][i]
        else:
            child1_kolorowanie[i]=parent1[0][i]
        if(konflikty2[i]):
            child2_kolorowanie[i]=parent1[0][i]
        else:
            child2_kolorowanie[i]=parent2[0][i]
    child1=[child1_kolorowanie,None,None]
    child2=[child2_kolorowanie,None,None]
    funkcja_oceny(child1,graf,best_osobnik)
    funkcja_oceny(child2,graf,best_osobnik)
    return child1, child2


def matingCEX(best,graf,best_osobnik):
    children=[]
    for i in range(len(best)):
        for j in range(i+1):
            child1,child2=CEX(best[i],best[j],graf,best_osobnik)
            children.append(child1)
            children.append(child2)
    return children

##################################################################################################################################################
##################################################################
######           GREEDY PARTITION CROSSOVER
##################################################################



def GPX(parent1,parent2,graf,best_osobnik):
    k=len(set(best_osobnik[0]))
    child_partition=[]
    i=1
    parent1_partition=color2partition(parent1)
    parent2_partition=color2partition(parent2)
    while(i<=k): #or any(part!=[] for part in parent1_partition) or any(part!=[] for part in parent2_partition)
        if(i%2==0):
            if(any(part!=[] for part in parent1_partition)):
                biggest_block= max((part for part in parent1_partition if part!=[]), key=len)
                child_partition.append(biggest_block)
                parent1_partition.remove(biggest_block)

                for v in biggest_block:
                    for part in parent2_partition:
                        for v2 in part:
                            if(v==v2):
                                part.remove(v2)

        else:
            if(any(part!=[] for part in parent2_partition)):
                biggest_block= max((part for part in parent2_partition if part!=[]), key=len)
                child_partition.append(biggest_block)
                parent2_partition.remove(biggest_block)

                for v in biggest_block:
                    for part in parent1_partition:
                        for v2 in part:
                            if(v==v2):
                                part.remove(v2)
    
        i+=1

    if(parent1_partition !=[]):
        for part in parent1_partition:
            for v in part:
                random_block_from_child = choice(child_partition)
                child_partition.remove(random_block_from_child)
                random_block_from_child.append(v)
                child_partition.append(random_block_from_child)
    if(parent2_partition !=[]):
        for part in parent2_partition:
            for v in part:
                random_block_from_child = choice(child_partition)
                child_partition.remove(random_block_from_child)
                random_block_from_child.append(v)
                child_partition.append(random_block_from_child)

    child_kolorowanie=partition2color(child_partition,len(graf))
    child=[child_kolorowanie,None,None]
    funkcja_oceny(child,graf,best_osobnik)
    return child

def matingGPX(rodzice,graf,best_osobnik):
    children=[]
    for i in range(len(rodzice)):
        for j in range(i+1):
            child=GPX(rodzice[i],rodzice[j],graf,best_osobnik)
            children.append(child)
    return children

    



##################################################################################################################################################
##################################################################
######           ROULETTE WHEEL SELECTION
##################################################################
def RoulleteWheelSelection(Populacja,best_osobnik):
    #fbo=best_osobnik[1] #funkcja_best_osobnika
    T=0.0
    for p in Populacja:
        T=T+p[1]
    s=0.0
    r=uniform(0.0,1.0)
    for p in Populacja:
        s=s+ (T-p[1])/T
        if(s>=r):
            return p

def dobor_ruletkowy(Populacja,best_osobnik,graf):
    rodzice=[]
    for _ in range(20):
        rodzic=RoulleteWheelSelection(Populacja,best_osobnik)
        rodzice.append(rodzic)
    return matingCEX(rodzice,graf,best_osobnik)

##################################################################################################################################################







##################################################################################################################################################
##################################################################
######           STOCHASTIC UNIVERSAL SAMPLING
##################################################################

"""
SUS(Population, N)
    F := total fitness of Population
    N := number of offspring to keep
    P := distance between the pointers (F/N)
    Start := random number between 0 and P
    Pointers := [Start + i*P | i in [0..(N-1)]]
    return RWS(Population,Pointers)

RWS(Population, Points)
    Keep = []
    for P in Points
        i := 0
        while fitness sum of Population[0..i] < P
            i++
        add Population[i] to Keep
    return Keep
"""
def SUS(Populacja,best_osobnik):
    T=0.0
    for p in Populacja:
        T=T+p[1]
    
    N=10.0 #liczba osobnikow do wyboru
    distance= 1/N
    r=uniform(0.0,1.0)
    pointers=[]
    pointers.append(r)
    for _ in range(int(N)-1):
        r=r+distance
        pointers.append(r)
    print(pointers)
    s=0.0
    pointer_iterator=0
    rodzice=[]
    for p in Populacja:
        s=s+ (T-p[1])/T
        if(s>=pointers[pointer_iterator]):
            rodzice.append(p)
            pointer_iterator+=1
    return matingCEX(rodzice,graf,best_osobnik)    
##################################################################################################################################################


##################################################################
######                TOURNAMENT SELECTION
##################################################################

def tournament(populacja,graf,best_osobnik):
    liczba_rodzicow=10
    k=5
    rodzice=[]
    POP=populacja.copy()
    while(liczba_rodzicow>0):
        liczba_rodzicow-=1
        rywale=[]
        for _ in range(k):
            rywale.append(choice(POP))
        wybraniec=min(rywale,key=itemgetter(1))
        rodzice.append(wybraniec)
        POP.remove(wybraniec)
    return matingGPX(rodzice,graf,best_osobnik)  
        

##################################################################################################################################################

##################################################################
######             REKOMBINACJA POPULACJI
##################################################################

def replace(populacja,children):
    p=len(populacja)-len(children) ## P - 2
    j=0
    for i in range(p,len(populacja)):
        populacja[i]=children[j]
        j+=1

def new_replace(populacja,children):
    for c in children:
        shuffle(populacja)
        ind=populacja.index(max(populacja,key=itemgetter(1)))
        populacja[ind] = c
    return populacja

def comma_selection(populacja,pop_size,children,best_osobnik,graf):
    old_populacja=populacja.copy()
    new_populacja=[]
    population_size=pop_size
    for c in children:
        new_populacja.append(c)
    population_size-=len(children)
    for i in range(5):
        new_populacja.append(best_osobnik)
        population_size -= 1 
    while(population_size>0):
        wybrancy= tournament(old_populacja,graf,best_osobnik)
        for w in wybrancy:
            new_populacja.append(w)
        population_size -= len(wybrancy)
        shuffle(new_populacja)
    return new_populacja


##################################################################################################################################################


##################################################################
######             MUTACJA W LOSOWYM WIERZCHOLKU
##################################################################
def mutacja(chromosom,prob):
    kolorowanie = chromosom[0]
    if prob > uniform(0.0,1.0):  
        point=randint(0,len(kolorowanie)-1)
        #mut=choice([i for i in range(1,len(chromosom)+1) if i!=chromosom[point]])
        mut=choice([i for i in set(kolorowanie)])
        chromosom[0][point]=mut

def mutuj_populacje(populacja,prob):
    for i in range(len(populacja)):
        mutacja(populacja[i],prob)
##################################################################################################################################################



##################################################################
######       MUTACJA ZAMIANA KOLOROW DWOCH WIERZCHOLKOW
##################################################################
def mutacjaSWAP(chromosom,prob,graf):
    if prob > uniform(0.0,1.0):
        indexes = [chromosom[0].index(x) for x in set(chromosom[0])]
        ind1=choice(indexes)
        indexes.remove(ind1)
        ind2=choice(indexes)

        tmp=chromosom[0][ind1]
        chromosom[0][ind1]=chromosom[0][ind2]
        chromosom[0][ind2]=tmp

def mutuj_populacjeSWAP(populacja,prob,graf):
    for i in range(len(populacja)):
        mutacjaSWAP(populacja[i],prob,graf)

##################################################################################################################################################


##################################################################
######                     MUTACJA FIRST FIT
##################################################################

def color2partition(chromosom):
    kolorowanie=chromosom[0]
    k=len(kolorowanie)
    partition=[ [] for _ in range(k)]
    
    for i in range(0,len(kolorowanie)):
        partition[kolorowanie[i]-1].append(i)   
    return partition

def partition2color(partition,n):
    kolor=[0]*n
    for i in range(len(partition)):
        for v in partition[i]:
            if(partition[i]!=[]):
                kolor[v]=i+1
    return kolor

def czy_maja_wspolna_krawedz(w1,w2,graf):
    if(graf[w1][w2] or graf[w2][w1]):
        return True
    return False

def FirstFitMutation(chromosom, prob, graf):
    if prob > uniform(0.0,1.0):
        osobnik=chromosom.copy()
        partition=color2partition(osobnik)
        block=choice(partition)
        partition.remove(block)
        for v in block:
            for part in partition:
                if(all(czy_maja_wspolna_krawedz(v,w2,graf)==False for w2 in part if part!=[])):
                    part.append(v)
                    block.remove(v)
                    break
        
        if(len(block)!=0):
            partition.append(block)
        mutant=partition2color(partition,len(graf))
        chromosom[0]=mutant

def mutuj_populacjeFF(populacja,prob,graf):
    for p in populacja:
        FirstFitMutation(p,prob,graf)
##################################################################################################################################################
##################################################################
######                     TABUCOL
##################################################################

def checkTABU(graf,kolor):
    kolorowanie=kolor.copy()
    n=len(graf)
    conflict_count = 0
    move_candidates = set()
    for i in range(n):
        for j in range(i+1,n):
            if (graf[i][j]):
                if(kolorowanie[i]==kolorowanie[j]):
                    conflict_count+=1
                    move_candidates.add(i)
                    move_candidates.add(j)
    return conflict_count,list(move_candidates)

def tabucol(k,graph,best_osobnik):

    #############################################
    ####         PARAMETRY TABU SEARCH       ####
    n=len(graph)
    tabu_size=7
    reps=100
    max_iterations=10000
    #############################################
    kolorowanie_best_osobnika=best_osobnik[0].copy()
    


    colors=list(set(kolorowanie_best_osobnika))     #lista mozliwych kolorow do uzycia
    #colors=list(range(k))

    solution=dict()
    for i in range(n):
        for c in colors:
            if c!=kolorowanie_best_osobnika[i]:
                tmp=[]
                tmp=kolorowanie_best_osobnika.copy()
                tmp[i]=c
                if(len(set(tmp))==k):
                    for x in range(n):
                        solution[x]=tmp[x]
                    
                    break
    

    if(not bool(solution)):
        print("Nie ma sasiadow, ktorzy by doprowadzali do dekrementacji liczby kolorow")
        return None



    iterations=0

    tabu=deque()
    aspiration_level=dict()
    """
    for i in range(len(graph)):
        #solution[i]=colors[randrange(0,len(colors))]
        solution[i]=choice(colors)
    """
    
    while iterations < max_iterations:
        conflict_count , move_candidates = checkTABU(graph, solution)
        #conflict_count ,_  = checkTABU(graph, solution)
        move_candidates=[]
        for i in range(len(graph)):
            move_candidates.append(i)
        
        if conflict_count == 0 : #and len(set(list(solution.values())))==k
            #znaleziono prawidlowe 
            break
        
        new_solution = None
        new_conflicts=0
        for _ in range(reps):
            node = move_candidates[randrange(0,len(move_candidates))] #Wybierz wierzcholek do zmienienia
            #node = choice(move_candidates)     #Wybierz wierzcholek do zmienienia  
            new_color = choice([c for c in colors if c!=solution[node]])
            """
            new_color = colors[randrange(0,len(colors)-1)] #inny kolor niz aktualny, if i != solution[node]]
            if solution[node] == new_color:
                new_color=colors[-1]
            """
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
        osobnik=[solution,None,None]
        funkcja_oceny(osobnik,graph,best_osobnik)
        #print(len(set(osobnik[0])))
        return osobnik

def LocalSearch(populacja,graph,best_osobnik,max_nb_of_no_improvments, no_improvments):
    if no_improvments >= max_nb_of_no_improvments:
        no_improvments=0
        POP=populacja.copy()
        wybrancy=[]
        w1 = best_osobnik.copy()
        if w1 in POP:
            POP.remove(w1)
        wybrancy.append(w1)

        #k=len(set(best_osobnik[0]))-1 #kolorowanie o jeden kolor mniejsze niz aktualny najlepszy osobnik
        
        for _ in range(4):
            wn=RoulleteWheelSelection(POP,best_osobnik)
            if wn in POP:
                POP.remove(wn)
            wybrancy.append(wn)
        for w in wybrancy:
            print(w[0])
            #k=len(set(w[0]))
            k=len(Set(best_osobnik[0]))
            k -= 1
            print("Rozpoczynam przeszukiwanie lokalne dla liczby kolorow: ",k)
            znaleziony_osobnik=tabucol(k,graph,w)
            if(znaleziony_osobnik != None):
                ind=populacja.index(max(populacja,key=itemgetter(1)))
                populacja[ind]=znaleziony_osobnik
                print("Koncze przeszukiwanie - Znalazlem osobnika o kolorowaniu: ", len(set(znaleziony_osobnik[0])))
                break
            else:
                print("Koncze przeszukiwanie - Nie znalazlem osobnika o kolorowaniu: ",k)
    return no_improvments


##################################################################################################################################################
def rating(Populacja,graf,best_osobnik):
    shuffle(Populacja)
    for p in Populacja:
        funkcja_oceny(p,graf,best_osobnik)

def znajdz_najlepszego_osobnika(populacja,aktualny_najlepszy,no_improvments):
    aktualna_najmniejsza_liczba_kolorw = len(set(aktualny_najlepszy[0]))
    for p in populacja:
        if(p[1] < aktualny_najlepszy[1] and sum(p[2]) == 0):
            aktualny_najlepszy = p
            no_improvments = 0
            break
    if (aktualna_najmniejsza_liczba_kolorw <= len(set(aktualny_najlepszy[0]))):
        no_improvments += 1
    return aktualny_najlepszy, no_improvments

##################################################################################################################################################
def algorytm_ewolucyjny(graf,n):
    pop_size=200
    prob_mut=0.8

    max_nb_of_no_improvments=50
    no_improvments=0

    best_kolorowanie=[i for i in range(1,n+1)]
    min_liczba_kolorow=len(set(best_kolorowanie))+1
    best_osobnik=[best_kolorowanie, min_liczba_kolorow, [0]*n]

    print("Inicjalizuje...")
    Populacja= inicjalizacjaPopulacji(pop_size,n,graf,best_osobnik) 
    print("Koniec incijalizacji")
  
    t=1000
    while(t):
        print("Iteracja: ", t, "Najlepszy kolor: ", len(set(best_osobnik[0])), "Liczba błędów: ", sum(best_osobnik[2]), "Srednia ocena: ", sum([f for _,f,_ in Populacja])/len(Populacja), "Avg nb",sum([len(set(f)) for f,_,_ in Populacja])/len(Populacja), "Liczba iteracji bez poprawy: ", no_improvments)
        t-=1
        #Populacja=sorted(Populacja,key=lambda x:funkcja_oceny(x,graf,best_osobnik))
        
        rating(Populacja,graf,best_osobnik)
        
        dzieci = selekcja_top5(Populacja,graf,best_osobnik)
        #dzieci=dobor_ruletkowy(Populacja,best_osobnik,graf)
        #dzieci=SUS(Populacja,best_osobnik)
        #dzieci=tournament(Populacja,graf,best_osobnik)
         


        #Populacja = new_replace(Populacja,dzieci)
        Populacja= comma_selection(Populacja,pop_size,dzieci,best_osobnik,graf)
        tmp,no_improvments= znajdz_najlepszego_osobnika(Populacja,best_osobnik,no_improvments)
        best_osobnik=tmp.copy()
        
        #mutuj_populacje(Populacja,prob_mut)
        #mutuj_populacjeSWAP(Populacja,prob_mut,graf)
        mutuj_populacjeFF(Populacja,prob_mut,graf)
        no_improvments=LocalSearch(Populacja,graf,best_osobnik,max_nb_of_no_improvments,no_improvments)
    
    return best_osobnik
##################################################################################################################################################

graf1_edges, num_of_v, num_of_e = load_graph("myciel6.txt")
graf1=edges2matrix(graf1_edges,num_of_v)
printM(graf1)
sol=algorytm_ewolucyjny(graf1,num_of_v)
printM(sol)
print("Legalne?: ",sum(sol[2]))
print("Ilosc kolorow: ", len(set(sol[0])))

##################################################################################################################################################


"""
Instancje:
DSJC500.1 - BEST = 12, OPT nie znane 
DSJR500.1 - OPT = 86 kolory

miles750 - OPT=31 kolory
miles1000 - OPT=42 kolory, znalazl 43
miles1500 - OPT=73 kolory, znalazl optymalne kolorowanie po 150 iteracjach

queen7_7 - OPT= 7 kolorow - Lokalne optimum dla 11 kolorow
queen11_11 - OPT=11 kolorow 

myciel3 - OPT=4 kolory - zaliczone
myciel4 - OPT=5 kolorow - zaliczone 
myciel5 - OPT=6 kolorow - zaliczone 
myciel6 - OPT=7 kolorow - zaliczone (POPSIZE= 200, PROB_MUT=0.8, T=700, Selekcja_top5, Mutacja First Fit, CEX Crossover)
myciel7 - OPT=8 kolorow - znaleziono 9 dla (POPSIZE= 200, PROB_MUT=0.8, T=1000,Selekcja_top5/Tournament selection, Mutacja First Fit, GPX Crossover)
"""




