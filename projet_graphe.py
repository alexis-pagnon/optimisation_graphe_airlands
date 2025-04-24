# --------------------------------------------------
# Projet Optimisation et Graphes
# JAMIN Antoine, PAGNON Alexis, SANCHEZ Adam
# 31/05/2025
# --------------------------------------------------

import cplex
"""
Contraintes :

Ei<=Xi
Xi<=Li
Xj-Xi>=Sij*δij-MYji
Yij+Yji=1
δij>=γir+γjr-1
Somme(r=1 à m) de γir =1

0<=δij<=1
0<=γir<=1
0<=Yij<=1

Li>=0
Lj>=0
Xi>=0

Problème 1:

Problème 2:

Problème 3:
"""
# Étape 0: Ouvrir le fichier avec nos variables
airland_file = open("airland1.txt")
"""
The format of these data files is:
number of planes (p), freeze time
for each plane i (i=1,...,p):
   appearance time, earliest landing time, target landing time,
   latest landing time, penalty cost per unit of time for landing
   before target, penalty cost per unit of time for landing
   after target
   for each plane j (j=1,...p): separation time required after i lands before j can land
"""
E = [] # Tableau des Ei
T = [] # Tableau des Ti
L = [] # Tableau des Li
s = [] # Tableau des s_i, étant eux-même des tableaux de s_ij

alpha = []
beta = []

line = airland_file.readline().strip()
n = int(line.split(" ")[0])

# Lire le fichier ligne par ligne
for i in range(0,n):
    line = airland_file.readline().strip()

    # Ajouter les valeurs dans les listes appropriées
    values = line.split(" ")

    E.append(float(values[1]))
    T.append(float(values[2]))
    L.append(float(values[3]))

    alpha.append(float(values[4]))
    beta.append(float(values[5]))

    # Récupère les S_i
    s_i = []
    #Puisque les valeurs de S sont réparties sur plusieurs lignes, on utilise un while sur la longueur de  
    #notre tableau pour les récupérer sans lire une ligne en trop
    while(len(s_i)<n):
        line = airland_file.readline().strip() #On lit une ligne
        values = line.split(" ")
        for v in values: #On ajoute toutes les valeurs d'une ligne à s_i, et si la taille de s_i n'est pas égal à n, on lit la ligne suivante
            s_i.append(int(v))
    s.append(s_i) #On ajoute les valeurs de s_i de cet avion à s qui comprend les valeurs s_i de tous les avions

print(n)
print(E)
print(T)
print(L)
print(s)

# Étape 1: Création du modèle
model = cplex.Cplex()
model.objective.set_sense(model.objective.sense.minimize)

# Étape 2: Ajout des variables de  décision
# Nom des variables, coefficients dans la fonction objectif et borne inf
names = ["Index", "E", "X", "Y", "L", "S", "M", "Delta", "Gamma"]
objective = []
lower_bounds = []
model.variables.add(obj=objective, lb=lower_bounds, names=names)

# Étape 3: Ajout des contraintes
# On ajoute les contraintes 
for i in range(0, n):

    model.add_constraint(E[i] >= L[i].earliest_landing)
    

# Étape 4: Résolution et affichage de la solution
model.solve()

"""
print("Valeur optimale de l'objectif:", model.solution.get_objective_value())
solution_values = model.solution.get_values()
for var_name, value in zip(names, solution_values):
    print(f"Valeur optimale de {var_name} : {value}")
"""