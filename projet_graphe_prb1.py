# --------------------------------------------------
# Projet Optimisation et Graphes
# JAMIN Antoine, PAGNON Alexis, SANCHEZ Adam, DUMOULIN Simon
# 31/05/2025
# --------------------------------------------------

import docplex
import docplex.mp
import docplex.mp.model
import numpy as np

"""
Problème 1 : Minimisation du retard pondéré avec des heures d'atterrissage cibles 

Contraintes :

Ei<=Xi<=Li                                 (2)
Xi-Ti=αi-βi                                (3)
Xj-Xi>=Sij*δij-M*Yji                       (4)
Yij+Yji=1                                  (5)
δij>=γir+γjr-1                             (6)
Somme(r=1 à m) de γir = 1                  (7)
Yij, δij, γir ∈ {0,1}                      (8)
Xi, αi, βi >= 0                            (9)

Objectif: min z = Somme(i=1 à n) de (αi*ci^- + βi*ci^+)  (1)
"""

# Étape 0: Ouvrir le fichier avec nos variables
airland_file = open("./airlands/airland1.txt")
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
E = []  # Tableau des Ei (earliest landing time)
T = []  # Tableau des Ti (target landing time)
L = []  # Tableau des Li (latest landing time)
s = []  # Tableau des s_i, étant eux-même des tableaux de s_ij
c_minus = []  # Coût de pénalité par unité de temps pour atterrissage avant l'heure cible
c_plus = []  # Coût de pénalité par unité de temps pour atterrissage après l'heure cible

line = airland_file.readline().strip()
n = int(line.split(" ")[0])
m = 2  # Nombre de pistes d'atterrissage (à ajuster selon le problème)

# Lire le fichier ligne par ligne
for i in range(0, n):
    line = airland_file.readline().strip()

    # Ajouter les valeurs dans les listes appropriées
    values = line.split(" ")

    E.append(float(values[1]))
    T.append(float(values[2]))
    L.append(float(values[3]))

    c_minus.append(float(values[4]))  # Coût de pénalité pour atterrissage prématuré
    c_plus.append(float(values[5]))  # Coût de pénalité pour atterrissage tardif

    # Récupère les S_i
    s_i = []
    # Puisque les valeurs de S sont réparties sur plusieurs lignes, on utilise un while sur la longueur de
    # notre tableau pour les récupérer sans lire une ligne en trop
    while (len(s_i) < n):
        line = airland_file.readline().strip()  # On lit une ligne
        values = line.split(" ")
        for v in values:  # On ajoute toutes les valeurs d'une ligne à s_i, et si la taille de s_i n'est pas égal à n, on lit la ligne suivante
            s_i.append(float(v))
    s.append(s_i)  # On ajoute les valeurs de s_i de cet avion à s qui comprend les valeurs s_i de tous les avions

airland_file.close()

print(f"Nombre d'avions: {n}")
print(f"Heures d'atterrissage au plus tôt: {E}")
print(f"Heures d'atterrissage cibles: {T}")
print(f"Heures d'atterrissage au plus tard: {L}")
print(f"Coûts de pénalité pour atterrissage avant la cible: {c_minus}")
print(f"Coûts de pénalité pour atterrissage après la cible: {c_plus}")
print(f"Exemple de temps de séparation (premier avion): {s[0]}")

# Étape 1: Création du modèle
model = docplex.mp.model.Model("Probleme1")

# Calculer une grande valeur M pour les contraintes (pour transformer les OU en ET)
M = max(L) + max([max(s_i) for s_i in s]) * n

# Étape 2: Ajout des variables de décision
# x_i: heure d'atterrissage de l'avion i
x = model.continuous_var_dict(range(n), lb=lambda i: E[i], ub=lambda i: L[i], name=lambda i: f"x_{i}")

# alpha_i, beta_i: écarts par rapport à l'heure cible
alpha = model.continuous_var_dict(range(n), lb=0, name=lambda i: f"alpha_{i}")
beta = model.continuous_var_dict(range(n), lb=0, name=lambda i: f"beta_{i}")

# y_ij: variable binaire pour l'ordre d'atterrissage
y = model.binary_var_dict([(i, j) for i in range(n) for j in range(n) if i != j], name=lambda ij: f"y_{ij[0]}_{ij[1]}")

# delta_ij: variable binaire pour l'affectation des pistes
delta = model.binary_var_dict([(i, j) for i in range(n) for j in range(n) if i != j], name=lambda ij: f"delta_{ij[0]}_{ij[1]}")

# gamma_ir: variable binaire pour l'affectation de l'avion i à la piste r
gamma = model.binary_var_dict([(i, r) for i in range(n) for r in range(m)], name=lambda ir: f"gamma_{ir[0]}_{ir[1]}")

# Définir la fonction objectif: minimiser le makespan
model.minimize(sum((alpha[i]*c_minus[i] + beta[i]*c_plus[i]) for i in range(n)))

# Étape 3: Ajout des contraintes
for i in range(n):
    # Contrainte (3): Xi - Ti = αi - βi
    model.add_constraint(x[i] - T[i] == alpha[i] - beta[i])

    # Contrainte (7): (Somme(r=1 à m) de γir) = 1
    model.add_constraint(sum(gamma[(i,r)] for r in range(m)) == 1)

    for j in range(n):
        # Contrainte (4): Xj - Xi >= Sij * δij - M * Yji
        if i != j:
            model.add_constraint(x[j] - x[i] >= delta[(i,j)] * s[i][j] - y[(j,i)] * M )

        # Contrainte (5): Yij + Yji = 1
        if i < j:  # Pour éviter la redondance, on ne considère que les paires (i,j) où i < j
            model.add_constraint(y[(i,j)] + y[(j,i)] == 1)
        
        # Contrainte (6): δij >= γir + γjr - 1
        if i != j:
            for r in range(m):
                model.add_constraint(delta[(i,j)] - gamma[(i,r)] - gamma[(j,r)] >= -1)
        
# Étape 4: Résolution et affichage de la solution
try:
    solution = model.solve()

    if(solution):
        print("\nSolution trouvée :")
        # model.print_solution()
        print(f"Valeur optimale de l'objectif: {np.round(model.objective_value)}")

        # Afficher les heures d'atterrissage
        print("\nHeures d'atterrissage optimales :")
        landing_times = []
        for i in range(n):
            x_val = x[i].solution_value
            landing_times.append((i, x_val))
            alpha_val = alpha[i].solution_value
            beta_val = beta[i].solution_value

            # Trouver la piste assignée
            assigned_runway = -1
            for r in range(m):
                gamma_var = model.get_var_by_name(f"gamma_{i}_{r}")
                if abs(gamma_var.solution_value - 1) < 1e-6:  # Presque égal à 1
                    assigned_runway = r
                    break

            status = "a l'heure"
            if alpha_val > 1e-6:
                status = f"en avance de {alpha_val:.2f}"
            elif beta_val > 1e-6:
                status = f"en retard de {beta_val:.2f}"

            print(f"Avion {i}: atterrissage à {x_val:.2f} ({status}) sur la piste {assigned_runway}")

        # Afficher l'ordre d'atterrissage par piste
        print("\nOrdre d'atterrissage par piste :")
        landing_order = {}

        for i in range(n):
            for r in range(m):
                gamma_var = model.get_var_by_name(f"gamma_{i}_{r}")
                if abs(gamma_var.solution_value - 1) < 1e-6:
                    x_val = x[i].solution_value
                    if r not in landing_order:
                        landing_order[r] = []
                    landing_order[r].append((i, x_val))

        for r in landing_order:
            # Trier les avions par heure d'atterrissage
            landing_order[r].sort(key=lambda x: x[1])
            print(f"Piste {r}: {[i for i, _ in landing_order[r]]}")

    else:
        print("Aucune solution trouvée")

except Exception as e:
    print(f"Erreur lors de la résolution: {e}")