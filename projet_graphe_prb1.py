# --------------------------------------------------
# Projet Optimisation et Graphes
# JAMIN Antoine, PAGNON Alexis, SANCHEZ Adam, DUMOULIN Simon
# 31/05/2025
# --------------------------------------------------

import cplex
import numpy as np

"""
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
model = cplex.Cplex()
model.objective.set_sense(model.objective.sense.minimize)

# Calculer une grande valeur M pour les contraintes (pour transformer les OU en ET)
M = max(L) + max([max(s_i) for s_i in s]) * n

# Étape 2: Ajout des variables de décision
# x_i: heure d'atterrissage de l'avion i
x_names = [f"x_{i}" for i in range(n)]
model.variables.add(names=x_names, lb=[E[i] for i in range(n)], ub=[L[i] for i in range(n)]) # E_i <= x_i <= L_i

# alpha_i, beta_i: écarts par rapport à l'heure cible
alpha_names = [f"alpha_{i}" for i in range(n)]
beta_names = [f"beta_{i}" for i in range(n)]
model.variables.add(names=alpha_names, lb=[0] * n)  # alpha_i >= 0
model.variables.add(names=beta_names, lb=[0] * n)   # beta_i >= 0

# y_ij: variable binaire pour l'ordre d'atterrissage
y_names = [f"y_{i}_{j}" for i in range(n) for j in range(n) if i != j]
model.variables.add(names=y_names, lb=[0] * len(y_names), ub=[1] * len(y_names), types=["B"] * len(y_names))    # 0 <= y_i_j <= 1 variable binaire

# delta_ij: variable binaire pour l'affectation des pistes
delta_names = [f"delta_{i}_{j}" for i in range(n) for j in range(n) if i != j]
model.variables.add(names=delta_names, lb=[0] * len(delta_names), ub=[1] * len(delta_names), # 0 <= delta_i_j <= 1 variable binaire
                    types=["B"] * len(delta_names))

# gamma_ir: variable binaire pour l'affectation de l'avion i à la piste r
gamma_names = [f"gamma_{i}_{r}" for i in range(n) for r in range(m)]
model.variables.add(names=gamma_names, lb=[0] * len(gamma_names), ub=[1] * len(gamma_names), # 0 <= gamma_i_j <= 1 variable binaire
                    types=["B"] * len(gamma_names))

# Définir la fonction objectif: minimiser la somme des pénalités
objective = [(alpha_names[i], c_minus[i]) for i in range(n)] + [(beta_names[i], c_plus[i]) for i in range(n)] # alpha_i * c_i- + beta_i * c_i+
model.objective.set_linear(objective)

# Étape 3: Ajout des contraintes
for i in range(n):
    # Contrainte (3): Xi - Ti = αi - βi
    ind = [x_names[i], alpha_names[i], beta_names[i]]
    val = [1.0, -1.0, 1.0]
    model.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=ind, val=val)],  # Xi - αi + βi
        senses=["E"],                                   # ==
        rhs=[T[i]]                                      # Ti
    )

    # Contrainte (7): (Somme(r=1 à m) de γir) = 1
    ind = [gamma_names[gamma_names.index(f"gamma_{i}_{r}")] for r in range(m)]
    val = [1.0] * m
    model.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=ind, val=val)],  # y1r + y2r + ...
        senses=["E"],                                   # ==
        rhs=[1]                                         # 1
    )

    for j in range(n):
        # Contrainte (4): Xj - Xi >= Sij * δij - M * Yji
        if i != j:
            # Récupérer l'indice de y_ji
            y_ji_idx = y_names.index(f"y_{j}_{i}")
            delta_ij_idx = delta_names.index(f"delta_{i}_{j}")

            ind = [x_names[j], x_names[i], delta_names[delta_ij_idx], y_names[y_ji_idx]]
            val = [1.0, -1.0, -s[i][j], M]
            model.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=ind, val=val)],  # Xj - Xi - Sij * δij + M * Yji
                senses=["G"],                                   # >=
                rhs=[0]                                         # 0
            )

        # Contrainte (5): Yij + Yji = 1
        if i < j:  # Pour éviter la redondance, on ne considère que les paires (i,j) où i < j
            y_ij_idx = y_names.index(f"y_{i}_{j}")
            y_ji_idx = y_names.index(f"y_{j}_{i}")

            ind = [y_names[y_ij_idx], y_names[y_ji_idx]]
            val = [1.0, 1.0]
            model.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=ind, val=val)],  # Yij + Yji
                senses=["E"],                                   # ==
                rhs=[1]                                         # 1
            )
        
        # Contrainte (6): δij >= γir + γjr - 1
        if i != j:
            delta_ij_idx = delta_names.index(f"delta_{i}_{j}")

            for r in range(m):
                gamma_ir_idx = gamma_names.index(f"gamma_{i}_{r}")
                gamma_jr_idx = gamma_names.index(f"gamma_{j}_{r}")

                ind = [delta_names[delta_ij_idx], gamma_names[gamma_ir_idx], gamma_names[gamma_jr_idx]]
                val = [1.0, -1.0, -1.0]
                model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=ind, val=val)],  # δij - γir - γjr
                    senses=["G"],                                   # >=
                    rhs=[-1]                                        # -1
                )
        

# Étape 4: Résolution et affichage de la solution
try:
    model.solve()

    print("\nSolution trouvée :")
    print("Valeur optimale de l'objectif:", model.solution.get_objective_value())

    # Afficher les heures d'atterrissage
    print("\nHeures d'atterrissage optimales :")
    for i in range(n):
        x_val = model.solution.get_values(x_names[i])
        alpha_val = model.solution.get_values(alpha_names[i])
        beta_val = model.solution.get_values(beta_names[i])

        # Trouver la piste assignée
        assigned_runway = -1
        for r in range(m):
            gamma_val = model.solution.get_values(f"gamma_{i}_{r}")
            if abs(gamma_val - 1) < 1e-6:  # Presque égal à 1
                assigned_runway = r
                break

        status = "à l'heure"
        if alpha_val > 1e-6:
            status = f"en avance de {alpha_val:.2f}"
        elif beta_val > 1e-6:
            status = f"en retard de {beta_val:.2f}"

        print(f"Avion {i}: atterrissage à {x_val:.2f} ({status}) sur la piste {assigned_runway}")

    # Afficher l'ordre d'atterrissage
    print("\nOrdre d'atterrissage par piste :")
    landing_order = {}

    for i in range(n):
        for r in range(m):
            gamma_val = model.solution.get_values(f"gamma_{i}_{r}")
            if abs(gamma_val - 1) < 1e-6:  # Si l'avion i est assigné à la piste r
                x_val = model.solution.get_values(x_names[i])
                if r not in landing_order:
                    landing_order[r] = []
                landing_order[r].append((i, x_val))

    for r in landing_order:
        # Trier les avions par heure d'atterrissage
        landing_order[r].sort(key=lambda x: x[1])
        print(f"Piste {r}: {[i for i, _ in landing_order[r]]}")

except cplex.exceptions.CplexError as e:
    print(f"Erreur lors de la résolution: {e}")