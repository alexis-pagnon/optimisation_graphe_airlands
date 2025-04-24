# --------------------------------------------------
# Projet Optimisation et Graphes
# JAMIN Antoine, PAGNON Alexis, SANCHEZ Adam, DUMOULIN Simon
# 31/05/2025
# --------------------------------------------------

import cplex
import numpy as np
import random

"""
Problème 3: Minimisation de la latence totale avec temps de roulage

Contraintes générales:
Ei<=Xi<=Li                                 (2)
lateness_i >= xi + ti - Ai                 (3)
Xj-Xi>=Sij*δij-M*Yji                       (4)
Yij+Yji=1                                  (5)
δij>=γir+γjr-1                             (6)
Somme(r=1 à m) de γir = 1                  (7)
Yij, δij, γir ∈ {0,1}                      (8)
Xi >= 0                                    (9)

Objectif: min Σ max(0, xi + ti - Ai)
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
A = []  # Heure limite d'arrivée au point de stationnement (supposée égale à Ti pour chaque avion)

line = airland_file.readline().strip()
n = int(line.split(" ")[0])
m = 3  # Nombre de pistes d'atterrissage

# Lire le fichier ligne par ligne
for i in range(0, n):
    line = airland_file.readline().strip()

    # Ajouter les valeurs dans les listes appropriées
    values = line.split(" ")

    E.append(float(values[1]))
    T.append(float(values[2]))
    L.append(float(values[3]))
    A.append(float(values[2]))  # A_i = T_i selon l'énoncé

    # Récupère les S_i
    s_i = []
    while (len(s_i) < n):
        line = airland_file.readline().strip()
        values = line.split(" ")
        for v in values:
            s_i.append(float(v))
    s.append(s_i)

airland_file.close()

# Génération des temps de roulage t_ir selon l'énoncé: t_ir ~ Uniform(1, Ti - Ei)
t = []  # Tableau de temps de roulage
for i in range(n):
    # Générer un temps de roulage uniforme entre 1 et Ti - Ei
    t_i = random.uniform(1, max(1, T[i] - E[i]))
    t.append(t_i)

print(f"Nombre d'avions: {n}")
print(f"Heures d'atterrissage au plus tôt: {E}")
print(f"Heures limites d'arrivée au stationnement: {A}")
print(f"Exemple de temps de roulage (premier avion): {t[0]}")

# Étape 1: Création du modèle
model = cplex.Cplex()
model.objective.set_sense(model.objective.sense.minimize)

# Calculer une grande valeur M pour les contraintes
M = max(L) + max([max(s_i) for s_i in s]) * n

# Étape 2: Ajout des variables de décision
# x_i: heure d'atterrissage de l'avion i
x_names = [f"x_{i}" for i in range(n)]
model.variables.add(names=x_names, lb=[E[i] for i in range(n)], ub=[L[i] for i in range(n)])

# y_ij: variable binaire pour l'ordre d'atterrissage
y_names = [f"y_{i}_{j}" for i in range(n) for j in range(n) if i != j]
model.variables.add(names=y_names, lb=[0] * len(y_names), ub=[1] * len(y_names), types=["B"] * len(y_names))

# delta_ij: variable binaire pour l'affectation des pistes
delta_names = [f"delta_{i}_{j}" for i in range(n) for j in range(n) if i != j]
model.variables.add(names=delta_names, lb=[0] * len(delta_names), ub=[1] * len(delta_names),
                    types=["B"] * len(delta_names))

# gamma_ir: variable binaire pour l'affectation de l'avion i à la piste r
gamma_names = [f"gamma_{i}_{r}" for i in range(n) for r in range(m)]
model.variables.add(names=gamma_names, lb=[0] * len(gamma_names), ub=[1] * len(gamma_names),
                    types=["B"] * len(gamma_names))

# lateness_i: variable représentant max(0, xi + t_ir_i - Ai) pour chaque avion
lateness_names = [f"lateness_{i}" for i in range(n)]
model.variables.add(names=lateness_names, lb=[0] * n)

# Définir la fonction objectif: minimiser la somme des lateness
objective = [(lateness_names[i], 1.0) for i in range(n)]
model.objective.set_linear(objective)

# Étape 3: Ajout des contraintes
for i in range(n):
    # Contrainte (7): Somme(r=1 à m) de γir = 1
    ind = [gamma_names[gamma_names.index(f"gamma_{i}_{r}")] for r in range(m)]
    val = [1.0] * m
    model.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=ind, val=val)],
        senses=["E"],
        rhs=[1]
    )

    # Contrainte (3): lateness_i >= xi + t_i - Ai
    ind = [lateness_names[i], x_names[i]]
    val = [1.0, -1.0]
    model.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=ind, val=val)],
        senses=["G"],
        rhs=[-A[i] + t[i]]
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
                lin_expr=[cplex.SparsePair(ind=ind, val=val)],
                senses=["G"],
                rhs=[0]
            )

        # Contrainte (5): Yij + Yji = 1
        if i < j:  # Pour éviter la redondance
            y_ij_idx = y_names.index(f"y_{i}_{j}")
            y_ji_idx = y_names.index(f"y_{j}_{i}")

            ind = [y_names[y_ij_idx], y_names[y_ji_idx]]
            val = [1.0, 1.0]
            model.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=ind, val=val)],
                senses=["E"],
                rhs=[1]
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
                    lin_expr=[cplex.SparsePair(ind=ind, val=val)],
                    senses=["G"],
                    rhs=[-1]
                )

# Étape 4: Résolution et affichage de la solution
try:
    # Option pour limiter le temps de calcul si nécessaire
    model.parameters.timelimit.set(3600)  # 1 heure de temps limite

    model.solve()

    print("\nSolution trouvée :")
    print(f"Latence totale: {model.solution.get_objective_value():.2f}")

    # Afficher les détails pour chaque avion
    print("\nHoraires et latences par avion:")
    total_lateness = 0
    for i in range(n):
        x_val = model.solution.get_values(x_names[i])
        lateness_val = model.solution.get_values(lateness_names[i])
        total_lateness += lateness_val

        # Trouver la piste assignée et le temps de roulage associé
        assigned_runway = -1
        runway_taxi_time = 0
        for r in range(m):
            gamma_val = model.solution.get_values(f"gamma_{i}_{r}")
            if abs(gamma_val - 1) < 1e-6:  # Presque égal à 1
                assigned_runway = r
                break

        # Calculer l'heure d'arrivée au stationnement
        parking_time = x_val + t[i]

        print(f"Avion {i}:")
        print(f"  Atterrissage à {x_val:.2f}")
        print(f"  Piste assignée: {assigned_runway}")
        print(f"  Temps de roulage: {t[i]:.2f}")
        print(f"  Arrivée au stationnement à {parking_time:.2f}")
        print(f"  Heure limite au stationnement: {A[i]}")
        print(f"  Latence: {lateness_val:.2f}")

    print(f"\nSomme des latences: {total_lateness:.2f}")

    # Afficher l'ordre d'atterrissage par piste
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