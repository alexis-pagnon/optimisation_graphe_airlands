# --------------------------------------------------
# Projet Optimisation et Graphes
# JAMIN Antoine, PAGNON Alexis, SANCHEZ Adam, DUMOULIN Simon
# 31/05/2025
# --------------------------------------------------

import docplex
import docplex.mp
import docplex.mp.model
import numpy as np
import random
random.seed(42)

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
airland_file = open("airlands/airland1.txt")

E = []  # Tableau des Ei (earliest landing time)
T = []  # Tableau des Ti (target landing time)
L = []  # Tableau des Li (latest landing time)
s = []  # Tableau des s_i, étant eux-même des tableaux de s_ij
A = []  # Heure limite d'arrivée au point de stationnement (supposée égale à Ti pour chaque avion)

line = airland_file.readline().strip()
n = int(line.split(" ")[0])
m = 4  # Nombre de pistes d'atterrissage

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
model = docplex.mp.model.Model("Probleme2")

# Calculer une grande valeur M pour les contraintes
M = max(L) + max([max(s_i) for s_i in s]) * n

# Étape 2: Ajout des variables de décision
# x_i: heure d'atterrissage de l'avion i
x = model.continuous_var_dict(range(n), lb=lambda i: E[i], ub=lambda i: L[i], name=lambda i: f"x_{i}")

# y_ij: variable binaire pour l'ordre d'atterrissage
y = model.binary_var_dict([(i, j) for i in range(n) for j in range(n) if i != j], name=lambda ij: f"y_{ij[0]}_{ij[1]}")

# delta_ij: variable binaire pour l'affectation des pistes
delta = model.binary_var_dict([(i, j) for i in range(n) for j in range(n) if i != j], name=lambda ij: f"delta_{ij[0]}_{ij[1]}")

# gamma_ir: variable binaire pour l'affectation de l'avion i à la piste r
gamma = model.binary_var_dict([(i, r) for i in range(n) for r in range(m)], name=lambda ir: f"gamma_{ir[0]}_{ir[1]}")

# lateness_i: variable représentant max(0, xi + ti - Ai) pour chaque avion
lateness = model.continuous_var_dict(range(n), lb=0, name=lambda i: f"lateness_{i}")

# Définir la fonction objectif: minimiser la somme des lateness
model.minimize(sum(lateness[i] for i in range(n)))

# Étape 3: Ajout des contraintes
for i in range(n):
    # Contrainte (7): Somme(r=1 à m) de γir = 1
    model.add_constraint(sum(gamma[i, r] for r in range(m)) == 1)

    # Contrainte (3): lateness_i >= xi + ti - Ai
    model.add_constraint(lateness[i] >= x[i] + t[i] - A[i])

    for j in range(n):
        # Contrainte (4): Xj - Xi >= Sij * δij - M * Yji
        if i != j:
            model.add_constraint(x[j] - x[i] >= s[i][j] * delta[i, j] - M * y[j, i])

        # Contrainte (5): Yij + Yji = 1
        if i < j:  # Pour éviter la redondance
            model.add_constraint(y[i, j] + y[j, i] == 1)

        # Contrainte (6): δij >= γir + γjr - 1
        if i != j:
            for r in range(m):
                model.add_constraint(delta[i, j] >= gamma[i, r] + gamma[j, r] - 1)

# Étape 4: Résolution et affichage de la solution
try:
    solution = model.solve()

    if solution:
        print("\nSolution trouvée :")
        print(f"Latence totale: {model.objective_value:.2f}")

        # Afficher les détails pour chaque avion
        print("\nHoraires et latences par avion:")
        total_lateness = 0
        for i in range(n):
            x_val = solution.get_value(x[i])
            lateness_val = solution.get_value(lateness[i])
            total_lateness += lateness_val

            # Trouver la piste assignée et le temps de roulage associé
            assigned_runway = -1
            for r in range(m):
                gamma_val = solution.get_value(gamma[i, r])
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
                gamma_val = solution.get_value(gamma[i, r])
                if abs(gamma_val - 1) < 1e-6:  # Si l'avion i est assigné à la piste r
                    x_val = solution.get_value(x[i])
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