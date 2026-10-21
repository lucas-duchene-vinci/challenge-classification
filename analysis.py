import numpy as np
from dataset import load_data
import matplotlib.pyplot as plt
data = load_data("datasets/manchots.csv")

columns = [
    "species",
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex"
]

print("Dimensions :", data.shape)
print("Valeurs manquantes par colonne :")
print(dict(zip(columns, np.isnan(data).sum(axis=0))))

cleaned_data = data[~np.isnan(data).any(axis=1)]

print("Lignes avant nettoyage :", len(data))
print("Lignes après nettoyage :", len(cleaned_data))
print("Lignes supprimées :", len(data) - len(cleaned_data))

species, counts = np.unique(
    cleaned_data[:, 0],
    return_counts=True
)
plt.figure()
plt.bar(species, counts)
plt.xlabel("Espèce")
plt.ylabel("Nombre de manchots")
plt.title("Répartition des espèces")
plt.xticks([0, 1, 2], ["Adélie", "Papou", "Jugulaire"])
plt.savefig("graphiques/repartition_especes.png", dpi=300, bbox_inches="tight")


plt.figure()
for s, name in [(0, "Adélie"), (1, "Papou"), (2, "Jugulaire")]:
    weights = cleaned_data[cleaned_data[:, 0] == s, 4]
    plt.hist(weights, alpha=0.5, label=name)

plt.xlabel("Poids (g)")
plt.ylabel("Nombre de manchots")
plt.title("Distribution du poids selon l'espèce")
plt.legend()
plt.savefig("graphiques/distribution_poids.png", dpi=300, bbox_inches="tight")

# Graphique : longueur des nageoires selon l'espèce
plt.figure()

for s, name in [(0, "Adélie"), (1, "Papou"), (2, "Jugulaire")]:
    rows = cleaned_data[cleaned_data[:, 0] == s]
    flipper_lengths = rows[:, 3]

    plt.hist(
        flipper_lengths,
        alpha=0.5,
        label=name
    )

plt.xlabel("Longueur des nageoires (mm)")
plt.ylabel("Nombre de manchots")
plt.title("Distribution de la longueur des nageoires selon l'espèce")
plt.legend()
plt.savefig("graphiques/longueur_nageoires.png", dpi=300, bbox_inches="tight")
plt.figure()
for s, name in [(0, "Adélie"), (1, "Papou"), (2, "Jugulaire")]:
    rows = cleaned_data[cleaned_data[:, 0] == s]
    plt.scatter(rows[:, 1], rows[:, 2], label=name)

plt.xlabel("Longueur du bec (mm)")
plt.ylabel("Épaisseur du bec (mm)")
plt.title("Longueur et épaisseur du bec selon l'espèce")
plt.legend()
plt.savefig("graphiques/dimensions_bec.png", dpi=300, bbox_inches="tight")

plt.show()
plt.savefig(".")


"""

"""