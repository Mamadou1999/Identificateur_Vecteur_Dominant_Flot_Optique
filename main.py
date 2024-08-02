import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

file_path = 'resultats_flot_optique.csv'
data = pd.read_csv(file_path)

print("Aperçu des données :")
print(data.head())

seuil_magnitude = 3.0

moyenne_direction = np.mean(data['direction'])
ecart_type_direction = np.std(data['direction'])

intervalle_direction = (moyenne_direction - ecart_type_direction, moyenne_direction + ecart_type_direction)

vecteurs_filtres = data[(data['magnitude'] >= seuil_magnitude) &
                        (data['direction'] >= intervalle_direction[0]) &
                        (data['direction'] <= intervalle_direction[1])]

print(f"Nombre de vecteurs conservés après filtrage : {vecteurs_filtres.shape[0]}")

plt.figure(figsize=(12, 6))
plt.hist(data['magnitude'], bins=30, alpha=0.7, color='b')
plt.title('Distribution des Magnitudes des Vecteurs de Flot Optique')
plt.xlabel('Magnitude')
plt.ylabel('Fréquence')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.hist(data['direction'], bins=30, alpha=0.7, color='g')
plt.title('Distribution des Directions des Vecteurs de Flot Optique')
plt.xlabel('Direction (radians)')
plt.ylabel('Fréquence')
plt.grid(True)
plt.show()

vecteurs_deplacement = vecteurs_filtres[['u_deplacement', 'v_deplacement']].values

def determine_k(vecteurs):
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(vecteurs)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Nombre de clusters (k)')
    plt.ylabel('Distorsion')
    plt.title('Méthode du coude pour déterminer le nombre optimal de clusters')
    plt.grid(True)
    plt.show()

vecteurs_sample = vecteurs_deplacement[
    np.random.choice(vecteurs_deplacement.shape[0], int(vecteurs_deplacement.shape[0] * 0.2), replace=False)]

determine_k(vecteurs_sample)

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(vecteurs_deplacement)

vecteurs_filtres = vecteurs_filtres.copy()
vecteurs_filtres['cluster'] = labels

cluster_counts = vecteurs_filtres['cluster'].value_counts()
cluster_dominant = cluster_counts.idxmax()

vecteurs_dominants = vecteurs_filtres[vecteurs_filtres['cluster'] == cluster_dominant]

direction_moyenne = vecteurs_dominants['direction'].mean()
intensite_moyenne = vecteurs_dominants['magnitude'].mean()

print(f"Direction moyenne du cluster dominant : {direction_moyenne:.4f} radians")
print(f"Intensité moyenne du cluster dominant : {intensite_moyenne:.2f}")

plt.figure(figsize=(10, 8))
colors = ['b', 'g', 'c']
for cluster in range(3):
    cluster_vectors = vecteurs_filtres[vecteurs_filtres['cluster'] == cluster]
    plt.quiver(
        cluster_vectors['x_initial'],
        cluster_vectors['y_initial'],
        cluster_vectors['u_deplacement'],
        cluster_vectors['v_deplacement'],
        scale=70,
        color=colors[cluster],
        alpha=0.5,
        width=0.002, 
        label=f'Cluster {cluster}'
    )

plt.quiver(
    vecteurs_dominants['x_initial'],
    vecteurs_dominants['y_initial'],
    vecteurs_dominants['u_deplacement'],
    vecteurs_dominants['v_deplacement'],
    scale=70,
    color='red',
    alpha=0.7,
    width=0.002, 
    label='Cluster Dominant'
)

plt.title('Vecteurs de Flot Optique et des Clusters')
plt.xlabel('Position X Initiale')
plt.ylabel('Position Y Initiale')
plt.legend()
plt.grid(True)
plt.show()