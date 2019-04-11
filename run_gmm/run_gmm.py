import pandas as pd 
import matplotlib.pyplot as plt  

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics

dataset = pd.read_csv("dataset.csv",header = None)

print(dataset.head())

plt.scatter(dataset[0],dataset[1])
plt.savefig("scatter.png")

'''kmeans_predictions4 = KMeans(n_clusters = 4).fit_predict(dataset)# unsuvervised can combine predict at once after
plt.scatter(dataset[0],dataset[1], c= kmeans_predictions4)
plt.savefig("scatter_kmean4.png")
print("kmeans 4 clusters")
print(metrics.silhouette_score(dataset,kmeans_predictions4))


kmeans_predictions3 = KMeans(n_clusters = 3).fit_predict(dataset)# unsuvervised can combine predict at once after
plt.scatter(dataset[0],dataset[1], c= kmeans_predictions3)
plt.savefig("scatter_kmean3.png")
print("kmeans 3 clusters")
print(metrics.silhouette_score(dataset,kmeans_predictions3))


kmeans_predictions2 = KMeans(n_clusters = 4).fit_predict(dataset)# unsuvervised can combine predict at once after
plt.scatter(dataset[0],dataset[1], c= kmeans_predictions2)
plt.savefig("scatter_kmean2.png")
print("kmeans 2 clusters")
print(metrics.silhouette_score(dataset,kmeans_predictions2))'''

for i in range(4):
    n = i+2
    print(n)
    kmeans_predictions = KMeans(n_clusters = n).fit_predict(dataset)# unsuvervised can combine predict at once after
    plt.scatter(dataset[0],dataset[1], c= kmeans_predictions)
    plt.savefig("scatter_kmean"+str(n)+".png")
    print("kmeans " + str(n) + " clusters")
    print(metrics.silhouette_score(dataset,kmeans_predictions))

'''gaussian_predictions3 = GaussianMixture(n_components = 3).fit(dataset).predict(dataset)
plt.scatter(dataset[0],dataset[1], c = gaussian_predictions3)
plt.savefig("scatter_gaussian3.png")
print("gaussian 3 clusters")
print(metrics.silhouette_score(dataset,gaussian_predictions3))
'''

for i in range(5):
    n = i+2
    print(n)
    gaussian_predictions = GaussianMixture(n_components = n).fit(dataset).predict(dataset)#gausian do not fit_predict
    plt.scatter(dataset[0],dataset[1], c = gaussian_predictions)
    plt.savefig("scatter_gaussian" + str(n) + ".png")
    print("gaussian " +str(n) + " clusters")
    print(metrics.silhouette_score(dataset,gaussian_predictions))