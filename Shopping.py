import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import matplotlib
import numpy as np
import scipy.cluster.hierarchy as shc # thư viện phân cấp thứ bậc
from sklearn.cluster import AgglomerativeClustering # dùng sklean để gom nhóm các cluster

dataset = pd.read_csv("shopping_data.csv")

dataset.shape #trả về ma trận (200, 5) nghĩa là 200 dòng dữ liệu và 5 feature.
data = dataset.iloc[:, 3:5].values #Lấy từ cột index thứ 3 đến trước index 5
#Phân loại khách hàng theo thu nhập hằng năm(Annual Income) và điểm mua sắm (spending score)
plt.figure(figsize=(10, 7))
plt.title("Incoming and Score")
plt.scatter(data[:,0], data[:,1], cmap=cm.cmapname)

#Vẽ sơ đồ gộm
plt.figure(figsize=(10, 7))
plt.title("Sơ đồ phân cụm thứ bậc")
dend = shc.dendrogram(shc.linkage(data, method='ward'))
plt.show()

#dự vào biểu đồ ta chọn số cluster để phân cụm dữ liêu
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)


plt.figure(figsize=(10, 7))
plt.title("Dữ liệu sau khi phân cụm")
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')
plt.colorbar()
plt.show()

dataset.shape #trả về ma trận (200, 5) nghĩa là 200 dòng dữ liệu và 5 feature.
data = dataset.iloc[:, [2,3]].values #Lấy từ cột index thứ 3 đến trước index 5
#Phân loại khách hàng theo thu nhập hằng năm(Annual Income) và điểm mua sắm (spending score)

plt.figure(figsize=(10, 7))
plt.title("Age and Incoming")
plt.scatter(data[:,0], data[:,1], cmap=cm.cmapname)
plt.show()
#Vẽ sơ đồ gộm
plt.figure(figsize=(10, 7))
plt.title("Sơ đồ phân cụm thứ bậc")
dend = shc.dendrogram(shc.linkage(data, method='ward'))
plt.show()

#dự vào biểu đồ ta chọn số cluster để phân cụm dữ liêu
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)

plt.figure(figsize=(10, 7))
plt.title("Dữ liệu sau khi phân cụm")
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')
plt.colorbar()
plt.show()