#K Means Clustering
# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Load Data
from scipy.io import loadmat
df1=loadmat("ex7data2.mat")
x=df1["X"]
m=len(x)

#Scale Input
x_m=np.mean(x)
x_s=np.std(x)
x=(x-x_m)/x_s


#Visualize data
x1=pd.Series(x[:,0])
x2=pd.Series(x[:,1])
plt.scatter(x1,x2,c="k",marker="o")
plt.show()

#Random intialization
c1=np.zeros((3,2))
for t in range(0,3):
 c1[t]=x[np.random.randint(0,m)]

#K-means clustering
costs=[]
t=[]
for l in range(0,10):
 cost=np.zeros((m,3))
 ci=np.zeros(m)
 for j in range (0,3):
  cost[:,j]=np.sum((x-c1[j])**2,axis=1)
 ci=np.argmin(cost,axis=1)
 t.append(l)
 costs.append(np.sum(np.min(cost,axis=1)))

 c1n=np.zeros((3,2))
 n1=np.zeros((3,1))
 for o in range(0,m):
  if ci[o]==0:
   c1n[0]=c1n[0]+x[o]
   n1[0]=n1[0]+1
  elif ci[o]==1:
   c1n[1]=c1n[1]+x[o]
   n1[1]=n1[1]+1
  elif ci[o]==2:
   c1n[2]=c1n[2]+x[o]
   n1[2]=n1[2]+1

 c1=c1n/n1

 x1=pd.Series(x[:,0])
 x2=pd.Series(x[:,1])
 plt.scatter(x1.loc[ci==0],x2.loc[ci==0],c="y",marker="+")
 plt.scatter(x1.loc[ci==1],x2.loc[ci==1],c="g",marker="+")
 plt.scatter(x1.loc[ci==2],x2.loc[ci==2],c="r",marker="+")
 plt.scatter(c1[:,0],c1[:,1],c="k",marker="o")
 plt.show()

plt.plot(t,costs)
plt.show()

#using Scikit learn
from sklearn.cluster import KMeans
k=4
kmeans = KMeans(n_clusters=k)
yp=kmeans.fit_transform(x)
kmeans.inertia_




#Image compression
#not finished
# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Load Data
import imageio.v2
df2=imageio.v2.imread("bird_small.png")
x=df2

#visualize data
plt.imshow(x)
plt.show()

#Scale Input
x=x.reshape(16384,3)
x_m=np.mean(x)
x_s=np.std(x)
x=(x-x_m)/x_s




#PCA
# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Load Data
from scipy.io import loadmat
df1=loadmat("ex7data1.mat")
x=df1["X"]
m=len(x)

#Visualize data
x1=pd.Series(x[:,0])
x2=pd.Series(x[:,1])
plt.scatter(x1,x2,c="k",marker="o")
plt.show()

#Calculate covariance matrix and SVD
x=(x-x.mean(axis=0))/(x.std(axis=0))
u,s,v=np.linalg.svd(x)

#plot principal component
x1=pd.Series(x[:,0])
x2=pd.Series(x[:,1])
plt.scatter(x1,x2,c="k",marker="o")
xl=[0,v.T[:,0][0]]
yl=[0,v.T[:,0][1]]
xm=[0,v.T[:,1][0]]
ym=[0,v.T[:,1][1]]
plt.plot(xl,yl)
plt.plot(xm,ym)
plt.show()

#reduce dimension
c=v.T[:,0]
xr=np.dot(x,c)

#plot result
c=c.reshape((2,1))
xr=xr.reshape((50,1))
xn=np.dot(xr,c.T)
x1=pd.Series(x[:,0])
x2=pd.Series(x[:,1])
plt.scatter(x1,x2,c="k",marker="o")
plt.scatter(xn[:,0],xn[:,1])
plt.show()


#Image compression
#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Load Data
from scipy.io import loadmat
df1=loadmat("ex7faces.mat")
x=df1["X"]
m=len(x)

#visualize data
im=x.reshape((5000,32,32))
plt.imshow(im[0])
plt.show()

#Calculate covariance matrix and SVD
x=(x-x.mean(axis=0))/(x.std(axis=0))
u,s,v=np.linalg.svd(x)

#reduce dimension
c=v.T[:,0:100]
xr=np.dot(x,c)

#Reshape result
xn=np.dot(xr,c.T)
im2=xn.reshape((5000,32,32))
plt.imshow(im2[0])
plt.show()

