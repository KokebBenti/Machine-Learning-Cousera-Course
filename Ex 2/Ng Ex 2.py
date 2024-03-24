# Logistic Regression
# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Visualize data
d1=pd.read_csv("ex2data1.txt",header=None)
d1.info()
x1=d1[0]
x2=d1[1]
y=d1[2]
m=len(y)
plt.scatter(x1.loc[y==0],x2.loc[y==0],c="y",marker="o",label="not admitted")
plt.scatter(x1.loc[y==1],x2.loc[y==1],c="k",marker="+",label="admitted")
plt.legend()
plt.show()

#Feature Normalization
x1_m=np.mean(x1)
x1_s=np.std(x1)
x2_m=np.mean(x2)
x2_s=np.std(x2)
x1=(x1-x1_m)/x1_s
x2=(x2-x2_m)/x2_s


#Apply Logistic regression
x0=np.ones(m)
xt=np.c_[x0,x1,x2]
thetas=np.array([0,0,0])
s=np.dot(xt,thetas)
h=1/(1+np.exp(-s))
cost=np.sum((-y*np.log(h)-(1-y)*np.log(1-h))/m)
j=[]
costs=[]
for t in range(0,15000):
 thetas=thetas-(0.1/m)*(np.dot((h-y),xt))
 s = np.dot(xt,thetas)
 h = 1/(1+np.exp(-s))
 j.append(t)
 costs.append(np.sum(-y*np.log(h)-(1-y)*np.log(1-h))/m)

#Visualize result
plt.plot(j,costs)
plt.show()

boundary_line=-(np.dot(thetas[1],x1)/thetas[2])-(thetas[0]/thetas[2])
plt.scatter(x1.loc[y==0],x2.loc[y==0],c="y",marker="o",label="not admitted")
plt.scatter(x1.loc[y==1],x2.loc[y==1],c="k",marker="+",label="admitted")
plt.plot(x1,boundary_line)
plt.legend()
plt.show()

#Test values
x1test=np.array([45])
x2test=np.array([85])
x1test=(x1test-x1_m)/x1_s
x2test=(x2test-x2_m)/x2_s
xtest0=np.ones(1)
xtestt=np.c_[xtest0,x1test,x2test]
s=np.dot(xtestt,thetas)
h=1/(1+np.exp(-s))
print("Probablity of acceptance = "+str(h))



# Regularized Logistic Regression
# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Visualize data
d1=pd.read_csv("ex2data2.txt",header=None)
d1.info()
x1=d1[0]
x2=d1[1]
y=d1[2]
m=len(y)
plt.scatter(x1.loc[y==0],x2.loc[y==0],c="y",marker="o",label="y=0")
plt.scatter(x1.loc[y==1],x2.loc[y==1],c="k",marker="+",label="y=1")
plt.legend()
plt.show()

#Feature Normalization
x1_m=np.mean(x1)
x1_s=np.std(x1)
x2_m=np.mean(x2)
x2_s=np.std(x2)
x1=(x1-x1_m)/x1_s
x2=(x2-x2_m)/x2_s


#Apply Logistic regression
x0=np.ones(m)
xt=np.c_[x0,x1,x2,x1**2,x1*x2,x2**2,x1**3,x1**2*x2,x1*x2**2,x2**3,x1**4,x1**3*x2,x1**2*x2**2,
x1*x2**3,x2**4,x1**5,x1**4*x2,x1**3*x2**2,x1**2*x2**3,x1*x2**4,x2**5,x1**6,x1**5*x2,x1**4*x2**2,x1**3*x2**3,
x1**2*x2**4,x1*x2**5,x2**6]
thetas=np.zeros(28)
s=np.dot(xt,thetas)
h=1/(1+np.exp(-s))
cost=np.sum((-y*np.log(h)-(1-y)*np.log(1-h))/m)+3*np.sum(thetas**2)/(2*m)
j=[]
costs=[]
for t in range(0,1500):
 thetas[0]=thetas[0]-(0.1/m)*(np.dot((h-y),x0))
 thetas[1:]=thetas[1:]*(1-0.1*3/m)-(0.1/m)*(np.dot((h-y),xt[:,1:]))
 s = np.dot(xt,thetas)
 h = 1/(1+np.exp(-s))
 j.append(t)
 costs.append(np.sum(-y*np.log(h)-(1-y)*np.log(1-h))/m)

#Visualize result
plt.plot(j,costs)
plt.show()

#using a simpler circular model
x0=np.ones(m)
xt=np.c_[x0,x1,x2,x1**2,x1*x2,x2**2]
thetas=np.zeros(6)
s=np.dot(xt,thetas)
h=1/(1+np.exp(-s))
cost=np.sum((-y*np.log(h)-(1-y)*np.log(1-h))/m)+3*np.sum(thetas**2)/(2*m)
j=[]
costs=[]
for t in range(0,1500):
 thetas[0]=thetas[0]-(0.1/m)*(np.dot((h-y),x0))
 thetas[1:]=thetas[1:]*(1-0.1*3/m)-(0.1/m)*(np.dot((h-y),xt[:,1:]))
 s = np.dot(xt,thetas)
 h = 1/(1+np.exp(-s))
 j.append(t)
 costs.append(np.sum(-y*np.log(h)-(1-y)*np.log(1-h))/m)

x1b=np.linspace(x1.min(),x1.max())
bl1,bl2=[],[]
for r in range(0,len(x1b)):
 tem=((thetas[2]+np.dot(thetas[4],x1b))**2)-4*thetas[5]*(thetas[0]+np.dot(thetas[1],x1b)+np.dot(thetas[3],x1b**2))
 if tem[r]<0:
   bl1.append(None)
   bl2.append(None)
 else:
   boundary_line1=(1/(2*thetas[5]))*((-thetas[2]-np.dot(thetas[4],x1b))+np.sqrt(((thetas[2]+np.dot(thetas[4],x1b))**2)-4*thetas[5]*(thetas[0]+np.dot(thetas[1],x1b)+np.dot(thetas[3],x1b**2))))
   boundary_line2=(1/(2*thetas[5]))*((-thetas[2]-np.dot(thetas[4],x1b))-np.sqrt(((thetas[2]+np.dot(thetas[4],x1b))**2)-4*thetas[5]*(thetas[0]+np.dot(thetas[1],x1b)+np.dot(thetas[3],x1b**2))))
   bl1.append(boundary_line1[r])
   bl2.append(boundary_line2[r])
plt.scatter(x1.loc[y==0],x2.loc[y==0],c="y",marker="o",label="y=0")
plt.scatter(x1.loc[y==1],x2.loc[y==1],c="k",marker="+",label="y=1")
plt.plot(x1b,bl1)
plt.plot(x1b,bl2)
plt.legend()
plt.show()
