# Import Libraries
import numpy as np
import matplotlib.pyplot as plt

#Load data
from scipy.io import loadmat
d2=loadmat("ex4data1.mat")
x=np.array(d2["X"])
y=np.array(d2["y"])
m=len(y)

#visualize data
import matplotlib.image as mpimg
fig, axis = plt.subplots(10,10,figsize=(8,8))
for i in range(10):
 for j in range(10):
  axis[i,j].imshow(x[np.random.randint(0,5001),:].reshape(20,20,order="F"),cmap="binary")
  axis[i,j].axis("off")
plt.show()

#Regularized Feed Forward Neural Network
from scipy.io import loadmat
w=loadmat("ex4weights.mat")
w1=(w["Theta1"]).T
w2=(w["Theta2"]).T
def CalculateCost(w1,w2):
 a10=np.ones(m)
 a1=np.c_[a10,x]
 z2=np.dot(a1,w1)
 a2n=1/(1+np.exp(-z2))
 a20=np.ones(m)
 a2=np.c_[a20,a2n]
 z3=np.dot(a2,w2)
 a3=1/(1+np.exp(-z3))
 yk=[]
 for i in range(0,len(y)):
  if y[i]==10:
   ytmp = np.zeros(10)
   ytmp[9] = 1
   yk.append(ytmp)
  else:
   ytmp=np.zeros(10)
   ytmp[int(y[i])-1]=1
   yk.append(ytmp)
 yk=np.array(yk)
 lambda1=1
 cost=(np.sum(-yk*np.log(a3)-(1-yk)*np.log(1-a3))/m)+(lambda1*np.sum(w1**2)/(2*m))+(lambda1*np.sum(w2**2)/(2*m))
 return(cost)
print(CalculateCost(w1,w2))


#Back Propagation using Regularized Sigmoid Function
from sklearn.base import BaseEstimator
class NNClassifier(BaseEstimator):
 def __init__(self):
  self.w01=2*0.12*np.random.random(size=(401,25))-0.12
  self.w02=2*0.12*np.random.random(size=(26,10))-0.12
 def fit(self,x,y):
  yk=[]
  for i in range(0,len(y)):
   if y[i]==10:
    ytmp = np.zeros(10)
    ytmp[9] = 1
    yk.append(ytmp)
   else:
    ytmp=np.zeros(10)
    ytmp[int(y[i])-1]=1
    yk.append(ytmp)
  yk=np.array(yk)
  j=[]
  costs=[]
  lambda1=1
  a=0.9
  for t in range(0,800):
   a10=np.ones(m)
   a1=np.c_[a10,x]
   z2=np.dot(a1,self.w01)
   a2n=1/(1+np.exp(-z2))
   a20=np.ones(m)
   a2=np.c_[a20,a2n]
   z3=np.dot(a2,self.w02)
   a3=1/(1+np.exp(-z3))
   delta3=a3-yk
   grad2=(1/(1+np.exp(-z2)))*(1-(1/(1+np.exp(-z2))))
   delta2=grad2*np.dot(delta3,self.w02.T)[:,1:]
   Delta1=(np.dot(a1.T,delta2))/m
   Delta2=(np.dot(a2.T,delta3))/m
   self.w01[0]=self.w01[0]-a*Delta1[0]
   self.w01[1:]=self.w01[1:]*(1-a*lambda1/m)-a*Delta1[1:]
   self.w02[0]=self.w02[0]-a*Delta2[0]
   self.w02[1:]=self.w02[1:]*(1-a*lambda1/m)-a*Delta2[1:]
   j.append(t)
   costs.append((np.sum(-yk*np.log(a3)-(1-yk)*np.log(1-a3))/m)+(lambda1*np.sum(self.w01**2)/(2*m))+(lambda1*np.sum(self.w02**2)/(2*m)))
  print(costs)
  plt.plot(j, costs)
  plt.xlabel("Number of Trials")
  plt.ylabel("Cost")
  plt.title("Cost Function")
  plt.show()
  grad=np.concatenate((np.ravel(Delta1),np.ravel(Delta2)))
  return(self.w01,self.w02,grad)
 def predict(self,x):
  a10=np.ones(m)
  a1=np.c_[a10,x]
  z2=np.dot(a1,self.w01)
  a2n=1/(1+np.exp(-z2))
  a20=np.ones(m)
  a2=np.c_[a20,a2n]
  z3=np.dot(a2,self.w02)
  a3=1/(1+np.exp(-z3))
  p=np.argmax(a3,axis=1)
  y_pred=p+1
  return(y_pred)

NNC=NNClassifier()
NNC.fit(x,y)
z=NNC.predict(x)

#Performance Measuring
accuracy=np.sum(z==y.flatten())*100/5000
print("Accuracy is "+ str(accuracy)+" %")

#Confusion Matrix
from sklearn.metrics import confusion_matrix
A=confusion_matrix(y,z)
plt.imshow(A,cmap=plt.cm.gray)
plt.show()


#Gradient checking
def GradientChecking():
 e=10**(-4)
 w01plus=w01+e
 w02plus=w02+e
 w01minus=w01-e
 w02minus=w02-e
 Jp=CalculateCost(w01plus,w02plus)
 Jm=CalculateCost(w01minus,w02minus)
 Grad=(Jp-Jm)/(2*e)
 return(Grad)
g=NNC.fit(x,y)
check=(GradientChecking()-g[2])

