# Import Libraries
import numpy as np
import matplotlib.pyplot as plt

#Load data
from scipy.io import loadmat
d1=loadmat("ex3data1.mat")

#training and test set
x=np.array(d1["X"])
y=np.array(d1["y"])
Total=np.c_[x,y]
from sklearn.model_selection import train_test_split
train,test=train_test_split(Total, test_size=0.2, random_state=42)
x=train[:,:-1]
y=train[:,-1:]
m=len(y)

#Visualize data
import matplotlib.image as mpimg
fig, axis = plt.subplots(10,10,figsize=(8,8))
for i in range(10):
 for j in range(10):
  axis[i,j].imshow(x[np.random.randint(0,4001),:].reshape(20,20,order="F"),cmap="binary")
  axis[i,j].axis("off")
plt.show()


#Apply Regularized Logistic regression
from sklearn.base import BaseEstimator
class OnevsAll(BaseEstimator):
 def __init__(self):
  self.thetas_k=[]
 def fit(self, x, y):
  lambda1 = 3
  self.thetas_k=[]
  for t in range(1,11):
   thetas=np.zeros(400)
   s=np.dot(x,thetas)
   h=1/(1+np.exp(-s))
   j=[]
   costs=[]
   ytmp=(y==t).astype(int).flatten()
   ytmp=np.array(ytmp)
   for l in range(0,1000):
    thetas[0]=thetas[0]-(0.1/m)*(np.dot((h-ytmp),x[:,0:1]))
    thetas[1:]=thetas[1:]*(1-0.1*lambda1/m)-(0.1/m)*(np.dot((h-ytmp),x[:,1:]))
    s=np.dot(x,thetas)
    h=1/(1+np.exp(-s))
    j.append(l)
    costs.append(np.sum(-ytmp*np.log(h)-(1-ytmp)*np.log(1-h))/m)
   self.thetas_k.append(thetas)
   '''plt.plot(j, costs)
   plt.xlabel("Number of Trials")
   plt.ylabel("Cost")
   plt.title("Cost Function "+str(t))
   plt.show()'''
  return (self.thetas_k)
 def predict(self, x):
  thetas_k=np.array(self.thetas_k)
  st = np.dot(x,thetas_k.T)
  ht = 1/(1+np.exp(-st))
  if x.shape==(len(x),):
   p=np.argmax(ht)
   y_pred=p+1
  else:
   p=np.argmax(ht,axis=1)
   y_pred=p+1
  return(y_pred)


classifier=OnevsAll()
classifier.fit(x,y)
x_test=x[1500]
z=classifier.predict(x)

#Performance Measuring
#Cross Validation
from sklearn.model_selection import cross_val_score
cross_val_score(classifier,x,y,cv=3, scoring="accuracy")

#Confusion Matrix
from sklearn.metrics import confusion_matrix
A=confusion_matrix(y,z)
plt.imshow(A,cmap=plt.cm.gray)
plt.show()

#using scikit-learn
from sklearn.linear_model import SGDClassifier
lr=SGDClassifier()
lr.fit(x,y)
y=y.flatten()
from sklearn.model_selection import cross_val_score
cross_val_score(lr,x,y,cv=3, scoring="accuracy")


#Testing
xt=test[:,:-1]
yt=test[:,-1:]
mt=len(yt)
zt=classifier.predict(xt)
from sklearn.model_selection import cross_val_score
cross_val_score(classifier,xt,yt,cv=3, scoring="accuracy")
from sklearn.metrics import confusion_matrix
At=confusion_matrix(yt,zt)
plt.imshow(At,cmap=plt.cm.gray)
plt.show()

#Feed Forward Neural Network
from scipy.io import loadmat
w=loadmat("ex3weights.mat")
w1=w["Theta1"]
w2=w["Theta2"]
from sklearn.base import BaseEstimator
class NeuralNetwork(BaseEstimator):
 def fit(self, x, y):
  m=len(y)
  a10=np.ones(m)
  a1=np.c_[a10,x]
  z2=np.dot(a1,w1.T)
  a2n=1/(1+np.exp(-z2))
  a20=np.ones(m)
  a2=np.c_[a20,a2n]
  z3=np.dot(a2,w2.T)
  a3=1/(1+np.exp(-z3))

 def predict(self, x):
  a10t=np.ones(len(x))
  a1t=np.c_[a10t,x]
  z2t=np.dot(a1t,w1.T)
  a2nt=1/(1+np.exp(-z2t))
  a20t=np.ones(len(x))
  a2t=np.c_[a20t,a2nt]
  z3t=np.dot(a2t,w2.T)
  a3t=1/(1+np.exp(-z3t))
  if x.shape == (len(x),):
   pt = np.argmax(a3t)
   y_pred = pt + 1
  else:
   pt = np.argmax(a3t, axis=1)
   y_pred = pt + 1
  return (y_pred)

classifier_NN=NeuralNetwork()
classifier_NN.fit(x,y)
x_test=x[1500]
z=classifier_NN.predict(x)

from sklearn.model_selection import cross_val_score
cross_val_score(classifier_NN,x,y,cv=3, scoring="accuracy")

