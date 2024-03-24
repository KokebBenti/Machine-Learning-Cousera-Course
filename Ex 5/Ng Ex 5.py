# Import Libraries
import numpy as np
import matplotlib.pyplot as plt

#Load data
from scipy.io import loadmat
df1=loadmat("ex5data1.mat")
xt=np.array(df1["X"])
yt=(np.array(df1["y"])).flatten()
mt=len(yt)
xv=np.array(df1["Xval"])
yv=np.array(df1["yval"])
mv=len(yv)
xtt=np.array(df1["Xtest"])
ytt=np.array(df1["ytest"])
mtt=len(ytt)

#Visualize data
plt.scatter(xt,yt)
plt.show()

#Linear Regression
def LinReg(xt,yt):
 mt=len(yt)
 a=0.002
 x0t=np.ones(mt)
 x1t=np.c_[x0t,xt]
 thetas=np.ones(2)
 ht=np.dot(x1t,thetas)
 cost=(np.sum((ht-yt)**2))/(2*mt)
 j=[]
 costs=[]
 for t in range(0,16000):
  thetas[0]=thetas[0]-(a/mt)*(np.dot((ht-yt),x0t))
  thetas[1]=thetas[1]-(a/mt)*(np.dot((ht-yt),x1t[:,1:]))
  ht=np.dot(x1t,thetas)
  j.append(t)
  costs.append(np.sum((ht-yt)**2)/(2*mt))
 return(ht)

#Visualize result
plt.scatter(xt,yt)
plt.plot(xt,LinReg(xt,yt))
plt.show()

#using Sklearn LinearRegression
from sklearn.base import BaseEstimator
class LR(BaseEstimator):
 def __init__(self):
  from sklearn.linear_model import LinearRegression
  self.lin_reg = LinearRegression()
 def fit(self,xt,yt):
  mt=len(yt)
  x0t=np.ones(mt)
  x1t= np.c_[x0t,xt]
  self.lin_reg.fit(x1t,yt)
  hp=self.lin_reg.predict(x1t)
  from sklearn.metrics import mean_squared_error
  lin_mse = mean_squared_error(yt,hp)
  return(lin_mse,self.lin_reg.coef_,self.lin_reg.intercept_)
 def predict(self,xt):
  mt=len(xt)
  x0t=np.ones(mt)
  x1t=np.c_[x0t,xt]
  hp=self.lin_reg.predict(x1t)
  return(hp)
lr=LR()

#Learning Curve
def LearningCurve(xt,yt,xv,yv):
 errors_train = []
 errors_validate = []
 j=[]
 for i in range(1,len(xt)+1):
  x=xt[0:i]
  y=yt[0:i]
  j.append(i)
  cost,coef,b=lr.fit(x,y)
  errors_train.append(cost)
  hv=lr.predict(xv)
  from sklearn.metrics import mean_squared_error
  lin_mse=mean_squared_error(yv,hv)
  errors_validate.append(lin_mse)
 plt.plot(j,errors_train)
 plt.plot(j,errors_validate)
 plt.show()
LearningCurve(xt,yt,xv,yv)

#Polynomial Regression
def FeatureNorm(xt):
 x1_m=np.mean(xt)
 x1_s=np.std(xt)
 xt1=(xt-x1_m)/x1_s
 return(xt1)

def PolyReg(xt):
 xtn=FeatureNorm(xt)
 xn=np.c_[xtn,xtn**2,xtn**3,xtn**4,xtn**5,xtn**6,xtn**7,xtn**8]
 return (xn)
x1t=PolyReg(xt)
cost,coef,b=lr.fit(x1t,yt)

xlim=np.sort(xt,axis=None)
x1lim=PolyReg(xlim)
hlim=lr.predict(x1lim)
plt.scatter(xt,yt)
plt.plot(xlim,hlim)
plt.show()

x1v=PolyReg(xv)
LearningCurve(x1t,yt,x1v,yv)

#Regularization
from sklearn.base import BaseEstimator
class SGDR(BaseEstimator):
 def __init__(self,alpha,eta0):
  from sklearn.linear_model import SGDRegressor
  self.sgd_reg = SGDRegressor(max_iter=100000,penalty="l2",alpha=alpha,eta0=eta0)
 def fit(self,xt,yt):
  mt=len(yt)
  x0t=np.ones(mt)
  x1t= np.c_[x0t,xt]
  self.sgd_reg.fit(x1t,yt)
  hp=self.sgd_reg.predict(x1t)
  from sklearn.metrics import mean_squared_error
  lin_mse = mean_squared_error(yt,hp)
  return(lin_mse,self.sgd_reg.coef_,self.sgd_reg.intercept_)
 def predict(self,xt):
  mt=len(xt)
  x0t=np.ones(mt)
  x1t=np.c_[x0t,xt]
  hp=self.sgd_reg.predict(x1t)
  return(hp)
sgdr=SGDR(0,0.003)
sgdr.fit(x1t,yt)
z=sgdr.predict(x1t)

def LearningCurveSGDR(xt,yt,xv,yv):
 errors_train = []
 errors_validate = []
 j=[]
 for i in range(1,len(xt)+1):
  x=xt[0:i]
  y=yt[0:i]
  j.append(i)
  cost,coef,b=sgdr.fit(x,y)
  errors_train.append(cost)
  hv=sgdr.predict(xv)
  from sklearn.metrics import mean_squared_error
  lin_mse=mean_squared_error(yv,hv)
  errors_validate.append(lin_mse)
 plt.plot(j,errors_train)
 plt.plot(j,errors_validate)
 plt.show()
 return(errors_train[-1],errors_validate[-1])
LearningCurveSGDR(x1t,yt,x1v,yv)

#lambda=1
sgdr=SGDR(1,0.003)
sgdr.fit(x1t,yt)
LearningCurveSGDR(x1t,yt,x1v,yv)
xlim=np.sort(xt,axis=None)
x1lim=PolyReg(xlim)
hlim=sgdr.predict(x1lim)
plt.scatter(xt,yt)
plt.plot(xlim,hlim)
plt.show()

#lambda=100
sgdr=SGDR(100,0.003)
sgdr.fit(x1t,yt)
LearningCurveSGDR(x1t,yt,x1v,yv)
xlim=np.sort(xt,axis=None)
x1lim=PolyReg(xlim)
hlim=sgdr.predict(x1lim)
plt.scatter(xt,yt)
plt.plot(xlim,hlim)
plt.show()

#select lambda
alpha=[0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]
te=[]
ve=[]
for n in range(len(alpha)):
 sgdr=SGDR(alpha[n],0.003)
 sgdr.fit(x1t,yt)
 tn,vn=LearningCurveSGDR(x1t,yt,x1v,yv)
 te.append(tn)
 ve.append(vn)
plt.plot(alpha,te)
plt.plot(alpha,ve)
plt.show()


#test error
sgdr=SGDR(0.3,0.003)
sgdr.fit(x1t,yt)
x1tt=PolyReg(xtt)
htt=sgdr.predict(x1tt)
htt2=lr.predict(x1tt)
from sklearn.metrics import mean_squared_error
lin_mse=mean_squared_error(ytt,htt)









