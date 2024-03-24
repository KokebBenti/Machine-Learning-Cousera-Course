#import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Load Data
from scipy.io import loadmat
df1=loadmat("ex6data1.mat")
x=df1["X"]
y=(df1["y"]).flatten()
m=len(y)

#Visualize Data
x1=pd.Series(x[:,0])
x2=pd.Series(x[:,1])
plt.scatter(x1.loc[y==0],x2.loc[y==0],c="k",marker="o")
plt.scatter(x1.loc[y==1],x2.loc[y==1],c="y",marker="+")
plt.show()


#Linear SVM Classification (c=1)
from sklearn.svm import SVC
svc=SVC(C=1,kernel="linear")
xt=np.c_[x1,x2]
svc.fit(xt,y)
z=svc.predict(xt)

#(c=100)
from sklearn.svm import SVC
svc100=SVC(C=100,kernel="linear")
xt=np.c_[x1,x2]
svc100.fit(xt,y)
z2=svc100.predict(xt)

#Visualize Result
boundary_line=-((svc.coef_[:,0]*x1)/svc.coef_[:,1])-(svc.intercept_[0]/svc.coef_[:,1])
boundary_line2=-((svc100.coef_[:,0]*x1)/svc100.coef_[:,1])-(svc100.intercept_[0]/svc100.coef_[:,1])
plt.scatter(x1.loc[y==0],x2.loc[y==0],c="y",marker="o")
plt.scatter(x1.loc[y==1],x2.loc[y==1],c="k",marker="+")
plt.plot(x1,boundary_line,label="c=1")
plt.plot(x1,boundary_line2,label="c=100")
plt.legend()
plt.show()

#Performance Measuring
accuracy=np.sum(z==y)*100/(m)
print("Accuracy is "+ str(accuracy)+" %")




#SVM with Gaussian Kernel
#import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Load Data
from scipy.io import loadmat
df2=loadmat("ex6data2.mat")
x=df2["X"]
y=(df2["y"]).flatten()
m=len(y)

#Visualize Data
x1=pd.Series(x[:,0])
x2=pd.Series(x[:,1])
plt.scatter(x1.loc[y==0],x2.loc[y==0],c="y",marker="o")
plt.scatter(x1.loc[y==1],x2.loc[y==1],c="k",marker="+")
plt.show()

#Gaussian RBF Kernel
from sklearn.svm import SVC
svcrbf=SVC(C=1000,kernel="rbf",gamma=5)
xt=np.c_[x1,x2]
svcrbf.fit(xt,y)
z=svcrbf.predict(xt)

#Visualize Result
x1_min,x1_max=x1.min()-1,x1.max()+1
x2_min,x2_max=x2.min()-1,x2.max()+1
axes=plt.gca()
axes.set_xlim([0,1])
axes.set_ylim([0.4,1])
h=0.01
x11,x22=np.meshgrid(np.arange(x1_min,x1_max,h),np.arange(x1_min,x2_max,h))
zg=svcrbf.predict((np.c_[x11.ravel(),x22.ravel()]))
zg=zg.reshape(x11.shape)
plt.contourf(x11,x22,zg,cmap=plt.cm.Spectral)
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(x1.loc[y==0],x2.loc[y==0],c="b",marker="o")
plt.scatter(x1.loc[y==1],x2.loc[y==1],c="k",marker="+")
plt.show()




#SVM with Gaussian Kernel with Validation data
#import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Load Data
from scipy.io import loadmat
df3=loadmat("ex6data3.mat")
x=df3["X"]
y=(df3["y"]).flatten()
xv=df3["Xval"]
yv=(df3["yval"]).flatten()
m=len(y)
mv=len(yv)

#Visualize Data
x1=pd.Series(x[:,0])
x2=pd.Series(x[:,1])
plt.scatter(x1.loc[y==0],x2.loc[y==0],c="y",marker="o")
plt.scatter(x1.loc[y==1],x2.loc[y==1],c="k",marker="+")
plt.show()

#Gaussian RBF Kernel
from sklearn.svm import SVC
svcrbf=SVC(C=1000,kernel="rbf",gamma=5)
xt=np.c_[x1,x2]
svcrbf.fit(xt,y)
z=svcrbf.predict(xt)

#Performance Measuring on Train set
accuracy=np.sum(z==y)*100/(len(y))
print("Accuracy is "+ str(accuracy)+" %")

#Choose best values using grid search
from sklearn.model_selection import GridSearchCV
param_grid=[{'C':np.array([0.01,0.03,0.1,0.3,1,3,10,30]),'gamma':np.array([5000,555.5,50,5.55,0.5,0.055,0.005,0.000556])}]
grid_search = GridSearchCV(svcrbf,param_grid,scoring='accuracy',cv=2,return_train_score=True)
grid_search.fit(xv,yv)
grid_search.best_params_
grid_search.cv_results_


#Performance Measuring on Validation set
from sklearn.svm import SVC
svcrbf=SVC(C=10,kernel="rbf",gamma=5.5)
xt=np.c_[x1,x2]
svcrbf.fit(xt,y)
zv=svcrbf.predict(xv)
accuracy=np.sum(zv==yv)*100/(len(yv))
print("Accuracy is "+ str(accuracy)+" %")


#Visualize Result
x1_min,x1_max=x1.min()-1,x1.max()+1
x2_min,x2_max=x2.min()-1,x2.max()+1
axes=plt.gca()
axes.set_xlim([-0.6,0.3])
axes.set_ylim([-0.8,0.6])
h=0.01
x11,x22=np.meshgrid(np.arange(x1_min,x1_max,h),np.arange(x1_min,x2_max,h))
zg=svcrbf.predict((np.c_[x11.ravel(),x22.ravel()]))
zg=zg.reshape(x11.shape)
plt.contourf(x11,x22,zg,cmap=plt.cm.Spectral)
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(x1.loc[y==0],x2.loc[y==0],c="b",marker="o")
plt.scatter(x1.loc[y==1],x2.loc[y==1],c="k",marker="+")
plt.show()






#Spam Classification
#import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Load Data
from scipy.io import loadmat
df4=loadmat("spamTrain.mat")
df5=loadmat("spamTest.mat")
x=df4["X"]
y=(df4["y"]).flatten()
xt=df5["Xtest"]
yt=(df5["ytest"]).flatten()
m=len(y)
mt=len(yt)


#Gaussian RBF Kernel
from sklearn.svm import SVC
svcrbf2=SVC(C=10,kernel="rbf",gamma=0.005)
svcrbf2.fit(x,y)


#Choose best values using grid search
from sklearn.model_selection import GridSearchCV
param_grid=[{'C':np.array([1,10,100]),'gamma':np.array([0.5,0.05,0.005])}]
grid_search = GridSearchCV(svcrbf2,param_grid,scoring='accuracy',cv=2,return_train_score=True)
grid_search.fit(x,y)
grid_search.best_params_
grid_search.cv_results_


#Performance Measuring on Train set
z=svcrbf2.predict(x)
accuracy=np.sum(z==y)*100/(len(y))
print("Accuracy is "+ str(accuracy)+" %")


#Performance Measuring on Test set
zt=svcrbf2.predict(xt)
accuracy=np.sum(zt==yt)*100/(len(yt))
print("Accuracy is "+ str(accuracy)+" %")
