#Single Variable Gradient Descent
#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Assign Variables
d1=pd.read_csv('ex1data1.txt',header=None)
x=np.array(d1[0])
y=np.array(d1[1])
m=len(x)

#Visualize Variables
plt.scatter(x,y)
plt.show()

#Apply Gradient Descent and Calculate Cost
x0=np.ones(m)
x1=np.c_[x0,x]
thetas=np.array([0,0])
h=np.dot(x1,thetas)
thetas=thetas-(0.01/m)*np.dot((h-y),x1)
cost=(np.sum((h-y)**2))/(2*m)
cost1=[]
for i in range(1500):
 thetas=thetas-(0.01/m)*np.dot((h-y),x1)
 h = np.dot(x1,thetas)
 cost1.append((np.sum((h-y)**2))/(2*m))


#Visulize Result
plt.scatter(x,y,marker="o")
plt.plot(x,h)
plt.show()



#Mutiple Variable Linear Regression
#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Assign Variables
d2=pd.read_csv('ex1data2.txt',header=None)
x1=np.array(d2[0])
x2=np.array(d2[1])
y=np.array(d2[2])
m=len(x1)

#Visualize Variables
plt.scatter(x1,y)
plt.scatter(x2,y)
plt.show()

#Feature Normalization
x1_m=np.mean(x1)
x1_s=np.std(x1)
x2_m=np.mean(x2)
x2_s=np.std(x2)
y_m=np.mean(y)
y_s=np.std(y)
x1=(x1-x1_m)/x1_s
x2=(x2-x2_m)/x2_s
y=(y-y_m)/y_s

#Apply Gradient Descent and Calculate Cost
x0=np.ones(m)
xt=np.c_[x0,x1,x2]
thetas=np.array([0,0,0])
h=np.dot(xt,thetas)
thetas=thetas-(0.1/m)*np.dot((h-y),xt)
cost=(np.sum((h-y)**2))/(2*m)
cost1=[]
l=[]
for i in range(15000):
 thetas=thetas-(0.01/m)*np.dot((h-y),xt)
 h = np.dot(xt,thetas)
 cost1.append((np.sum((h-y)**2))/(2*m))
 l.append(i)

#Visualize Result
plt.plot(l,cost1)
plt.show()
plt.scatter(x1,y,marker="o")
plt.scatter(x2,y,marker="o")
plt.plot(x1,h)
plt.plot(x2,h)
plt.show()
