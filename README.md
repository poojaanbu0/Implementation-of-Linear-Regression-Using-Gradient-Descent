# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the program.
2.import numpy as np.
3.Give the header to the data.
4.Find the profit of population.
5.Plot the required graph for both for Gradient Descent Graph and Prediction Graph.
6.End the program.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: POOJA A
RegisterNumber: 212222240072
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1.txt", header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Popuation of city (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  m=len(y)
  h=X.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
  
  data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(X,y,theta))
  return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0]for y in x_value]
plt.plot(x_value,y_value,color="purple")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions = np.dot(theta.transpose(),x)
    return predictions[0]
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000 , we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000 , we predict a profit of $"+str(round(predict2,0)))
*/
```

## Output:
## COMPUTE COST VALUE
![image](https://github.com/poojaanbu0/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119390329/9f8dd0d9-71cb-438d-b172-8426560068e7)

## H(X) VALUE
![image](https://github.com/poojaanbu0/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119390329/7825cab3-87cc-4e17-8607-b850d603fb6c)

## COST FUNCTION USING GRADIENT DESCENT GRAPH
![image](https://github.com/poojaanbu0/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119390329/3717567e-e326-4f5e-8a4c-7c540b85804b)

## PROFIT PREDICTION GRAPH
![image](https://github.com/poojaanbu0/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119390329/cbe25494-bf70-4d66-b988-c43bc15a64ba)

## PROFIT FOR THE POPULATION 35,000
![image](https://github.com/poojaanbu0/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119390329/532c211c-1f2a-444a-9388-2bead7339c02)

## PROFIT FOR THE POPULATION 70,000
![image](https://github.com/poojaanbu0/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119390329/68308261-7103-449c-9b74-894a00169c44)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
