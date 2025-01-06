import numpy as np
from sklearn.datasets import make_blobs
# ref: https://www.kaggle.com/code/fareselmenshawii/logistic-regression-from-scratch/notebook
class LogisticRegression():

    def __init__(self,X,learning_rate=0.1,num_iters=10000,convergence_tol=1e-10):
        self.lr=learning_rate
        self.num_iters=num_iters
        self.convergence_tol=convergence_tol

        self.m,self.n=X.shape #(number of rows,num of feature)


    def train(self,X,y):
        self.weights=np.zeros((self.n,1))
        self.bias=0
        costs=[]
        
        for iteration in range(self.num_iters + 1):
            fwb=self.sigmoid(np.dot(X,self.weights)+self.bias)
            #binary cross entropy loss function
            cost=(-1/self.m)* np.sum(y*np.log(fwb)+(1-y)*np.log(1-fwb))
            

            #back propagation
            db=(1/self.m)*np.sum(fwb-y)
            
            #print((fwb-y).shape)
            #print(X.shape)
            dw=(1/self.m)*(np.dot(X.T,(fwb-y)))

            self.weights-=(self.lr*dw)
            self.bias-=(self.lr*db)

            if iteration%1000==0:
                print(f'iteration: {iteration} | BCE cost: {cost}')

            if iteration>0 and abs(costs[-1]-cost)<self.convergence_tol:
                print(f'converged after {iter} iterations')
                break
            costs.append(cost)

        return self.weights,self.bias

    def predict(self,X):
        y_pred=self.sigmoid(np.dot(X,self.weights)+self.bias)
        
        y_pred_labels=y_pred>0.5
        return y_pred_labels

    def sigmoid(self,z):
        #z here is the linear regression fwd function
        return 1/(1+np.exp(-z))


X,y=make_blobs(1000,centers=2)
y=y[:,np.newaxis]
print('X.Shape')
print(X.shape)
print('Y.Shape')
print(y.shape)

logreg=LogisticRegression(X)
w,b=logreg.train(X,y)

y_predict=logreg.predict(X)

print(f'Accuracy: {np.sum(y==y_predict)/X.shape[0]}')