
import sys
sys.path.append('util')

import os
import time
import math
import numpy as np
import mnist_loader
from sklearn.preprocessing import OneHotEncoder


class NN:
    def __init__(self, file_name, retrain = False):
        self.file_name = file_name
        self.retrain = retrain


        #useful values
        # m,n=np.shape(X)
        self.K = 10
        self.num_of_hidden = 1
        self.init_epsilon = .12
        self.num_of_neurons = 25

        self.n_of_features = 784

        #hyperparameters
        self.lmbda = .001
        self.alpha = .001

        #initialization
        self.theta={}
        self.a={}
        self.z={}

        if retrain or not self.load():
            self.__initialize_Theta(1, self.n_of_features, self.num_of_neurons)
            self.__initialize_Theta(2, self.num_of_neurons, 10)

        np.random.seed(0)

    def load(self):
        if os.path.exists(self.file_name):
            thetas_trained = np.load(self.file_name, allow_pickle = True)
            self.theta = {1:thetas_trained.item().get(1),2:thetas_trained.item().get(2)}
            return True
        return False

    def save(self):
        np.save(self.file_name, self.theta)
        return True

    def get_test(self):
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper('db/')
        test_data = np.array(list(test_data))
        X_test=np.reshape(np.concatenate(test_data[:,0]),(10000,28,28))
        return X_test

    def predict(self, x):
        x = x / 255.0
        x = np.reshape(x, (784))
        x = np.insert(x,0,1,axis=0)
        h1 = self.__sigmoid(np.matmul(x, self.theta[1].T))
        h1 = np.insert(h1,0,1,axis=0)
        h2 = self.__sigmoid(np.matmul(h1, self.theta[2].T))
        return np.argmax(h2), h2

    def train(self, epochs = 50, alpha = 0.001, lmbda = 0.001, batch_size=10):
        print('Training...')
    
        start = time.time()

        training_data, validation_data, test_data = mnist_loader.load_data_wrapper('db/')

        training_data = np.array(list(training_data))
        test_data=np.array(list(test_data))

        X = np.reshape(np.concatenate(training_data[:,0]),(50000,784))
        y_matrix=np.reshape(np.concatenate(training_data[:,1]),(50000,10))

        X_test=np.reshape(np.concatenate(test_data[:,0]),(10000,784))
        y_test=np.reshape(np.concatenate(np.array([test_data[:,1]]).T),(10000,1))

        encoder = OneHotEncoder(sparse=False)
        y_test = encoder.fit_transform(y_test)

        Xc = X

        acc = 0

        for j in range(epochs):
            for i in range( int( ( np.size(X,0) / batch_size ) ) ):
                J, Theta1_grad, Theta2_grad = self.__feedForward(Xc[(batch_size*i):(batch_size*i+batch_size)],self.theta,y_matrix[(batch_size*i):(batch_size*i+batch_size)],lmbda)
                self.theta[1] = self.theta[1]-alpha*Theta1_grad
                self.theta[2] = self.theta[2]-alpha*Theta2_grad
            J_total = self.__computeCost(Xc,self.theta,y_matrix,lmbda)
            J_test = self.__computeCost(X_test,self.theta,y_test,lmbda)
            print(f'Epoch: {j+1}, alpha: {alpha}, lmbda: {lmbda}, J_batch = : {J}, J_total={J_total}, J_test={J_test}, J_diff={(J_total-J_test)}')
            acc = self.__accuracy(X, self.theta, y_test=y_matrix)
            print(f'Training accuracy is {acc}')
            print(f'Test accuracy is {self.__accuracy(X_test, self.theta, y_test)}')
            
        end = time.time()
        elapsed = end - start

        return acc, elapsed

    def __computeCost(self, X,thetaC,y_matrix,lmbda):
        self.a[1] = np.insert(X,0,1,axis=1)
        self.z[2] = np.matmul(self.a[1],thetaC[1].T)
        self.a[2] = np.insert(self.__sigmoid(self.z[2]),0,1,axis=1)
        self.z[3] = np.matmul(self.a[2],(thetaC[2].T))
        self.a[3] = self.__sigmoid(self.z[3])
        regCost = (lmbda/(2*len(X)))*(np.sum(thetaC[1][:,1:(self.n_of_features+1)])**2 + np.sum(thetaC[2][:,1:(self.num_of_neurons+1)])**2)
        J=(1/len(X))*np.sum((-y_matrix*np.log(self.a[3])-(1-y_matrix)*np.log(1-self.a[3]))) + regCost
        return J

    def __accuracy(self, X, thetaC, y_test):
        count = 0
        badCount=[]
        for i,x in enumerate(X):
            output = self.__predict(x,thetaC)
            if np.argmax(output) == np.argmax(y_test[i]):
                count += 1
            else:
                badCount.append(i)
        return count/len(X) 

    def __predict(self, x, thetaC):
        x = np.insert(x,0,1,axis=0)
        h1 = self.__sigmoid(np.matmul(x,thetaC[1].T))
        h1 = np.insert(h1,0,1,axis=0)
        h2 = self.__sigmoid(np.matmul(h1,thetaC[2].T))
        return h2
    
    # takes numpy array
    def __sigmoid(self, z): 
        s = 1 / ( 1 + np.exp(-z) )
        return s

    def __sigmoidGrad(self, z):
        g = self.__sigmoid(z) * ( 1 - self.__sigmoid(z) )
        return g

    def __initialize_Theta(self, lNum, l1_size, l2_size):
        self.theta[lNum] = np.random.random( (l2_size, (l1_size+1) ) ) * ( 2*self.init_epsilon ) - self.init_epsilon

    def __feedForward(self, X, thetaC, y_matrix, lmbda):
        self.a[1] = np.insert(X, 0, 1, axis=1)

        self.z[2] = np.matmul(self.a[1], thetaC[1].T)
        
        self.a[2] = np.insert(self.__sigmoid(self.z[2]), 0, 1, axis=1)
        
        self.z[3] = np.matmul(self.a[2], (thetaC[2].T))
        
        self.a[3] = self.__sigmoid(self.z[3])
        
        regCost = (lmbda / ( 2 * len(X))) * ( np.sum( thetaC[1][:, 1:(self.n_of_features+1) ] ) ** 2 + np.sum(thetaC[2][:, 1:(self.num_of_neurons+1)]) ** 2)

        J= (1/len(X)) * np.sum( (-y_matrix * np.log(self.a[3]) - (1-y_matrix) * np.log(1 - self.a[3]))) + regCost
        
        #Back Propagation
        d3 = self.a[3] - y_matrix
        
        d2 = np.matmul(d3, thetaC[2][:,1:(self.num_of_neurons+1)]) * self.__sigmoidGrad(self.z[2])
        
        Delta1 = np.matmul(d2.T, self.a[1])
        Delta2 = np.matmul(d3.T, self.a[2])
        #unregularized gradient
        Theta1_grad = (1/len(X)) * Delta1
        Theta2_grad = (1/len(X)) * Delta2
        
        #regularized gradient
        self.theta[1][:,0] = 0
        self.theta[2][:,0] = 0
        
        Theta1_grad = Theta1_grad + thetaC[1] * (lmbda/len(X))
        Theta2_grad = Theta2_grad + thetaC[2] * (lmbda/len(X))
        
        
        return J,Theta1_grad,Theta2_grad
    
    