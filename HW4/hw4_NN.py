# -*- coding: utf-8 -*-
"""HW4_NN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1r03p5Qwu3tKTKqCD2TaTiN2EtZhMrO2_
"""

#############################################################
## Stat 202A - Homework 4
## Author: Chaojie Feng
## Date : Oct.29, 2018
## Description: This is HW4. You are require to implement a 
## 2-layer neural network in 3 way to classify mnist.
#############################################################

#############################################################

# INSTRUCTIONS: Please fill in the missing lines of code
# only where specified. You only need to fill "# Define it"
# and "FILL IN CODE HERE" area.
#
# You are required to implement neural network in three way:
# 1) Write from scratch, following professor's code. Change output layer to 10 classes
# 2) Write it through tensorflow
# 3) Write it through pyTorch 
# For each function, you are given a dataset and required to output parameters: W1, b1, W2, b2 stand two layer weights and its bias.
# Try to run main_test() to evaluate your code. evaluate() return the accuracy of your parameter. Receive full score if the accuracy is (>80% for scratch version, >95% for tensorflow and pytorch version), receive half score if the accuracy is >20%. Notice on grading stage, the input dataset may changes. 
# Function (1) worth 60% score, (2)(3) worth 30% each. That mean you can select only one function to finish to have 90% score. 

## The structure of network should be: 
## Input data : n * p ( p = 784 )
## Input >> FC >> relu >> FC >> Output
## Output data : n * 10 >> 
## See evaluate function for detail structure.
## 2) solve()
#############################################################

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import torch
import numpy as np

###############################################
## Function 1: Write 2-layer NN from scratch ##
###############################################

def my_NN_scratch(mnist):
    
    # get data and parameters
    X_test = mnist.test.images
    Y_test = mnist.test.labels
    ntest = X_test.shape[0]
    num_output = 10
    num_hidden = 100
    num_iterations = 1000
    learning_rate = 1e-1
    

    #######################
    ## FILL IN CODE HERE ##
    #######################
    
    
    def sigmoid(X):
      return 1/(1+np.exp(-X))
    
    
    def gradSig(X):
      return sigmoid(X)*(1-sigmoid(X))
    
    def ReLU(X):
      return X*(X>0)
    
    def gradReLU(X):
      return X>0
    
    
    # initialize
    n,p = X_test.shape
    alpha = np.random.randn(p+1,num_hidden)
    beta = np.random.randn(num_hidden+1,num_output)

    for it in range(num_iterations):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        #######################
        ## FILL IN CODE HERE ##
        #######################
        # forward propagation
        batch_xs = np.column_stack((np.ones(100),batch_xs))
        layer1 = sigmoid(batch_xs.dot(alpha))
        layer1 = np.column_stack((np.ones(100),layer1))
        layer2 = sigmoid(layer1.dot(beta))
        
        # back propagation
        dbeta = (batch_ys - layer2)*(sigmoid(layer2))
        dalpha = dbeta.dot(beta.T)*(gradReLU(layer1))
        beta = beta + learning_rate*layer1.T.dot(dbeta)
        alpha = alpha + learning_rate*batch_xs.T.dot(dalpha[:,1:])
        
        
        
                           
      

    return alpha[1:, :], alpha[0, :], beta[1:, :], beta[0, :]

#########################################################################
## Function 1.1: Write 2-layer NN from scratch with 2-class classifier ##
#########################################################################

def my_NN_2class(mnist_m):

    X_train = mnist_m.train.images
    Y_train = mnist_m.train.labels
    idx = np.where(Y_train < 2)
    X_train = X_train[idx[:1000]]
    n = X_train.shape[0]
    Y_train = Y_train[idx[:1000]].reshape((n, 1))
    X_test = mnist_m.test.images
    Y_test = mnist_m.test.labels
    idx = np.where(Y_test < 2)
    X_test = X_test[idx]
    ntest = X_test.shape[0]
    Y_test = Y_test[idx].reshape((ntest, 1))

    #######################
    ## FILL IN CODE HERE ##
    #######################

    for it in range(num_iterations):

        pass
    #######################
    ## FILL IN CODE HERE ##
    #######################

    return alpha, beta, acc_train, acc_test

################################################
## Function 2: Write 2-layer NN by Tensorflow ##
################################################
def my_NN_tensorflow(mnist):

    num_hidden = 100
    x = tf.placeholder(tf.float32, [None, 784])
    
    W1 = tf.Variable(tf.random_normal([784,num_hidden], stddev = 0.1))
    b1 = tf.Variable(tf.random_normal([100], stddev = 0.1))
    W2 = tf.Variable(tf.random_normal([num_hidden,10], stddev = 0.1))
    b2 = tf.Variable(tf.random_normal([10], stddev = 0.1))
    z = tf.nn.relu(tf.matmul(x, W1) + b1)
    y = tf.matmul(z, W2) + b2

    y_ = tf.placeholder(tf.float32,shape = [None,10]) # Define it
    cross_entropy =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))# Define it
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    for epoch in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        res = sess.run(train_step, feed_dict={
            x: batch_xs, y_: batch_ys})  # Define it
    print(sess.run(accuracy, feed_dict={
        x: mnist.test.images, y_: mnist.test.labels}))
    W1_e, b1_e, W2_e, b2_e = W1.eval(), b1.eval(), W2.eval(), b2.eval()
    sess.close()

    return W1_e, b1_e, W2_e, b2_e

###############################################
## Function 3: Write 2-layer NN by pyTorch   ##
###############################################
def my_NN_pytorch(mnist_m):

    class Net(torch.nn.Module):
        def __init__(self):
          super(Net, self).__init__()
          self.fc1 =  torch.nn.Linear(784,100)# Define it
          self.fc2 =  torch.nn.Linear(100,10)# Define it

        def forward(self, x):
          x = torch.nn.functional.relu(self.fc1(x))# Define it
          x = self.fc2(x)# Define it
          return x

    net = Net()
    net.zero_grad()
    Loss = torch.nn.CrossEntropyLoss() # Define it
    optimizer = torch.optim.SGD(net.parameters(),lr=0.1,momentum=0.9) # Define it

    for epoch in range(1000):  # loop over the dataset multiple times

        batch_xs, batch_ys = mnist_m.train.next_batch(100)
        #######################
        ## FILL IN CODE HERE ##
        #######################
        bx = torch.from_numpy(batch_xs)
        by = torch.from_numpy(batch_ys).long()
        optimizer.zero_grad()
        out = net(bx)
        loss = Loss(out,by)
        loss.backward()
        optimizer.step()
        

    params = list(net.parameters())
    return params[0].detach().numpy().T, params[1].detach().numpy(), \
        params[2].detach().numpy().T, params[3].detach().numpy()
  
  
main_test()

def evaluate(W1, b1, W2, b2, data):

    inputs = data.test.images
    outputs = np.dot(np.maximum(np.dot(inputs, W1) + b1, 0), W2) + b2
    predicted = np.argmax(outputs, axis=1)
    accuracy = np.sum(predicted == data.test.labels)*100 / outputs.shape[0]
    print('Accuracy of the network on test images: %.f %%' % accuracy)
    return accuracy

def main_test():

    mnist = input_data.read_data_sets('input_data', one_hot=True)
    mnist_m = input_data.read_data_sets('input_data', one_hot=False)
    W1, b1, W2, b2 = my_NN_scratch(mnist)
    evaluate(W1, b1, W2, b2, mnist_m)
    W1, b1, W2, b2 = my_NN_tensorflow(mnist)
    evaluate(W1, b1, W2, b2, mnist_m)
    W1, b1, W2, b2 = my_NN_pytorch(mnist_m)
    evaluate(W1, b1, W2, b2, mnist_m)



