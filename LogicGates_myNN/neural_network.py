# -*- coding: utf-8 -*-
import math
import torch
torch.set_default_tensor_type(torch.DoubleTensor)


class NeuralNetwork:

    
    def __init__(self, layers: list):

        #initializing dictionaries of parameters 
        self.theta={}
        self.dE_dtheta={}
        self.a ={}
        self.z={}
        self.delta={}
        self.layers = layers
        
        self.num_layers=len(layers)
        for i in range (0,self.num_layers-1):
            self.theta[i]=torch.normal(torch.zeros(self.layers[i]+1,self.layers[i+1]),1/math.sqrt(self.layers[i]))
            
    def getLayer(self, layer):
        #get the theta from dictionary of layers 
        return self.theta[layer]
    
    def forward(self, input: torch.DoubleTensor): 
        self.input = input
        self.a[0]=self.input
        (row,col) = self.a[0].size()
        self.bias=torch.ones((1,col))
        for i in range (0, self.num_layers-1):
            a_hat=torch.cat((self.bias,self.a[i]),0)
            self.z[i+1]=torch.mm(torch.transpose(self.theta[i],0,1),a_hat)
            self.a[i+1]=torch.sigmoid(self.z[i+1])
  
        return self.a[i+1]
        
    def backward(self, target: torch.DoubleTensor):
        
        self.target = target
        self.n = self.num_layers
        (row,no_of_samples) = self.target.size()
        last = self.n -1
        self.delta[last]= (-1/no_of_samples)*(target - self.a[last])*self.a[last]*(1-self.a[last])
        for i in range (self.num_layers-2,-1,-1): 
            self.delta[i] = torch.mm(self.theta[i][1:,:],self.delta[i+1])*self.a[i]*(1-self.a[i])
            self.dE_dtheta[i] = torch.mm(torch.cat((self.bias,self.a[i]),0),torch.t(self.delta[i+1]))
 

    def updateParams(self, eta):

        for i in range (self.n-2,-1,-1):
            self.theta[i] = self.theta[i] - eta * self.dE_dtheta[i]
            

