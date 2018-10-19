
# coding: utf-8

# In[35]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from neural_network import *
torch.set_default_tensor_type(torch.DoubleTensor)

class NnImg2Num:
    
    mnist_in_size , h_1, h_2, h_3, mnist_out_size = 784, 500, 500, 500, 10

    model = torch.nn.Sequential(
            torch.nn.Linear(mnist_in_size, h_1),
            torch.nn.Sigmoid(),
            torch.nn.Linear(h_1, h_2),
            torch.nn.Sigmoid(),
            torch.nn.Linear(h_2,h_3),
            torch.nn.Sigmoid(),
            torch.nn.Linear(h_3,mnist_out_size)
                )
    
    def train(self):

        #get the MNIST dataset and convert to torch tensor for input 

        self.train_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data', 
                                                                  download=True,train=True,
                                                                  transform=transforms.Compose([
                                                                transforms.ToTensor(),
                                                                transforms.Normalize((0.1307,), (0.3081,))
                                                                  ])), batch_size=500, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data', 
                                                                  download=True, train=False,
                                                                 transform=transforms.Compose([
                                                                    transforms.ToTensor(), 
                                                                      transforms.Normalize((0.1307,), (0.3081,)) 
                                                                  ])), batch_size=500, shuffle=True)
        
        
        def one_hot(target, n_classes):
            row = n_classes
            col = target.size(0)
            one_hot = torch.zeros(row,col)

            for i in range(target.size(0)):
                n = target[i]
                one_hot[n,i] = 1.0
            return one_hot
        
        


        loss_function = torch.nn.MSELoss(size_average=False)
        
        optimizer_function = optim.SGD(self.model.parameters(),lr=0.001)
        self.loss_list = []
        self.loss_list_test = []
         
        #for training set --------------------------------------------------
        for t in range(20):
            self.loss_total = 0
            self.loss_total_test = 0 
            print("epoch = ", t+1)
            for batch_id, (data, label) in enumerate(self.train_loader):
                data = Variable(data)
                target = Variable(label)
                
                #transforming data  
                x = data.squeeze(1).view(500,28*28).double()
                y = torch.t(one_hot(target,10)).double()
                
                y_pred = self.model(x)
                self.loss = loss_function(y_pred, y)
                self.loss_total = self.loss_total + self.loss
                
                self.model.zero_grad()
                self.loss.backward()
                
                optimizer_function.step()
                
            self.loss_list.append(self.loss_total.item())
                        
            #checking the test set loss -------------------------------------
            for batch_id, (data, label) in enumerate(self.test_loader):
                test_data = Variable(data)
                test_target = Variable(label)
                
                #transforming data  
                test_x = test_data.squeeze(1).view(500,28*28).double()
                test_y = torch.t(one_hot(test_target,10)).double()
                
                y_pred_test = self.model(x)
                self.loss_test = loss_function(y_pred_test, test_y)
                self.loss_total_test = self.loss_total + self.loss_test       
            
            print("training loss = ", self.loss_total_test)
            self.loss_list_test.append(self.loss_total_test.item())
        
                            
    def forward(self, img):
        #img = m*n  m=samples n=vector size
        #to return c*m where c = classes 
        return self.model(img.double())
        
    


# In[36]:


import time
tic = time.time()
pytorch = NnImg2Num()
pytorch.train()
toc = time.time()
print("time taken for pytorch = ", toc-tic)


# In[41]:


import matplotlib.pyplot as plt

plt.plot([i*(1/60000) for i in pytorch.loss_list])

plt.ylabel('Loss')
plt.xlabel('epochs')
plt.title(' Loss vs Epoch')
plt.legend(['training'], loc='upper right')
plt.savefig('pytorch_lossvsepoch.png')


# In[6]:


#testing forward function ----------------------
pytorch = NnImg2Num()
x = pytorch.forward(torch.randn(100,784)).size()
print(x)


# In[39]:


import matplotlib.pyplot as plt

plt.plot([i*(1/10000) for i in pytorch.loss_list_test])

plt.ylabel('Loss')
plt.xlabel('epochs')
plt.title(' Loss vs Epoch')
plt.legend(['test'], loc='upper right')
plt.savefig('pytorch_lossvsepoch_test.png')



