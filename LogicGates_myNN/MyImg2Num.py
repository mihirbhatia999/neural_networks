
# coding: utf-8

# In[5]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from neural_network import *


class MyImg2Num:
    mnist_in_size , h_1, h_2, h_3, mnist_out_size = 784, 500, 500, 500, 10
    mnist_model = NeuralNetwork([mnist_in_size ,h_1, h_2, h_3, mnist_out_size])
    
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
                
        #define the model 
        #mnist_in_size , h_1, h_2, h_3, mnist_out_size = 784, 500, 500, 500, 10
        #self.mnist_model = NeuralNetwork([mnist_in_size ,h_1, h_2, h_3, mnist_out_size])
        self.learning_rate = 0.001
        self.n_epochs = 20

        
        self.accuracy = 0 
        self.epoch_loss_list = []
        self.epoch_loss_list_test = []
        
        for i in range(self.n_epochs):
            self.epoch_loss = 0 
            self.epoch_loss_test = 0 
            self.correct = 0 
            self.wrong = 0 
            
            for batch_id, (data, label) in enumerate(self.train_loader):
                data = Variable(data)
                target = Variable(label)
                
                #TRANSFORMING
                #final_data = data.squeeze(1)
                #final_data = final_data.numpy()
                #final_data = final_data.reshape((500,28*28))
                #final_data = torch.tensor(final_data)
                
                final_data = torch.t(data.squeeze(1).view(500, 28*28))
                final_target = one_hot(target,10)

                
                #train the model 
                self.batch_result = self.mnist_model.forward(final_data.double())
                self.mnist_model.backward(final_target.double())
                self.mnist_model.updateParams(self.learning_rate)
                
                batch_loss = torch.pow((self.batch_result - final_target) ,2).sum()
                self.epoch_loss = self.epoch_loss + batch_loss 
                #print(self.batch_result)
                value , self.result_for_accuracy = self.batch_result.max(0)
                #print(result_for_accuracy)
                accuracy = self.result_for_accuracy - target 
                
                for p in range(accuracy.shape[0]):
                    if accuracy[p] == 0:
                        self.correct = self.correct + 1 
                    else:
                        self.wrong = self.wrong + 1   
            
            self.accuracy_to_report = (self.correct/(self.correct + self.wrong))*100
            print('accuracy from epoch no. ' + str(i+1) + '=' + str(self.accuracy_to_report) + ' %')
            print('training loss in epoch no. ' + str(i+1) + ' = ' + str(self.epoch_loss/60000))
            self.epoch_loss_list.append(self.epoch_loss/(60000))
            
            
            
            for batch_id_test, (data_test,label_test) in enumerate(self.test_loader):
                test_data = Variable(data_test)
                test_target = Variable(label)
                
                final_data_test = test_data.squeeze(1).view(28*28,500)
                
                final_target_test = one_hot(test_target,10)
                
                self.batch_result_test = self.mnist_model.forward(final_data_test.double())
                batch_loss_test = torch.pow((self.batch_result - final_target_test),2).sum()
                self.epoch_loss_test = self.epoch_loss_test + batch_loss_test 
                
            print('test loss in epoch no. ' + str(i+1) + ' = ' + str(self.epoch_loss_test/10000))
            self.epoch_loss_list_test.append(self.epoch_loss/10000)
                
                
          
    def forward(self, img):
        #predict value
        #img = m * n
        img = torch.t(img.double())
        output = torch.t(self.mnist_model.forward(img))
        return output 
        
    


# In[6]:


#training set error -----------------------------
import time
tic = time.time()
mnist_trainer = MyImg2Num()
mnist_trainer.train()
toc = time.time()

print("time taken  =" + str(toc - tic))


# In[4]:


#testing forward 
mnist_trainer = MyImg2Num()
img = torch.randn(100,784)
mnist_trainer.forward(img)


# In[10]:


import matplotlib.pyplot as plt
plt.plot(mnist_trainer.epoch_loss_list)
plt.ylabel('Loss')
plt.xlabel('epochs')
plt.title('training Loss vs Epoch')
plt.savefig('traininglossvsepoch.png')


# In[13]:


import matplotlib.pyplot as plt
plt.plot(mnist_trainer.epoch_loss_list_test)
plt.ylabel('Loss')
plt.xlabel('epochs')
plt.title('test Loss vs Epoch')
plt.savefig('testlossvsepoch.png')

