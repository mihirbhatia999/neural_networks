
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import pandas as pd 
import torchvision.transforms as transforms 
import torch 
import torchvision
import numpy as np
import cv2
import time



#img2obj 
class img2obj():
    
    def train(self):
        device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
      
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.bn0 = nn.BatchNorm2d(6)
                self.pool1 = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.pool2 = nn.MaxPool2d(2, 2)
                self.bn1 = nn.BatchNorm2d(16)
                self.conv3 = nn.Conv2d(16,120,5)
                self.bn2 = nn.BatchNorm2d(120)
                self.h4 = nn.Linear(120,84)
                self.bn3 = nn.BatchNorm1d(84)
                self.h5 = nn.Linear(84,100)
            
            def forward(self,x):
                #forward
                y1 = self.pool1(self.bn0(F.rrelu(self.conv1(x))))
                #print(y1.shape)
                y2 = self.pool2(self.bn1(F.rrelu(self.conv2(y1))))
                #print(y2.shape)
                y3 = self.bn2(F.rrelu(self.conv3(y2)))
                #print(y3.shape)
                y4 = self.bn3(F.rrelu(self.h4(y3.squeeze(3).squeeze(2))))
                #print(y4.shape)
                output_prob = F.softmax(self.h5(y4))
                return output_prob
        try : 
            self.model.load_state_dict(torch.load('model_cifar.pt'))
        except:
            self.model = Net()
        
        self.model.to(device)
        
        #load cifar dataset
        transform = transforms.Compose(
            [transforms.ToTensor(),transforms.Normalize((0.4782,),(0.2682,))
             ])

        trainset = torchvision.datasets.CIFAR100(root='./data_cifar', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=500,
                                                  shuffle=True, num_workers=1)

        testset = torchvision.datasets.CIFAR100(root='./data_cifar', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=500,
                                                 shuffle=False, num_workers=1)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=2e-3)
        
        #initializing list to store all the parameters 
        self.running_loss_test_list = []
        self.running_loss_list = []
        self.total_list = [] 
        self.total_test_list = []
        self.correct_list = []
        self.correct_test_list = []
        self.training_accuracy_list = []
        self.testing_accuracy_list = [] 
        print('total no of epochs')
        epochs = 1
        print(epochs)
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss_test = 0.0
            running_loss = 0.0
            total = 0.0 
            total_test = 0 
            correct = 0
            correct_test = 0 
            for i, data in enumerate(trainloader):
                # get the inputs
                inputs, labels = data
                #print(inputs)
                #print(type(inputs))
                #print(inputs.shape)
                #pilTrans = transforms.ToPILImage()
                #self.pilImg = pilTrans(inputs.squeeze(0))
                #print(self.pilImg)
                #break
                inputs, labels = inputs.to(device),labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                #outputs = net(inputs)
                output = self.model.forward(inputs) 
                
                #print(output.shape)
                _ , predicted = torch.max(output.data,1)
                total = total + labels.size(0)
                correct = correct + (predicted == labels).sum().item()
                #backward
                #print(output)
                #print(labels.shape)
                #print(predicted.shape)                   
                loss = criterion(output, labels)
                #self.loss_list.append(loss.data[0])
                
                #optimizer 
                loss.backward()
                optimizer.step()

                #print statistics
                running_loss += loss.item()
            print('EPOCH NO.', epoch + 1)
            print('training loss  = ',running_loss)
            #print('total train = ',total)
            #print('correct train = ', correct)
            print('train accuracy = ', (correct/total)*100, ' %')
            
            for i, data in enumerate(testloader):
                inputs,labels = data
                inputs, labels = inputs.to(device),labels.to(device)
                output = self.model.forward(inputs)
                _ , predicted = torch.max(output.data,1)
                total_test = total_test + labels.size(0)
                correct_test = correct_test + (predicted == labels).sum().item()
                loss = criterion(output, labels)
                running_loss_test += loss.item()
            print('test loss = ', running_loss_test)
            #print('total test = ',total_test)
            #print('correct test = ', correct_test)
            print('test accuracy = ', (correct_test/total_test)*100, ' %')
            print('-------------------------------- ')
            print(' ')
            
            #adding everything to lists 
            self.running_loss_test_list.append(running_loss_test)
            self.running_loss_list.append(running_loss)
            self.total_list.append(total)
            self.total_test_list.append(total_test)
            self.correct_list.append(correct)
            self.correct_test_list.append(correct_test)
            self.training_accuracy_list.append((correct/total)*100)
            self.testing_accuracy_list.append((correct_test/total_test)*100)
        
                

            
        torch.save(self.model.state_dict(), 'model_cifar.pt')
        print('Finished Training')
           
        
    def forward(self,img):
        m = self.model.eval()
        return m.forward(img)
        
      
    def view(self,img):
    
        CIFAR100_LABELS_LIST = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm']
        
        
        output = self.forward(img.float())
        _ , predicted = torch.max(output,1)
        class_of_detected_object = predicted.item()
        label_to_print = CIFAR100_LABELS_LIST[class_of_detected_object]
        
        #image to print 
        import torchvision.transforms.functional as X
        img = X.to_pil_image(img.squeeze(0))
        img.show()
        
        #print label 
        print(label_to_print)
    
        
        
        
    def cam(self):
        CIFAR100_LABELS_LIST = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm']
        cap = cv2.VideoCapture(0)
        while(True):

            ret, frame = cap.read()

            #sqaure image with side equal to height of rectangle
            image_to_feed = cv2.resize(frame,(32,32),interpolation = cv2.INTER_CUBIC)
            #print(image_to_feed.shape)
            image_to_feed = torch.from_numpy(np.moveaxis(image_to_feed[..., [2, 1, 0]], 2, 0)).type(torch.ByteTensor).unsqueeze(0)
            #image_to_feed = torch.tensor(image_to_feed).view(3,32,32).unsqueeze(0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            output = self.forward(image_to_feed.float())
            _ , predicted = torch.max(output,1)
            class_of_detected_object = predicted.item()
            label_to_print = CIFAR100_LABELS_LIST[class_of_detected_object]
            cv2.putText(frame,label_to_print,(100,100), font, 1,(0,0,255),2,cv2.LINE_AA)

            # Display the resulting frame
            cv2.imshow('frame',frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.25)


        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

obj = img2obj()
obj.train()
obj.cam()



