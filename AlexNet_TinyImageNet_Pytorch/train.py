#importing packages----------------------------------------------------------
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import torch.optim as optim
import torchvision.models as models
import cv2
import sys
import time
import torch
import numpy as np
import json


#we get pretrained model of alex net using the models package---------------
ref_alex = models.alexnet(pretrained=True)  
ref_alex.eval()


#defining alexnet class----------------------------------------------------
class AlexNet(nn.Module):

    #model architecture------- 
    def __init__(self, num_classes=200):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),                                
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        
    #forward propagation------
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),256*6*6)
        x = self.classifier(x)
        return x       


    #camera function to use webcam----
    def cam(self, alex):
        cap = cv2.VideoCapture(0)
        
        while(True):
            time.sleep(0.25)
            ret, frame = cap.read()
            cropped_image = cv2.resize(frame,(224,224), interpolation = cv2.INTER_CUBIC)
            tensor_image = torch.from_numpy(np.moveaxis(cropped_image[..., [2, 1, 0]], 2, 0)).type(torch.ByteTensor).unsqueeze(0)
            
            pred = np.argmax((alex.forward(tensor_image.float())).detach())
            predicted = pred.item()
            class_idx = json.load(open("imagenet_class_index.json"))
            class_label = class_idx[str(predicted)]
            
            print(class_label[1])
            cv2.putText(frame, class_label[1], (100, 180), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), lineType=cv2.LINE_AA) 
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        
    #training the neural network using the architecture---------     
    def train(self, alex, img_dir, save_dir, epochs=1):
        data_dir = img_dir
        input_shape = 224  
        batch_size = 128
        mean = [0.485, 0.456, 0.406]  
        std = [0.229, 0.224, 0.225] 
        scale = 256 
    
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(input_shape),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(scale),
                transforms.CenterCrop(input_shape),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ]),
        }
        
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                      data_transforms[x]) for x in ['train', 'val']}
        
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                         shuffle=True) for x in ['train', 'val']}
        device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        
        alex.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(alex.parameters(), lr=0.0001)
        if os.path.exists(save_dir):
            alex.load_state_dict(torch.load(save_dir))
            print("Loading saved model ... ")

        print('no model saved...creating new model...')
        for epoch in range(epochs):
            train_tot = 0
            train_correct = 0
            train_loss_local = 0
            test_tot = 0
            test_correct = 0
            test_loss_local = 0

            tic = time.time()

            #looping over training set-------
            for i, data in enumerate(dataloaders['train']):
                print(1)
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                output = alex.forward(inputs) 
                _, predicted = torch.max(output.data,1)
                train_tot += labels.size(0)
                correct_train += (predicted == labels).sum().item()                
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                train_loss_local += loss.item()

            
            toc = time.time()

            #training stats
            epoch_number_print = epoch + 1 
            print('epoch no : ' + str(epoch_number_print))
            print('training loss  = ' + str(train_loss_local))
            print("time :" + str( toc-tic))
            train_acc = (correct_train/train_tot)*100
            print('training accuracy = ' + str( train_acc )  + "%")

            #looping over validation set--------
            for i, data in enumerate(dataloaders['val']):
                print(1)
                inputs,labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                output = alex.forward(inputs)
                _, predicted = torch.max(output.data,1)
                test_tot += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                loss = criterion(output, labels)
                test_loss_local += loss.item()
                
            #validation stats
            toc = time.time()
            test_accuracy = 100*(test_correct/test_total)
            print('test accuracy= ' + str(test_accuracy) + "%") 
            print('test loss= ' + str(test_loss_local) +  "time taken:" + str( toc-tic ))
                          
        torch.save(alex.state_dict(), save_dir)
        print('TRAINING COMPLETE !')


#--------------------------------------------------------------------------------------------------------


my_alex = AlexNet()   

# Copying parameters---------
for i, j in zip(ref_alex.modules(), my_alex.modules()):
    if not list(i.children()):
        if isinstance(i, nn.Linear): 
            if (j.out_features is not 200):
                j.weight.data = i.weight.data
                j.bias.data = i.bias.data
                
        elif len(i.state_dict()) > 0:  
            j.weight.data = i.weight.data
            j.bias.data = i.bias.data 

# Freezing all except last layer---------- 
layer = 1
freeze_upto = 15

for i, param in my_alex.named_parameters():
    if layer < freeze_upto:
        layer +=1
        param.requires_grad=False


#to check no of arguments passed-----------
if(len(sys.argv) < 5):
    print("if model ready then start camera")
else:
    image_dir = sys.argv[2]
    model_dir = sys.argv[4]

    my_alex.train(my_alex,image_dir, model_dir,1)

