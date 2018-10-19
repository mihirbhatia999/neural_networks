
import sys,os
import torch
from train import AlexNet

my_alex = AlexNet()
directory_model = sys.argv[2]

if os.path.exists(directory_model):
    my_alex.load_state_dict(torch.load(directory_model,
                                       map_location=lambda storage, loc: storage))
    print("model loaded")
    
else:
    print("No model exists")
my_alex.cam(my_alex)









