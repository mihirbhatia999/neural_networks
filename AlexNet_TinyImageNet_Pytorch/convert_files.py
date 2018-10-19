'''
This script opens the tiny image file validation set
and segregates the images. We use this in order to be able
to use pytorch dataloader to for our model during training.

Run this script with the tiny image folder in the same directory
to segregate images in the validation set into their respective classes
'''

import os
from shutil import copyfile


file = open("tiny-imagenet-200/val/val_annotations.txt", "r")
for line in file:
    line = line.split()
    class_folder="tiny-imagenet-200/val/"+line[1]
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)
    source_file = "tiny-imagenet-200/val/images/"+line[0]
    dest_file = class_folder+"/"+line[0]
    try:
        copyfile(source_file, dest_file)
    except IOError as err:
        print("can't copy %s"%err)
        exit(1)




    
    
