"""
Created on Thu Oct 26 14:19:44 2017

Edited code, originally from github.com/utkuozbulak
"""
import os
import numpy as np
#from PIL import Image

from torch.optim import SGD
#from torchvision import models
import torch

from misc_functions_prior2 import preprocess_image, recreate_image, save_image


class ClassSpecificImageGeneration():
    """
        Produces an image that maximizes a certain class with gradient ascent
    """
    def __init__(self, model, target_class):
        self.mean = [-0.55, -0.6, -0.45] 
        self.std = [1/0.229, 1/0.224, 1/0.225]
        self.model = model
        self.model.eval()
        self.target_class = target_class
        # Generate a random image
        self.created_image = np.uint8(np.random.uniform(115, 255, (299, 299, 3)))
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def generate(self):
        initial_learning_rate = 6
        for i in range(1, 101):
            # Process image and return variable
            self.processed_image = preprocess_image(self.created_image, False)
            #print(self.processed_image.shape)
            dist = 0
            G_channel = self.processed_image[0,1]
            dist += np.linalg.norm(G_channel.detach().numpy())
            optimizer = SGD([self.processed_image], lr=initial_learning_rate)
            # Forward
            output = self.model(self.processed_image)
            print(output)
            # Target specific class
            class_loss = -output[0, self.target_class] + 3*dist
            print('Iteration:', str(i), 'Loss', "{0:.2f}".format(class_loss.data.numpy()))
            # Zero grads
            self.model.zero_grad()
            # Backward
            class_loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image)
            if i % 10 == 0:
                # Save image
                im_path = '../generated/c_specific_iteration_'+str(i)+'.jpg'
                save_image(self.created_image, im_path)
        return self.processed_image


if __name__ == '__main__':
    target_class = 0  #SFT=0 SS=1
    path_to_model = "/home/sophie/Documents/models/inception2_model.pt"
    pretrained_model = model_ft = torch.load(path_to_model, map_location='cpu')
    csig = ClassSpecificImageGeneration(pretrained_model, target_class)
    csig.generate()
