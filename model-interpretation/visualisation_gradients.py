#Code by Matineh Akhlaghinia
#only the GradientVisualisation class was used in this project, not ClassSpecificImageGeneration.


# coding: utf-8

# In[5]:

import os
import numpy as np
import torch
from torch import nn

from torch.optim import SGD
from torchvision import models
import torch
from torch.nn import ReLU
import os
import copy
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map

import torch
from torch.autograd import Variable
from torchvision import models
from torch import nn
import torch.nn.functional as F


# In[17]:


# In[7]:


class MyInceptionFeatureExtractor(nn.Module):
    def __init__(self, inception, transform_input=False):
        super(MyInceptionFeatureExtractor, self).__init__()
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.Mixed_5b = inception.Mixed_5b
        # stop where you want, copy paste from the model def
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c
        self.fc = nn.Linear(2048, 2)

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[0] = x[0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[1] = x[1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[2] = x[2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 2048
        x = self.fc(x)
        # copy paste from model definition, just stopping where you want
        return x


# In[8]:


# Reference: https://github.com/utkuozbulak/pytorch-cnn-visualizations
def convert_to_grayscale(im_as_arr):
    grayscale_im = np.max(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_gradient_images(gradient, file_name):
    if not os.path.exists('../results'):
        os.makedirs('../results')
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    path_to_file = os.path.join('../results', file_name + '.jpg')
    save_image(gradient, path_to_file)
    
def save_image(image, path):
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        if image.shape[0] == 1:
            image = np.repeat(image, 3, axis=0)
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0)) * 255
        image = Image.fromarray(np.uint8(image), mode='RGB')
    image.save(path)


def preprocess_image(pil_im, resize_im=True):
    mean = [0, 0, 0]
    std = [1, 1, 1]
    im_as_arr =  np.float32(pil_im).transpose(2, 0, 1)
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    im_as_ten = torch.from_numpy(im_as_arr).float().unsqueeze_(0)
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var):
    reverse_mean = [0, 0, 0]
    reverse_std = [1, 1, 1]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)
    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im

# Reference: https://arxiv.org/abs/1312.6034
# Reference: https://ai.googleblog.com/2015/07/deepdream-code-example-for-visualizing.html
class ClassSpecificImageGeneration():
    def __init__(self, model, created_image, target_class):
        self.model = model
        self.model.eval()
        self.target_class = target_class
        self.created_image = created_image
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def generate(self): #THIS IS THE IMPORTANT BIT!!!
        initial_learning_rate = 6.0
        for i in range(1, 351):
            self.processed_image = preprocess_image(self.created_image, False)
            optimizer = torch.optim.Adam([self.processed_image], lr=initial_learning_rate, weight_decay=0.0001)            
            output = self.model(self.processed_image)
            class_loss = -output[0, self.target_class] #PUT IN PRIOR TERM HERE, edit loss
            optimizer.zero_grad()
            self.model.zero_grad()
            class_loss.backward()
            optimizer.step()
            self.created_image = recreate_image(self.processed_image)
            if i % 10 == 0:
                im_path = '../generated/cg-ss2/c_specific_iteration_'+str(i)+'.jpg'
                save_image(self.created_image, im_path)
        return self.processed_image


# In[15]:


# Reference: https://arxiv.org/abs/1312.6034
class GradientVisualisation():
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.model.eval()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        first_layer = list(torch.nn.Sequential(*list(self.model.children()))._modules.items())[0][1].conv
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_class):
        input_image = preprocess_image(input_image)
        model_output = self.model(input_image)
        self.model.zero_grad()
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        model_output.backward(gradient=one_hot_output)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


# In[19]:



if __name__ == '__main__':
    # [0 -> SFT, 1 -> SS]
    target_class = 1
    path_to_model = "/home/sophie/Documents/models/inception2_model.pt"
    visualisation= "gradients"
    image_path = Image.open("/home/sophie/Documents/for_windows/SS.jpeg").convert('RGB')
    random_image = np.uint8(np.random.uniform(0, 255, (512, 512, 3)))
    model_ft = torch.load(path_to_model, map_location='cpu')
    
    if visualisation == "deepdream":
        # Start from an actual image.
        pretrained_model = MyInceptionFeatureExtractor(model_ft)
        csig = ClassSpecificImageGeneration(pretrained_model, image_path, target_class)
        csig.generate()
    elif visualisation == "class-generation":
        # Start from a random noisy image.
        pretrained_model = MyInceptionFeatureExtractor(model_ft)
        csig = ClassSpecificImageGeneration(pretrained_model, random_image, target_class)
        csig.generate()
    elif visualisation == "gradients":
        gd = GradientVisualisation(model_ft)
        gradients = gd.generate_gradients(image_path, target_class)
        grayscale_grads = convert_to_grayscale(gradients)
        save_gradient_images(grayscale_grads,  "gradients")
    

