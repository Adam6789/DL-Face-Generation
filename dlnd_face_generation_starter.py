#!/usr/bin/env python
# coding: utf-8

# # Face Generation
# 
# In this project, you'll define and train a Generative Adverserial network of your own creation on a dataset of faces. Your goal is to get a generator network to generate *new* images of faces that look as realistic as possible!
# 
# The project will be broken down into a series of tasks from **defining new architectures training adversarial networks**. At the end of the notebook, you'll be able to visualize the results of your trained Generator to see how it performs; your generated samples should look like fairly realistic faces with small amounts of noise.
# 
# ### Get the Data
# 
# You'll be using the [CelebFaces Attributes Dataset (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to train your adversarial networks.
# 
# This dataset has higher resolution images than datasets you have previously worked with (like MNIST or SVHN) you've been working with, and so, you should prepare to define deeper networks and train them for a longer time to get good results. It is suggested that you utilize a GPU for training.
# 
# ### Pre-processed Data
# 
# Since the project's main focus is on building the GANs, we've done *some* of the pre-processing for you. Each of the CelebA images has been cropped to remove parts of the image that don't include a face, then resized down to 64x64x3 NumPy images. Some sample data is show below.
# 
# <img src='assets/processed_face_data.png' width=60% />
# 
# > If you are working locally, you can download this data [by clicking here](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5be7eb6f_processed-celeba-small/processed-celeba-small.zip)
# 
# This is a zip file that you'll need to extract in the home directory of this notebook for further loading and processing. After extracting the data, you should be left with a directory of data `processed-celeba-small/`.

# In[3]:


# run this once to unzip the file
#!unzip processed-celeba-small.zip


# In[5]:


from pathlib import Path
import matplotlib.image as mpimg

from glob import glob
from typing import Tuple, Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision.transforms import Normalize
import torch.nn as nn

#import tests


# In[6]:


data_dir = 'processed_celeba_small/celeba/'


# ## Data pipeline
# 
# The [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset contains over 200,000 celebrity images with annotations. Since you're going to be generating faces, you won't need the annotations, you'll only need the images. Note that these are color images with [3 color channels (RGB)](https://en.wikipedia.org/wiki/Channel_(digital_image)#RGB_Images) each.
# 
# ### Pre-process and Load the Data
# 
# Since the project's main focus is on building the GANs, we've done *some* of the pre-processing for you. Each of the CelebA images has been cropped to remove parts of the image that don't include a face, then resized down to 64x64x3 NumPy images. This *pre-processed* dataset is a smaller subset of the very large CelebA dataset and contains roughly 30,000 images. 
# 
# Your first task consists in building the dataloader. To do so, you need to do the following:
# * implement the get_transforms function
# * create a custom Dataset class that reads the CelebA data

# ### Exercise: implement the get_transforms function
# 
# The `get_transforms` function should output a [`torchvision.transforms.Compose`](https://pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html#torchvision.transforms.Compose) of different transformations. You have two constraints:
# * the function takes a tuple of size as input and should **resize the images** to the input size
# * the output images should have values **ranging from -1 to 1**

# In[7]:


def get_transforms(size: Tuple[int, int]) -> Callable:
    """ Transforms to apply to the image."""
    # TODO: edit this function by appening transforms to the below list
    transforms = [ToTensor(), Resize(size), Normalize(0.5,0.5)]
    
    return Compose(transforms)


# ### Exercise: implement the DatasetDirectory class
# 
# 
# The `DatasetDirectory` class is a torch Dataset that reads from the above data directory. The `__getitem__` method should output a transformed tensor and the `__len__` method should output the number of files in our dataset. You can look at [this custom dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files) for ideas. 

# In[8]:



class DatasetDirectory(Dataset):
    """
    A custom dataset class that loads images from folder.
    args:
    - directory: location of the images
    - transform: transform function to apply to the images
    - extension: file format
    """
    def __init__(self, 
                 directory: str, 
                 transforms: Callable = None, 
                 extension: str = '.jpg'):
        # TODO: implement the init method

        images = list(Path(directory).glob(f"*{extension}"))
        self.images = [transforms(mpimg.imread(str(p))) for p in images]
        
    def __len__(self) -> int:
        """ returns the number of items in the dataset """
        # TODO: return the number of elements in the dataset
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        """ load an image and apply transformation """
        # TODO: return the index-element of the dataset
        return self.images[index]


# In[9]:


"""
DO NOT MODIFY ANYTHING IN THIS CELL
"""
# run this cell to verify your dataset implementation
dataset = DatasetDirectory(data_dir, get_transforms((64, 64)))
#tests.check_dataset_outputs(dataset)


# The functions below will help you visualize images from the dataset.

# In[ ]:


"""
DO NOT MODIFY ANYTHING IN THIS CELL
"""

def denormalize(images):
    """Transform images from [-1.0, 1.0] to [0, 255] and cast them to uint8."""
    return ((images + 1.) / 2. * 255).astype(np.uint8)

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(20, 4))
plot_size=20
for idx in np.arange(plot_size):
    ax = fig.add_subplot(2, int(plot_size/2), idx+1, xticks=[], yticks=[])
    img = dataset[idx].numpy()
    img = np.transpose(img, (1, 2, 0))
    img = denormalize(img)
    ax.imshow(img)


# ## Model implementation
# 
# As you know, a GAN is comprised of two adversarial networks, a discriminator and a generator. Now that we have a working data pipeline, we need to implement the discriminator and the generator. 
# 
# Feel free to implement any additional class or function.

# ### Exercise: Create the discriminator
# 
# The discriminator's job is to score real and fake images. You have two constraints here:
# * the discriminator takes as input a **batch of 64x64x3 images**
# * the output should be a single value (=score)
# 
# Feel free to get inspiration from the different architectures we talked about in the course, such as DCGAN, WGAN-GP or DRAGAN.
# 
# #### Some tips
# * To scale down from the input image, you can either use `Conv2d` layers with the correct hyperparameters or Pooling layers.
# * If you plan on using gradient penalty, do not use Batch Normalization layers in the discriminator.

# In[ ]:





# In[ ]:


from torch.nn import Module


# In[ ]:


class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        factor = 128
        # TODO: instantiate the different layers
        
        # constraint: "input is a batch of 64x64x3 images"
        self.conv1 = nn.Conv2d(3,factor,4,2,1)
        self.conv2 = nn.Conv2d(factor,factor*2,4,2,1)
        self.conv3 = nn.Conv2d(factor*2,factor*4,4,2,1)
        self.bn2 = nn.BatchNorm2d(factor*2)
        self.bn3 = nn.BatchNorm2d(factor*4)
        self.act = nn.LeakyReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(factor*4*8*8,1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement the forward method
        #x = np.transpose(x,(0,3,1,2))
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.flatten(x)
        x = self.fc(x)
        
        return x
    
# hallo


# In[ ]:


img=dataset[0]
print(img.shape)
#img = np.transpose(img, (1, 2, 0))
D = Discriminator()
img = D(torch.unsqueeze(img,0))

img.shape


# In[ ]:


"""
DO NOT MODIFY ANYTHING IN THIS CELL
"""
# run this cell to check your discriminator implementation
discriminator = Discriminator()
#tests.check_discriminator(discriminator)


# ### Exercise: create the generator
# 
# The generator's job creates the "fake images" and learns the dataset distribution. You have three constraints here:
# * the generator takes as input a vector of dimension `[batch_size, latent_dimension, 1, 1]`
# * the generator must outputs **64x64x3 images**
# 
# Feel free to get inspiration from the different architectures we talked about in the course, such as DCGAN, WGAN-GP or DRAGAN.
# 
# #### Some tips:
# * to scale up from the latent vector input, you can use `ConvTranspose2d` layers
# * as often with Gan, **Batch Normalization** helps with training

# In[ ]:


class Generator(Module):
    def __init__(self, latent_dim: int):
        super(Generator, self).__init__()
        # TODO: instantiate the different layers
        factor = 16
        self.fc = nn.Linear(latent_dim * 1 * 1, factor * 32)
        self.deconv1 = nn.ConvTranspose2d(factor * 32, factor *16, 4,2,1)
        self.bn1 = nn.BatchNorm2d(factor*16)
        self.deconv2 = nn.ConvTranspose2d(factor*16,factor*8,4,2,1)
        self.bn2 = nn.BatchNorm2d(factor*8)
        self.deconv3 = nn.ConvTranspose2d(factor*8,factor*4,4,2,1)
        self.bn3 = nn.BatchNorm2d(factor*4)
        self.deconv4 = nn.ConvTranspose2d(factor*4,factor*2,4,2,1)
        self.bn4 = nn.BatchNorm2d(factor*2)
        self.deconv5 = nn.ConvTranspose2d(factor*2,factor,4,2,1)
        self.bn5 = nn.BatchNorm2d(factor)
        self.deconv6 = nn.ConvTranspose2d(factor,3,4,2,1)
        self.act = nn.ReLU()
        self.final_act = nn.Tanh()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement the forward method
        x = torch.permute(x, (0,2,3,1))
        x = self.fc(x)
        x = torch.permute(x, (0,3,1,2))
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.deconv4(x)
        x = self.bn4(x)
        x = self.act(x)
        x = self.deconv5(x)
        x = self.bn5(x)
        x = self.act(x)
        x = self.deconv6(x)
        x = self.final_act(x)
        return x


# In[ ]:


"""
DO NOT MODIFY ANYTHING IN THIS CELL
"""
# run this cell to verify your generator implementation
latent_dim = 128
generator = Generator(latent_dim)
tests.check_generator(generator, latent_dim)


# ## Optimizer
# 
# In the following section, we create the optimizers for the generator and discriminator. You may want to experiment with different optimizers, learning rates and other hyperparameters as they tend to impact the output quality.

# ### Exercise: implement the optimizers

# In[ ]:


import torch.optim as optim


def create_optimizers(generator: Module, discriminator: Module, lr: int):
    """ This function should return the optimizers of the generator and the discriminator """
    # TODO: implement the generator and discriminator optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr)
    return g_optimizer, d_optimizer


# ## Losses implementation
# 
# In this section, we are going to implement the loss function for the generator and the discriminator. You can and should experiment with different loss function.
# 
# Some tips:
# * You can choose the commonly used the binary cross entropy loss or select other losses we have discovered in the course, such as the Wasserstein distance.
# * You may want to implement a gradient penalty function as discussed in the course. It is not required and the code will work whether you implement it or not.

# ### Exercise: implement the generator loss
# 
# The generator's goal is to get the discriminator to think its generated images (= "fake" images) are real.

# In[ ]:


def generator_loss(fake_logits):
    """ Generator loss, takes the fake scores as inputs. """
    # TODO: implement the generator loss
    # fake_logits = D(G(z))
    criterion = nn.BCEWithLogitsLoss()
    should_be = torch.ones([len(fake_logits),1]).to(device)
    assert fake_logits.shape == should_be.shape, f"sizes are not the same:  {fake_logits.shape} vs. {fake_logits.shape}"
    loss = criterion(fake_logits, should_be)
    
    return loss


# ### Exercise: implement the discriminator loss
# 
# We want the discriminator to give high scores to real images and low scores to fake ones and the discriminator loss should reflect that.

# In[ ]:


def discriminator_loss(real_logits, fake_logits, smooth=False):
    """ Discriminator loss, takes the fake and real logits as inputs. """
    # TODO: implement the discriminator loss 
    # fake_logits = D(G(z))
    # real_logits = D(img)
    criterion = nn.BCEWithLogitsLoss()
    should_be_ones = torch.ones([len(real_logits),1]).to(device)
    if smooth:
        should_be_ones -= 0.1
    should_be_zeros = torch.zeros([len(fake_logits),1]).to(device)
    loss_r = criterion(real_logits, should_be_ones) 
    loss_f = criterion(fake_logits, should_be_zeros)
    
    return loss_r + loss_f


# ### Exercise (Optional): Implement the gradient Penalty
# 
# In the course, we discussed the importance of gradient penalty in training certain types of Gans. Implementing this function is not required and depends on some of the design decision you made (discriminator architecture, loss functions).

# In[ ]:


def gradient_penalty(real_sample: torch.Tensor, 
                     fake_sample: torch.Tensor,
                     critic: nn.Module) -> torch.Tensor:
    """
    Gradient penalty of the WGAN-GP model
    args:
    - real_sample: sample from the real dataset
    - fake_sample: generated sample
    
    returns:
    - gradient penalty
    """
    # sample a random point between both distributions
    alpha = torch.rand(real_sample.shape)
    x_hat = alpha * real_sample + (1 - alpha) * fake_sample
    
    # calculate the gradient
    x_hat.requires_grad = True
    pred = critic(x_hat)
    grad = torch.autograd.grad(pred, 
                               x_hat, 
                               grad_outputs=torch.ones_like(pred), 
                               create_graph=True)[0]
    
    # calculate the norm and the final penalty
    norm = torch.norm(grad.view(-1), 2)
    gp = ((norm - 1)**2).mean()    
    return gp


# ## Training
# 
# 
# Training will involve alternating between training the discriminator and the generator. You'll use your functions real_loss and fake_loss to help you calculate the discriminator losses.
# 
# * You should train the discriminator by alternating on real and fake images
# * Then the generator, which tries to trick the discriminator and should have an opposing loss function

# ### Exercise: implement the generator step and the discriminator step functions
# 
# Each function should do the following:
# * calculate the loss
# * backpropagate the gradient
# * perform one optimizer step

# In[ ]:


def generator_step(batch_size: int, latent_dim: int) -> Dict:
    """ One training step of the generator. """
    # TODO: implement the generator step (foward pass, loss calculation and backward pass)
    g_optimizer.zero_grad()
    
    
    z = torch.rand(batch_size,latent_dim,1,1).to(device)
    fake_images = generator(z)
    fake_logits = discriminator(fake_images)
    g_loss = generator_loss(fake_logits)
    
    g_loss.backward()
    g_optimizer.step()
    
    gp = None
    return {'loss': g_loss}





def discriminator_step(batch_size: int, latent_dim: int, real_images: torch.Tensor) -> Dict:
    """ One training step of the discriminator. """
    # TODO: implement the discriminator step (foward pass, loss calculation and backward pass)
    d_optimizer.zero_grad()
    

    z = torch.rand(batch_size,latent_dim,1,1)
    real_logits = discriminator(real_images)
    fake_images = generator(z.to(device))
    fake_logits = discriminator(fake_images.detach())
    gp = gradient_penalty(real_images, fake_images, discriminator)
    d_loss = discriminator_loss(real_logits,fake_logits,True) + gp
    
    d_loss.backward()
    d_optimizer.step()

    return {'loss': d_loss, 'gp': gp}


# ### Main training loop
# 
# You don't have to implement anything here but you can experiment with different hyperparameters.

# In[ ]:


from datetime import datetime


# In[ ]:


# you can experiment with different dimensions of latent spaces
latent_dim = 128

# update to cpu if you do not have access to a gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# number of epochs to train your model
n_epochs = 20

# number of images in each batch
batch_size = 64

### my params
lr = 0.0002


# In[ ]:


"""
DO NOT MODIFY ANYTHING IN THIS CELL
"""
print_every = 50

# Create optimizers for the discriminator D and generator G
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)
g_optimizer, d_optimizer = create_optimizers(generator, discriminator,lr)

dataloader = DataLoader(dataset, 
                        batch_size=64, 
                        shuffle=True, 
                        num_workers=4, 
                        drop_last=True,
                        pin_memory=False)


# In[ ]:


"""
DO NOT MODIFY ANYTHING IN THIS CELL
"""

def display(fixed_latent_vector: torch.Tensor):
    """ helper function to display images during training """
    fig = plt.figure(figsize=(14, 4))
    plot_size = 16
    for idx in np.arange(plot_size):
        ax = fig.add_subplot(2, int(plot_size/2), idx+1, xticks=[], yticks=[])
        img = fixed_latent_vector[idx, ...].detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = denormalize(img)
        ax.imshow(img)
    plt.show()


# ### Exercise: implement the training strategy
# 
# You should experiment with different training strategies. For example:
# 
# * train the generator more often than the discriminator. 
# * added noise to the input image
# * use label smoothing
# 
# Implement with your training strategy below.

# In[ ]:


# with open('experiments.csv','w') as f:
#     f.write('model,optimizers,training_strategy,lr,gp,result\n')
with open('experiments.csv','a') as f:
    f.write('DCGAN,Adam,smooth,0.0002,None,None\n')
    


# In[ ]:


fixed_latent_vector = torch.randn(16, latent_dim, 1, 1).float().to(device)

losses = []
for epoch in range(n_epochs):
    for batch_i, real_images in enumerate(dataloader):
        real_images = real_images.to(device)
        
        ####################################
        
        # TODO: implement the training strategy
        
        ####################################
        g_loss = generator_step(batch_size, latent_dim)
        d_loss = discriminator_step(batch_size, latent_dim, real_images)
        
        if batch_i % print_every == 0:
            # append discriminator loss and generator loss
            d = d_loss['loss'].item()
            g = g_loss['loss'].item()
            losses.append((d, g))
            # print discriminator and generator loss
            time = str(datetime.now()).split('.')[0]
            print(f'{time} | Epoch [{epoch+1}/{n_epochs}] | Batch {batch_i}/{len(dataloader)} | d_loss: {d:.4f} | g_loss: {g:.4f}')
    
    # display images during training
    generator.eval()
    generated_images = generator(fixed_latent_vector)
    display(generated_images)
    generator.train()


# ### Training losses
# 
# Plot the training losses for the generator and discriminator.

# In[ ]:


"""
DO NOT MODIFY ANYTHING IN THIS CELL
"""
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
plt.plot(losses.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses")
plt.legend()


# ### Question: What do you notice about your generated samples and how might you improve this model?
# When you answer this question, consider the following factors:
# * The dataset is biased; it is made of "celebrity" faces that are mostly white
# * Model size; larger models have the opportunity to learn more features in a data feature space
# * Optimization strategy; optimizers and number of epochs affect your final result
# * Loss functions

# **Answer:** (Write your answer in this cell)

# ### Submitting This Project
# When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_face_generation.ipynb".  
# 
# Submit the notebook using the ***SUBMIT*** button in the bottom right corner of the Project Workspace.

# In[ ]:




