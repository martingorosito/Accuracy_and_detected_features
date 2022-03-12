#!/usr/bin/env python
# coding: utf-8

# In[76]:


#Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]
import torch.optim as optim
import copy
import pandas as pd 
import sys
from PIL import Image
import os
import csv

np.set_printoptions(threshold=sys.maxsize)


# In[79]:


#Hyperparameters
GENERATIONS = 30
DATASET = '50_noise'
WEIGHTS = 'Weights/' + DATASET + '.pt'
DENSITY = 0.95
POP_SIZE = 8
SEED = 10
#Seeds used: 10 , 175, 247, 300, 465
BENCH_FEAT_FIT = 1
BENCH_ACC_FIT = 1
BENCH_TEST_FEAT_FIT = 1
BENCH_TEST_ACC_FIT = 1

#Change these for different optimization objectives
ACC_FIT = True 
FEAT_FIT = False

LAMBDA = 1/6
FILE_NAME = DATASET + '_' + str(ACC_FIT) + '_' + str(FEAT_FIT) + '.csv'
#Random seed for reproducibility
np.random.seed(SEED)



# In[3]:


#Data
#Load data

TRAIN_ROOT ="Data/" + DATASET + "_data/train"
VAL_ROOT ="Data/" + DATASET + "_data/val"
TEST_ROOT = "Data/" + DATASET + "_data/test"

val_dataset = torchvision.datasets.ImageFolder(root=VAL_ROOT, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.ImageFolder(root=TEST_ROOT, transform=transforms.ToTensor())

#Create data loaders
batch_size = 32


val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=len(val_dataset),
    shuffle = False 
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=len(test_dataset),
    shuffle = False
)


# In[4]:


#LRP Functions
def new_layer(layer, g):
    #Clone a layer and pass its parameters through the function g.
    layer = copy.deepcopy(layer)
    try: layer.weight = torch.nn.Parameter(g(layer.weight))
    except AttributeError: pass
    try: layer.bias = torch.nn.Parameter(g(layer.bias))
    except AttributeError: pass
    return layer 

def dense_to_conv(layers):
    #Converts a dense layer to a convolutional layer
    newlayers = []
    for i,layer in enumerate(layers):
        if isinstance(layer, nn.Linear):
            newlayer = None
            if i == 9: 
                m, n = 32, layer.weight.shape[0] 
                newlayer = nn.Conv2d(m,n,12) 
                newlayer.weight = nn.Parameter(layer.weight.reshape(n,m,12,12))
            else:
                m,n = layer.weight.shape[1],layer.weight.shape[0]
                newlayer = nn.Conv2d(m,n,1)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n,m,1,1))
            newlayer.bias = nn.Parameter(layer.bias)
            newlayers += [newlayer]
        else:
            newlayers += [layer]
    return newlayers

def get_linear_layer_indices(model):
    indices = []
    for i, layer in enumerate(model.modules()): 
        if isinstance(layer, nn.Linear): 
            indices.append(i)
    indices = [ val for val in indices]
    return indices

def apply_lrp(model, image, groundtruth_label):
    image = torch.unsqueeze(image, 0)
    
    # >>> Step 1: Extract layers
    layers = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
    layers.pop(0)
    layers = dense_to_conv(layers)
    linear_layer_indices = get_linear_layer_indices(model)
    
    # >>> Step 2: Propagate image through layers and store activations
    n_layers = len(layers)
    activations = [image] + [None] * n_layers # list of activations
    
    for layer in range(n_layers):
        if layer in linear_layer_indices:
            if layer == 9: 
                activations[layer] = activations[layer].reshape((32, 32, 12, 12))
        activation = layers[layer].forward(activations[layer])
        if isinstance(layers[layer], torch.nn.modules.pooling.AdaptiveAvgPool2d):
            activation = torch.flatten(activation, start_dim=1)
        activations[layer+1] = activation

    # >>> Step 3: Replace last layer with one-hot-encoding
    output_activation = activations[-1]
    groundtruth_activation = output_activation[:,groundtruth_label,:,:]
    one_hot_output = [val if val == groundtruth_activation else 0 
                        for val in output_activation[0]]
    activations[-1] = torch.FloatTensor(one_hot_output)

    # >>> Step 4: Backpropagate relevance scores
    relevances = [None] * n_layers + [activations[-1]]
    # Iterate over the layers in reverse order
    for layer in range(0, n_layers)[::-1]:
        current = layers[layer]
        # Treat max pooling layers as avg pooling
        if isinstance(current, torch.nn.MaxPool2d):
            layers[layer] = torch.nn.AvgPool2d(2)
            current = layers[layer]
        if isinstance(current, torch.nn.Conv2d) or            isinstance(current, torch.nn.AvgPool2d) or           isinstance(current, torch.nn.Linear):
            activations[layer] = activations[layer].data.requires_grad_(True)
            
            # Apply variants of LRP depending on the depth
            
            # Lower layers, LRP-gamma >> Favor positive contributions (activations)
            if layer <= 3:       rho = lambda p: p + 0.25*p.clamp(min=0); incr = lambda z: z+1e-9
                
            # Middle layers, LRP-epsilon >> Remove some noise / Only most salient factors survive
            if 4 <= layer <= 8: rho = lambda p: p;                       incr = lambda z: z+1e-9+0.25*((z**2).mean()**.5).data
            
            # Upper Layers, LRP-0 >> Basic rule
            if layer >= 9:       rho = lambda p: p;                       incr = lambda z: z+1e-9
            
            # Transform weights of layer and execute forward pass
            z = incr(new_layer(layers[layer],rho).forward(activations[layer]))
            # Element-wise division between relevance of the next layer and z
            s = (relevances[layer+1]/z).data                                     
            # Calculate the gradient and multiply it by the activation
            (z * s).sum().backward(); 
            c = activations[layer].grad       
            # Assign new relevance values           
            relevances[layer] = (activations[layer]*c).data                          
        else:
            relevances[layer] = relevances[layer+1]

    # >>> Potential Step 5: Apply different propagation rule for pixels
    return relevances[0]


# In[83]:


#EA Functions

#Setting the benchmark
def set_benchmark(benchmark):
    benchmark.mask_conv1 = torch.ones(benchmark.mask_conv1.shape)
    benchmark.mask_conv2 = torch.ones(benchmark.mask_conv2.shape)
    benchmark.mask_conv3 = torch.ones(benchmark.mask_conv3.shape)
    benchmark.mask_fc1 = torch.ones(benchmark.mask_fc1.shape)
    benchmark.dens_conv1 = 1
    benchmark.dens_conv2 = 1
    benchmark.dens_conv3 = 1
    benchmark.dens_fc1 = 1
    benchmark.apply_masks()
    benchmark.total_fitness = benchmark.get_fitness(acc_mode = ACC_FIT, feat_mode = FEAT_FIT)
    benchmark.test_total_fitness = benchmark.get_test_fitness(acc_mode = ACC_FIT, feat_mode = FEAT_FIT)

    BENCH_FEAT_FIT = benchmark.feature_fitness
    BENCH_ACC_FIT = benchmark.accuracy_fitness
    BENCH_TEST_FEAT_FIT = benchmark.test_feature_fitness
    BENCH_TEST_ACC_FIT = benchmark.test_accuracy_fitness

    if (ACC_FIT == True) and (FEAT_FIT == True): 
        benchmark.total_fitness = ((benchmark.accuracy_fitness-BENCH_ACC_FIT)/BENCH_ACC_FIT + (benchmark.feature_fitness-BENCH_FEAT_FIT)/BENCH_FEAT_FIT)/2

    if (ACC_FIT == True) and (FEAT_FIT == True): 
        benchmark.test_total_fitness = ((benchmark.test_accuracy_fitness-BENCH_TEST_ACC_FIT)/BENCH_TEST_ACC_FIT + (benchmark.test_feature_fitness-BENCH_TEST_FEAT_FIT)/BENCH_TEST_FEAT_FIT)/2
    return benchmark, BENCH_FEAT_FIT, BENCH_ACC_FIT, BENCH_TEST_FEAT_FIT, BENCH_TEST_ACC_FIT

#Population Creation
def Population_Creation(pop_size):
    population = []
    for i in range(pop_size):
        population.append(Individual())
        population[i].apply_masks()
        population[i].total_fitness = population[i].get_fitness(acc_mode = ACC_FIT, feat_mode = FEAT_FIT)
    return population

#Best Individual search
def search_best_individual(candidates, best_individual, print_choice = True):
    for i in range(len(candidates)):
        if best_individual.total_fitness <= candidates[i].total_fitness:
            best_individual = candidates[i]
    if print_choice == True:
        print("Best individual so far")
        print("Total Fitness: ", best_individual.total_fitness)
        print("Validation Fitness: ", best_individual.accuracy_fitness)
        print("Feature Fitness: ", best_individual.feature_fitness)
        print("Density Conv1: ", best_individual.dens_conv1)
        print("Density Conv2: ", best_individual.dens_conv2)
        print("Density Conv3: ", best_individual.dens_conv3)
        print("Density FC1: ", best_individual.dens_fc1)
    return best_individual

#Choose parents
def tournament(candidates):
    
    L = len(candidates) - 1
    
    maximum = np.random.randint(0,L)
    auxiliary = np.random.randint(0,L)
    
    if candidates[maximum].total_fitness < candidates[auxiliary].total_fitness:
        minimum = maximum
        maximum = auxiliary
    else:
        minimum = auxiliary
        
    auxiliary = np.random.randint(0,L)
    
    if candidates[auxiliary].total_fitness > candidates[minimum].total_fitness:
        minimum = auxiliary
        
    return candidates[maximum], candidates[minimum]

#Reproduction
def layer_crossover(parent_max, parent_min, offspring):
    #For reproduction we swap between full layers' masks
    
    #Conv1
    if np.random.uniform(0,1) > 0.4:
        offspring.mask_conv1 = parent_max.mask_conv1
        offspring.dens_conv1 = parent_max.dens_conv1
    else:
        offspring.mask_conv1 = parent_min.mask_conv1
        offspring.dens_conv1 = parent_min.dens_conv1
    
    #Conv2
    if np.random.uniform(0,1) > 0.4:
        offspring.mask_conv2 = parent_max.mask_conv2
        offspring.dens_conv2 = parent_max.dens_conv2
    else:
        offspring.mask_conv2 = parent_min.mask_conv2
        offspring.dens_conv2 = parent_min.dens_conv2
    
    #Conv3
    if np.random.uniform(0,1) > 0.4:
        offspring.mask_conv3 = parent_max.mask_conv3
        offspring.dens_conv3 = parent_max.dens_conv3
    else:
        offspring.mask_conv3 = parent_min.mask_conv3
        offspring.dens_conv3 = parent_min.dens_conv3
    
    #FC1
    if np.random.uniform(0,1) > 0.4:
        offspring.mask_fc1 = parent_max.mask_fc1
        offspring.dens_fc1 = parent_max.dens_fc1
    else:
        offspring.mask_fc1 = parent_min.mask_fc1
        offspring.dens_fc1 = parent_min.dens_fc1
    return offspring

def mutation(offspring):
    #For mutation, we generate a new layer mask. 
    if np.random.uniform(0,1) > 0.6:
        layer = np.random.choice(np.arange(0,4,1))
        
        if layer == 0:
            offspring.dens_conv1 = round(np.random.choice(np.arange(0.90, 1.01,0.01)),2)
            offspring.mask_conv1 = torch.tensor(np.random.binomial(1, offspring.dens_conv1, offspring.model.conv1.weight.shape))
        
        elif layer == 1:
            offspring.dens_conv2 = round(np.random.choice(np.arange(0.90, 1.01,0.01)),2)
            offspring.mask_conv2 = torch.tensor(np.random.binomial(1, offspring.dens_conv2, offspring.model.conv2.weight.shape))
        
        elif layer == 2:
            offspring.dens_conv3 = round(np.random.choice(np.arange(0.90, 1.01,0.01)),2)
            offspring.mask_conv3 = torch.tensor(np.random.binomial(1, offspring.dens_conv3, offspring.model.conv3.weight.shape))
        
        elif layer == 3:
            offspring.dens_fc1 = round(np.random.choice(np.arange(0.90, 1.01,0.01)),2)
            offspring.mask_fc1 = torch.tensor(np.random.binomial(1, offspring.dens_fc1, offspring.model.fc1.weight.shape))
    return offspring

#Choose survivors
def keep_best(candidates, offspring):
    new_population = candidates + offspring
    new_population.sort(key = lambda x: x.total_fitness, reverse = True)
    survivors = new_population[:POP_SIZE]
    return survivors

def keep_offspring(candidates, offspring):
    survivors = offspring
    candidates = []
    return survivors

#Utility functions
def print_individual(individual):
    print(str(individual))
    print("Fitness: ", individual.total_fitness)
    print("Density Conv1: ", individual.dens_conv1)
    print("Density Conv2: ", individual.dens_conv2)
    print("Density Conv3: ", individual.dens_conv3)
    print("Density FC1: ", individual.dens_fc1)
    return


# In[6]:


#Model for 128x128 images
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # kernel
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,2)
        
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2)
        
        
        self.conv3 = nn.Conv2d(32,32, 5)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2,2)
        
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(32*12*12, 5)         

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = self.fc1(x)
        return x
    
MODEL = Net()
MODEL.load_state_dict(torch.load(WEIGHTS))


# In[7]:


#Individual Representation
class Individual():
    def __init__(self):
        
        self.model = copy.deepcopy(MODEL)
        
        self.dens_conv1 = round(np.random.choice(np.arange(0.90, 1.01,0.01)),2)
        self.dens_conv2 = round(np.random.choice(np.arange(0.90, 1.01,0.01)),2)
        self.dens_conv3 = round(np.random.choice(np.arange(0.90, 1.01,0.01)),2)
        self.dens_fc1 = round(np.random.choice(np.arange(0.90, 1.01,0.01)),2)
        
        self.mask_conv1 = torch.tensor(np.random.binomial(1, self.dens_conv1, self.model.conv1.weight.shape))
        self.mask_conv2 = torch.tensor(np.random.binomial(1, self.dens_conv2, self.model.conv2.weight.shape))
        self.mask_conv3 = torch.tensor(np.random.binomial(1, self.dens_conv3, self.model.conv3.weight.shape))
        self.mask_fc1 = torch.tensor(np.random.binomial(1, self.dens_fc1, self.model.fc1.weight.shape))
        
        self.total_fitness = 0
        self.accuracy_fitness = 0
        self.feature_fitness = 0
        
        self.test_total_fitness = 0
        self.test_accuracy_fitness = 0
        self.test_feature_fitness = 0
        
        self.F_value = 0
        
        self.NRPO_count = 0
        self.PRPV_count = 0
        self.PRPE_count = 0
        self.PRPO_count = 0
        self.NRPV_count = 0
        self.NRPE_count = 0
        
        self.NRPO_sum = 0
        self.PRPV_sum = 0
        self.PRPE_sum = 0
        self.PRPO_sum = 0
        self.NRPV_sum = 0
        self.NRPE_sum = 0
        
        self.Ratio_adding = 0
        self.Ratio_counting = 0
        
        
    def get_fitness(self, acc_mode = True, feat_mode = True):
        accuracy_fitness = 0
        feature_fitness = 0
        total_fitness = 0
        
        #Accuracy Fitness
        inputs, labels = next(iter(val_loader))
        labels = labels.numpy()
        outputs = self.model(inputs).max(1).indices.detach().cpu().numpy()
        comparison = pd.DataFrame()
        accuracy_fitness = (labels==outputs).sum()/len(labels)
        self.accuracy_fitness = accuracy_fitness
        
        #Feature fitness
        feature_fitness = self.get_feat_scores(val_loader, val_dataset)
        self.feature_fitness = feature_fitness

        
        #Total fitness
        if (acc_mode == True) and (feat_mode == False):
            total_fitness = accuracy_fitness
            
        if (acc_mode == False) and (feat_mode == True):   
            total_fitness = feature_fitness 

        if (acc_mode == True) and (feat_mode == True): 
        	total_fitness = ((accuracy_fitness-BENCH_ACC_FIT)/BENCH_ACC_FIT + (feature_fitness-BENCH_FEAT_FIT)/BENCH_FEAT_FIT)/2

            
        return total_fitness
    
    def get_test_fitness(self, acc_mode = True, feat_mode = True):
        accuracy_fitness = 0
        feature_fitness = 0
        total_fitness = 0
        
        #Accuracy Fitness
        inputs, labels = next(iter(test_loader))
        labels = labels.numpy()
        outputs = self.model(inputs).max(1).indices.detach().cpu().numpy()
        comparison = pd.DataFrame()
        accuracy_fitness = (labels==outputs).sum()/len(labels)
        self.test_accuracy_fitness = accuracy_fitness
        
        #Feature fitness
        feature_fitness = self.get_feat_scores(test_loader, test_dataset)
        self.test_feature_fitness = feature_fitness
        
        #Total fitness
        #Total fitness
        if (acc_mode == True) and (feat_mode == False):
            total_fitness = accuracy_fitness
            
        if (acc_mode == False) and (feat_mode == True):   
            total_fitness = feature_fitness 

        if (acc_mode == True) and (feat_mode == True): 
        	total_fitness = ((accuracy_fitness-BENCH_TEST_ACC_FIT)/BENCH_TEST_ACC_FIT + (feature_fitness-BENCH_TEST_FEAT_FIT)/BENCH_TEST_FEAT_FIT)/2
            
        return total_fitness
    
    def apply_masks(self):
        prune.custom_from_mask(self.model.conv1, name="weight", mask = self.mask_conv1)
        prune.custom_from_mask(self.model.conv2, name="weight", mask = self.mask_conv2)
        prune.custom_from_mask(self.model.conv3, name="weight", mask = self.mask_conv3)
        prune.custom_from_mask(self.model.fc1, name="weight", mask = self.mask_fc1)
        
        prune.remove(self.model.conv1, name="weight")
        prune.remove(self.model.conv2, name="weight")
        prune.remove(self.model.conv3, name="weight")
        prune.remove(self.model.fc1, name="weight")
        return
    
    #Features scores
    def get_feat_scores(self, loader, set_slice):
        Nonzero_pixels = []
        PRPV_pixcount_avg = []
        PRPE_pixcount_avg = []
        NRPO_pixcount_avg = []

        NRPV_pixcount_avg = []
        NRPE_pixcount_avg = []
        PRPO_pixcount_avg = []

        PRPV_pixsum_avg = []
        PRPE_pixsum_avg = []
        NRPO_pixsum_avg = []

        NRPV_pixsum_avg = []
        NRPE_pixsum_avg = []
        PRPO_pixsum_avg = []
        
        Ratio_counting_avg = []
        Ratio_adding_avg = []
        Fitness_avg = []
        
        inputs, labels = next(iter(loader))
        
        model = copy.deepcopy(self.model)

        PRPV_pixcount = 0
        PRPE_pixcount = 0
        NRPO_pixcount = 0

        PRPV_pixsum = 0
        PRPE_pixsum = 0
        NRPO_pixsum = 0

        NRPV_pixcount =0
        NRPE_pixcount = 0
        PRPO_pixcount = 0

        NRPV_pixsum = 0
        NRPE_pixsum = 0
        PRPO_pixsum = 0


        for i in range(len(labels)):

            image_id = i
    
            image_relevances = apply_lrp(model, inputs[image_id], labels[image_id])
            image_relevances = image_relevances.permute(0,2,3,1).detach().cpu().numpy()[0]

            positive_relevances = np.where(image_relevances>0, image_relevances, 0)
            negative_relevances = np.where(image_relevances<0, image_relevances, 0)

            name, image_number = test_dataset.imgs[image_id]
            name = name.split('\\')
            name = name[-1]
            name = name[:-4]

            vertex_mask = name + ' vertex.bmp'
            edge_mask = name + ' edge.bmp'

            #Getting the vertex and edge masks
            vertex_path = 'Data/' + DATASET + '_test/' + str(vertex_mask)
            vertex_image = Image.open(vertex_path)
            vertex_image = (1/255)*np.array(vertex_image)

            edge_path = 'Data/' + DATASET + '_test/' + str(edge_mask)
            edge_image = Image.open(edge_path)
            edge_image = (1/255)*np.array(edge_image)

            #Removing the vertices from the edges to avoid counting them twice.
            edge_image = edge_image - vertex_image
            edge_image = np.where(edge_image>0,1,0)

            #Creating a full mask
            mask = edge_image + vertex_image
            mask = np.where(mask>0, 1, 0)
            pix_density = np.count_nonzero(mask)/(128*128)

            all_pixcount = np.count_nonzero(image_relevances)
            Nonzero_pixels.append(all_pixcount)

            #Filtering Correct relevant pixels
            NRPO = (1-mask[:,:,None])*negative_relevances
            PRPV = vertex_image[:,:,None]*positive_relevances
            PRPE = edge_image[:,:,None]*positive_relevances

            #Counting correct relevant pixels
            PRPV_pixcount = np.count_nonzero(PRPV)
            PRPE_pixcount = np.count_nonzero(PRPE)
            NRPO_pixcount = np.count_nonzero(NRPO)

            PRPV_pixcount_avg.append(PRPV_pixcount)
            PRPE_pixcount_avg.append(PRPE_pixcount)
            NRPO_pixcount_avg.append(NRPO_pixcount)

            #Suming correct relevant pixels
            PRPV_pixsum = np.sum(PRPV)
            PRPE_pixsum = np.sum(PRPE)
            NRPO_pixsum = np.sum(NRPO)

            PRPV_pixsum_avg.append(PRPV_pixsum)
            PRPE_pixsum_avg.append(PRPE_pixsum)
            NRPO_pixsum_avg.append(NRPO_pixsum)

            #Filtering Incorrect relevant pixels
            PRPO = (1-mask[:,:,None])*positive_relevances    
            NRPV = vertex_image[:,:,None]*negative_relevances   
            NRPE = edge_image[:,:,None]*negative_relevances  

            #Counting incorrect relevant pixels
            NRPV_pixcount = np.count_nonzero(NRPV)
            NRPE_pixcount = np.count_nonzero(NRPE)
            PRPO_pixcount = np.count_nonzero(PRPO)

            NRPV_pixcount_avg.append(NRPV_pixcount)
            NRPE_pixcount_avg.append(NRPE_pixcount)
            PRPO_pixcount_avg.append(PRPO_pixcount)

            #Suming incorrect relevant pixels
            NRPV_pixsum = np.sum(NRPV)
            NRPE_pixsum = np.sum(NRPE)
            PRPO_pixsum = np.sum(PRPO)

            NRPV_pixsum_avg.append(NRPV_pixsum)
            NRPE_pixsum_avg.append(NRPE_pixsum)
            PRPO_pixsum_avg.append(PRPO_pixsum)        

            #Cost Functions
            counting_factor = 1
            adding_factor = 1
            if (NRPV_pixcount + NRPE_pixcount + PRPO_pixcount) == 0:
            	counting_factor = 1
            else:
            	counting_factor = 0

            if (np.sqrt(NRPV_pixsum**2 + NRPE_pixsum**2 + PRPO_pixsum**2)) == 0:
            	adding_factor = 1
            else:
            	adding_factor = 0
            Ratio_counting = (PRPV_pixcount + PRPE_pixcount + NRPO_pixcount)/(NRPV_pixcount + NRPE_pixcount + PRPO_pixcount+counting_factor)
            Ratio_counting_avg.append(Ratio_counting)
            Ratio_adding = np.sqrt(PRPV_pixsum**2 + PRPE_pixsum**2 + NRPO_pixsum**2)/ (np.sqrt(NRPV_pixsum**2 + NRPE_pixsum**2 + PRPO_pixsum**2)+adding_factor)
            Ratio_adding_avg.append(Ratio_adding)
            Fitness = (1-LAMBDA)*Ratio_counting + LAMBDA*Ratio_adding
            Fitness_avg.append(Fitness)
        
        self.feature_fitness = np.mean(Fitness_avg)
        
        self.NRPO_count = np.mean(NRPO_pixcount_avg)
        self.PRPV_count = np.mean(PRPV_pixcount_avg)
        self.PRPE_count = np.mean(PRPE_pixcount_avg)
        self.PRPO_count = np.mean(PRPO_pixcount_avg)
        self.NRPV_count = np.mean(NRPV_pixcount_avg)
        self.NRPE_count = np.mean(NRPE_pixcount_avg)
        
        self.NRPO_sum = np.mean(NRPO_pixsum_avg)
        self.PRPV_sum = np.mean(PRPV_pixsum_avg)
        self.PRPE_sum = np.mean(PRPE_pixsum_avg)
        self.PRPO_sum = np.mean(PRPO_pixsum_avg)
        self.NRPV_sum = np.mean(NRPV_pixsum_avg)
        self.NRPE_sum = np.mean(NRPE_pixsum_avg)
        
        self.Ratio_adding = np.mean(Ratio_adding_avg)
        self.Ratio_counting = np.mean(Ratio_counting_avg)


        return self.feature_fitness
        


# In[8]:



benchmark = Individual()
benchmark, BENCH_FEAT_FIT, BENCH_ACC_FIT, BENCH_TEST_FEAT_FIT, BENCH_TEST_ACC_FIT = set_benchmark(benchmark)

history = []

#Population Generation
candidates = Population_Creation(POP_SIZE)
history.extend(candidates)
best_individual = candidates[0]
best_individual = search_best_individual(candidates, best_individual) 

g = 0

while g<GENERATIONS:
    print("\nGeneration " + str(g))
    
    #Choosing Parents
    parent_A, parent_B = tournament(candidates)
    
    #Reproduction
    offspring = []
    for i in range(POP_SIZE):
        offspring.append(Individual())
        offspring[i] = layer_crossover(parent_A, parent_B, offspring[i])
        offspring[i] = mutation(offspring[i])
        offspring[i].apply_masks()
        
        offspring[i].total_fitness = offspring[i].get_fitness(acc_mode = ACC_FIT, feat_mode = FEAT_FIT)
    
    #Choose survivors
    #candidates = keep_offspring(candidates, offspring)
    candidates = keep_best(candidates, offspring)
    history.extend(candidates)
    
    #Update best individual
    best_individual = search_best_individual(candidates, best_individual)
    
    #Update Generation
    g = g + 1

    


# In[73]:



#Getting the best individuals found
print("Getting the historical best")
history.sort(key = lambda x: x.total_fitness, reverse = True)
new_history = []
aux_list = []
historical_best = []

for obj in history:
    aux_val = np.concatenate((obj.mask_conv1.flatten(), obj.mask_conv2.flatten(), obj.mask_conv3.flatten(), obj.mask_fc1.flatten())) 
    aux_val = aux_val.tobytes()

    if aux_val not in aux_list:
        new_history.append(obj)
        aux_list.append(aux_val)

for element in new_history:
    if element.total_fitness == best_individual.total_fitness:
        historical_best.append(element)
    else:
        break

if len(historical_best)<5:
    i = 0
    L = len(historical_best)
    while i<(5 - L):
        historical_best.append(new_history[L+i])
        i = i + 1
print("Getting test fitness for historical best")
for j in range(len(historical_best)):
    historical_best[j].test_total_fitness = historical_best[j].get_test_fitness(acc_mode = ACC_FIT, feat_mode = FEAT_FIT)
    


# In[33]:


def CV_Training(loader, ind, dataset = DATASET):
    
    new_model = Net()
    if dataset == '25_noise':
        new_model.load_state_dict(torch.load('Weights/0_noise.pt'))
    elif dataset == '50_noise':
        new_model.load_state_dict(torch.load('Weights/25_noise.pt'))
    elif dataset == '75_noise':
        new_model.load_state_dict(torch.load('Weights/50_noise.pt'))
                
    prune.custom_from_mask(new_model.conv1, name="weight", mask = ind.mask_conv1)
    prune.custom_from_mask(new_model.conv2, name="weight", mask = ind.mask_conv2)
    prune.custom_from_mask(new_model.conv3, name="weight", mask = ind.mask_conv3)
    prune.custom_from_mask(new_model.fc1, name="weight", mask = ind.mask_fc1)

    
    cross_entropy_loss = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(new_model.parameters(), lr=0.001)
    epochs = 5

    # Iterate x epochs over the train data
    for epoch in range(epochs):  
        running_loss = 0
        n_batches = 0
        for i, batch in enumerate(loader, 0):
            inputs, labels = batch

            optimizer.zero_grad()
            outputs = new_model(inputs)
            # Labels are automatically one-hot-encoded
            loss = cross_entropy_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss = running_loss + loss
            n_batches = i
        
        #print("Epoch: ", epoch)
        #print("Epoch Loss: ", (running_loss/n_batches).item())
        running_loss = 0
        n_batches = 0

    return new_model


# In[12]:


def CV_Evaluate(loader, cv_model):
    inputs, labels = next(iter(loader))
    labels = labels.numpy()

    outputs = cv_model(inputs).max(1).indices.detach().cpu().numpy()
    comparison = pd.DataFrame()
    accuracy=(labels==outputs).sum()/len(labels)
    return accuracy
    


# In[31]:


def divide_dataset():
    new_dataset =torchvision.datasets.ImageFolder(root=TEST_ROOT, transform=transforms.ToTensor())
    L_50 = int(len(new_dataset)/2)
    new_train_set, new_test_set =torch.utils.data.random_split(new_dataset, [L_50,L_50])

    #Create data loaders
    new_train_loader = torch.utils.data.DataLoader(
        new_train_set,
        batch_size=32,
        shuffle=True
    )


    new_test_loader = torch.utils.data.DataLoader(
        new_test_set,
        batch_size=32,
        shuffle = True
    )
    return new_train_loader, new_test_loader


# In[42]:


#5x2CV-F test
mean = []
variance = []
numerator = []
print("Starting 5x2CV-F")
for fold in range(5):
    print("Fold N", fold)
    acc_difference = []
    #Divide dataset
    new_train_loader, new_test_loader = divide_dataset()
    #Train benchmark
    bench_model = CV_Training(new_train_loader, benchmark)
    #Evaluate benchmark
    bench_accuracy = CV_Evaluate(new_test_loader, bench_model)

    for i in range(len(historical_best)):
        cv_individual = historical_best[i]
        #Train Best Individual
        best_model = CV_Training(new_train_loader, cv_individual)
        #Evaluate benchmark
        best_accuracy = CV_Evaluate(new_test_loader, best_model)
        acc_difference.append(bench_accuracy - best_accuracy)

    #Train benchmark
    bench_model = CV_Training(new_test_loader, benchmark)
    #Evaluate benchmark
    bench_accuracy = CV_Evaluate(new_train_loader, bench_model)
    
    for i in range(len(historical_best)):
        cv_individual = historical_best[i]
        #Train Best Individual
        best_model = CV_Training(new_test_loader, cv_individual)
        #Evaluate benchmark
        best_accuracy = CV_Evaluate(new_train_loader, best_model)
        acc_difference.append(bench_accuracy - best_accuracy)
                              
    L = int(len(acc_difference)/2)

    for j in range(L):
        mu = (acc_difference[j] + acc_difference[j+L])/2
        mean.append(mu)
        var = (acc_difference[j]-mu)**2 + (acc_difference[j+L]-mu)**2
        variance.append(var) 
        num =acc_difference[j]**2 + acc_difference[j+L]**2
        numerator.append(num)


#Calculate F Value
print("Calculating F Values")
for v in range(len(historical_best)):
    Num = 0
    Den = 0
    f = 0
    for c in range(5):
        Num = Num + numerator[v + c*len(historical_best)]
        Den = Den + variance [v + c*len(historical_best)]
    f = Num/(2*Den)

    historical_best[v].F_value = f


# In[85]:

print("Writing the CSV file")
with open(FILE_NAME, 'a', newline = '') as csvfile:
    fieldnames = ["Seed", "DC1", "DC2", "DC3", "DFC1",
                  'PRPV Count', 'PRPE Count', 'NRPO Count', 'NRPV Count', 'NRPE Count', 'PRPO Count',
                  'PRPV Sum', 'PRPE Sum', 'NRPO Sum', 'NRPV Sum', 'NRPE Sum', 'PRPO Sum',
                  'Ratio Adding', 'Ratio Counting', 'Acc Fitness', 'Feat Fitness', 'Total Fitness', 'F Value']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    writer.writerow({"Seed":SEED, "DC1":benchmark.dens_conv1, "DC2":benchmark.dens_conv2, "DC3": benchmark.dens_conv3, "DFC1": benchmark.dens_fc1,
                  'PRPV Count':benchmark.PRPV_count, 'PRPE Count':benchmark.PRPE_count, 'NRPO Count':benchmark.NRPO_count, 'NRPV Count':benchmark.NRPV_count, 'NRPE Count':benchmark.NRPE_count, 'PRPO Count':benchmark.PRPO_count,
                  'PRPV Sum':benchmark.PRPV_sum, 'PRPE Sum':benchmark.PRPE_sum, 'NRPO Sum':benchmark.NRPO_sum, 'NRPV Sum':benchmark.NRPV_sum, 'NRPE Sum':benchmark.NRPE_sum, 'PRPO Sum':benchmark.PRPO_sum,
                  'Ratio Adding':benchmark.Ratio_adding, 'Ratio Counting':benchmark.Ratio_counting, 'Acc Fitness':benchmark.test_accuracy_fitness, 'Feat Fitness':benchmark.test_feature_fitness, 'Total Fitness':benchmark.test_total_fitness, 'F Value':benchmark.F_value})
    for element in historical_best:
        writer.writerow({"Seed":SEED, "DC1":element.dens_conv1, "DC2":element.dens_conv2, "DC3": element.dens_conv3, "DFC1": element.dens_fc1,
                  'PRPV Count':element.PRPV_count, 'PRPE Count':element.PRPE_count, 'NRPO Count':element.NRPO_count, 'NRPV Count':element.NRPV_count, 'NRPE Count':element.NRPE_count, 'PRPO Count':element.PRPO_count,
                  'PRPV Sum':element.PRPV_sum, 'PRPE Sum':element.PRPE_sum, 'NRPO Sum':element.NRPO_sum, 'NRPV Sum':element.NRPV_sum, 'NRPE Sum':element.NRPE_sum, 'PRPO Sum':element.PRPO_sum,
                  'Ratio Adding':element.Ratio_adding, 'Ratio Counting':element.Ratio_counting, 'Acc Fitness':element.test_accuracy_fitness, 'Feat Fitness':element.test_feature_fitness, 'Total Fitness':element.test_total_fitness, 'F Value':element.F_value})
        


# In[84]:




