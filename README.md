# The relationship between the accuracy of a neural network and the importance of detected features - A case study based on the classification of geometric shape images

We attempt to determine whether the accuracy of a neural network is an indicator that the network is learning the relevant features for generalization. We develop a test bed where we can determine beforehand where the most salient features are. Then we develop an algorithm that allows to see the features a network uses to make its decisions as well as allowing for a quantitative measure for them. We develop an evolutionary algorithm that can find optimized networks based on the accuracy of the network and/or the features learned. Finally we attempt to determine, if existing, the relationship between the accuracy of a neural network and the relevance of detected features.

## Description
Following the discoveries made in comparing the results of applying layer-wise relevance propagation in two different medical datasets ([repository](<https://github.com/martingorosito/LRP_CovidCT_Vs_BrainMRI>)), we create a toy dataset comprised of images with a single geometric shape on them. Furthermore, we generate for each image two black and white masks, one that contains the pixel locations for the shape's edges and another for the shape's vertices. These serve as filters to locate the pixels we are interested on i.e. the features. Furthermore, we create three additional datasets, increasing the noise level by 25% increments. Thus, having a '0 noise', '25 noise', '50 noise' and '75 noise' datasets.

![image](https://user-images.githubusercontent.com/29287072/158053490-d284fb08-07e8-442b-adbf-2a5f5fe2e298.png)

We train a simple convolutional neural network to classify the data. We develop an evolutionary algorithm similar to the one used in this [project](<https://github.com/martingorosito/EA_Pruning_After_Prune_Training>) to find optimized pruned networks. However, we add a layer-wise relevance propagation to the fitness evaluation and use the filters created to locate areas of interest. We filter the results further into positively and negatively relevant pixels and use this information to develop a formula that allows for quantitative qualification of the networks ability to recognize features. 

![image](https://user-images.githubusercontent.com/29287072/158054530-12e750d4-a8f5-4626-9341-42f17f3897e8.png)

We perform three different searches across the four datasets, each search with a different fitness function to optimize. On the first, we search for those networks that excel in classification accuracy on the validation set. On the second one, we search for the networks that improve on their feature detection capabilities. On the third and last search, we search for networks that improve on both. 

The networks found are tested against a network containing all of its weights using 5x2 CV combined F test. 

## Results
This is a work in progress.

