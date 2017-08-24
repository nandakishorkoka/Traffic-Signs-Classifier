# Traffic Sign Recognition 

## Overview: Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/samples.png "Samples"
[image2]: ./examples/train_hist.png "Training Data"
[image3]: ./examples/valid_hist.png "Validation Data"
[image4]: ./examples/test_hist.png "Testing Data"
[image5]: ./images/1.png "Traffic Sign 1"
[image6]: ./images/2.png "Traffic Sign 2"
[image7]: ./images/3.png "Traffic Sign 3"
[image8]: ./images/4.png "Traffic Sign 4"
[image9]: ./images/5.png "Traffic Sign 5"
[image10]: ./images/6.png "Traffic Sign 6"

## Data Set Summary & Exploration

### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

### 2. Include an exploratory visualization of the dataset.

### Samples: 
![alt text][image1]

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed among the classes for each of the sets. The histograms show that data is distributed similarly for each of the datasets. 

### Training: 
![alt text][image2]

### Validation: 
![alt text][image3]

### Testing: 
![alt text][image4]


## Design and Test a Model Architecture

### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I chose to keep the images in color as converting them to grayscale would not provide much benefit except for little bump in speed. I normalized the images so that the mean in zero and the intensity values are between -1. and +1. Normalization is an important step to ensure the model treats all weights equally. 

I did not plan to generating additional data. I wanted to first evaluate the performance of the model before generating new data. Additional data can be generated using  basic transformations on original image - such as shifting pixels or warping. 


### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I explored external resources and papers on choosing filter depths and kernel sizes for CNNs. Some good techniques were using multiple smaller receptive fields than a single larger receptive fields - for e.g. 2 3x3 convolution layer instead of a single 7x7 convolution. Secondly, a stack of 1x5 convolution and a 5x1 convolution seems to learn better representation than a single 5x5 convolution. I applied max pooling at the end of the stack to reduce information loss. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
| Convolution 5x1     	| 1x1 stride, Valid padding, outputs 24x28x12 	|
| RELU					|												|
| Convolution 1x5     	| 1x1 stride, Valid padding, outputs 24x24x24 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 20x20x36 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 10x10x36 				|
| Convolution 3x3     	| 1x1 stride, Valid padding, outputs 8x8x48 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x48 				|
| Fully connected	- 256 units	| Dropout with keep_prob 0.75        									|
| RELU					|												|
| Fully connected	- 64 units	| Dropout with keep_prob 0.75        									|
| RELU					|												|
| Fully connected	- 43 units	|      									|
| Softmax				|         									|
 

### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer with batch size of 128 and learning rate of 0.001 and 20 epochs. However the accuracy plateaus much sooner and 10 Epochs is sufficient. I did not explore tuning the hyperparameters as I was already achieving ~97% on the validation set. 

### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I first using the same architecture of Lenet to observe how well the model performs on the data. I was able to achieve a validation accuracy of ~ 90%. From there, I explored techniques on good practices for CNNs. I modified the network to be more deeper and used smaller receptive fields. These changes provided a bump in performance to ~ 97%. Applying dropouts further helped the accuracy. 

My final model results were:
* training set accuracy of 
* validation set accuracy of 95.7
* test set accuracy of 94.5

I found the following techniques useful: 
* using multiple smaller receptive fields than a single larger receptive fields - for e.g. 2 3x3 convolution layer instead of a single 7x7 convolution. 
* a stack of 1x5 convolution and a 5x1 convolution seems to learn better representation than a single 5x5 convolution.

## Test a Model on New Images

### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9] ![alt text][image10]

The first image might be difficult to classify because ...



####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


