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

I explored external resources and papers on choosing filter depths and kernel sizes for CNNs. After trying the Lenet architecture, I improvised by adding more layers. A stack of 1x5 convolution and a 5x1 convolution seems to learn better representation than a single 5x5 convolution. 

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
| Flatten            | Output 768 unit                 | 
| Fully connected	- 256 units	| Dropout with keep_prob 0.75        									|
| RELU					|												|
| Fully connected	- 64 units	| Dropout with keep_prob 0.75        									|
| RELU					|												|
| Fully connected	- 43 units	|      									|
| Softmax				|         									|
 

### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.


I first tried the Lenet architecture and achieved a validation accuracy of ~ 90%. It appears the model suffers from high bias. Also, the many pooling layers aggresively reduce the dimensionality which I suspected caused loss in information.  I modified the network to be more deeper and used smaller receptive fields. These changes provided a bump in performance to ~ 97%. Applying dropouts further helped the accuracy. 

An ideal method to select hyperparameters would be to perform a grid search. However, in the interest of time, and from the discussion (https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network), I experimented with a few batch sizes - 32, 128, 512 and learning rates. Batch size of 128 and learning rate of 0.001 performed best. 

I chose to increase the epochs to a value where I was sure the performance plateaued and increasing it further would not provide any further benefit. 

Following are some techniques which I found useful:
* using multiple smaller receptive fields than a single larger receptive fields - for e.g.  two 5x5 convolution layer instead of a single 9x9 convolution. 
* a stack of 1x5 convolution and a 5x1 convolution seems to learn better representation than a single 5x5 convolution.


### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.


I had to perform multiple iteration to achieve an high accuracy. I first began experimenting and changing the Lenet architecture. Finally, I made a few design choice as described above that led me to achieve better results. 

My final model results were:
* training set accuracy of 99.6
* validation set accuracy of 96.8
* test set accuracy of 94.3

I found the following techniques useful: 
* using multiple smaller receptive fields than a single larger receptive fields - for e.g. multiple 5x5 convolution layer instead of a single 9x9 convolution. 
* a stack of 1x5 convolution and a 5x1 convolution seems to learn better representation than a single 5x5 convolution.

## Test a Model on New Images

### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9] ![alt text][image10]

The 5th image is difficult to classify as it is similar to "General Caution" at low resolution and that might confuse the model. I expected the model to work well on all other images 


### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Children crossing      		| Children crossing   									| 
| Speed limit (60km/h)     			| Speed limit (30km/h) 										|
| Speed limit (30km/h)					| Speed limit (30km/h)											|
| Stop      		| Turn left ahead					 				|
| Pedestrians			| Traffic signals      							|
| Right-of-way at the next intersection			| Right-of-way at the next intersection      							|

The model was able to correctly guess 3 of the 6 traffic signs, which gives an accuracy of 50%. The accuracy of the test dataset was ~ 94%. This large difference appears to be due to either of the two issues: 
1. The network is overfitting. The difference in accuracy of the training and validation sets seem to imply overfitting. However they are quite close. 
2. The new images are very different from the training data. Taking a second look - I feel this might be the case. With better normalization and with additional image processing techniques this issue might be resolved. 


### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 32nd cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Children Crossing sign (probability of 0.92). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
0.928 | Children crossing | 
0.068 | Bicycles crossing | 
0.002 | Dangerous curve to the right | 
0.001 | Priority road | 
0 | Speed limit (30km/h) | 



For the second image, the model predicts incorrectly that the sign in 30km speed limit. The actual sign is 60km speed limit but is nowhere in the top five.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
0.43 | Speed limit (30km/h) | 
0.388 | Wild animals crossing | 
0.101 | Keep right | 
0.024 | Slippery road | 
0.01 | Dangerous curve to the right | 


For the third image, the model is relatively sure that this is a 30km speed limit sign (probability of 0.88). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
0.88 | Speed limit (30km/h) | 
0.12 | Roundabout mandatory | 
0 | Priority road | 
0 | Go straight or left | 
0 | General caution | 

For the fourth image, the model is relatively sure that this is a turn left ahead sign (probability of 0.7).And, stop sign is nowhere in the top five. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
0.773 | Turn left ahead | 
0.148 | Keep right | 
0.025 | Right-of-way at the next intersection | 
0.023 | Speed limit (60km/h) | 
0.008 | No passing for vehicles over 3.5 metric tons | 

For the fifth image, the model predicts Traffic signal sign instead of Pedestrians with a high probability of 0.9

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
0.9 | Traffic signals | 
0.046 | General caution | 
0.033 | Dangerous curve to the right | 
0.02 | Pedestrians | 
0 | Priority road | 

For the sixth image, the model is relatively sure that this is a Right of way at next intersection (probability of 0.99). 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
0.999 | Right-of-way at the next intersection | 
0.001 | Priority road | 
0 | Bicycles crossing | 
0 | Double curve | 
0 | Beware of ice/snow | 

##  (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Refer to cells 49 to 53 in the ipython notebook. The cells display the feature maps in each of the different layers. A each layer the the feature maps capture two important attributes - the shape of the sign and the text "Stop". I notice that a lot of the feature maps are completely blank and therefore leading to incorrect predictions. I feels there is scope for improvement observing the feature maps. 


