#**Traffic Sign Recognition**



**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[example-img]: ./report-images/example-img.png
[hist-img]: ./report-images/hist-img.png
[image1]: ./my-signs/no-passing.tiff
[image2]: ./my-signs/over-three-tons.tiff
[image3]: ./my-signs/priority-road.tiff
[image4]: ./my-signs/road-work.tiff
[image5]: ./my-signs/wild-animal-crossing.tiff


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/wrgrant/carnd-project2-sign-classifier)


####Data Set Summary & Exploration

The code for this step is contained in the third cell of the Jupyter notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799 samples
* The size of test set is 12,630 smaples
* The shape of a traffic sign image is (32, 32, 3) pixels
* The number of unique classes/labels in the data set is 43

#### Exploratory visualization

The code for this step is contained in the fourth and fifth code cells of the Jupyter notebook.  

Here is an image showing the form of a sample image from the data set. I was amazed at the low quality! Doubly amazed that this amount of quality is enough to predict an incoming image to such a good degree (as we cover below).

![][example-img]

This is another image showing the distribution of number of images per unique label (image type) in the training, validation, and test data sets, respectively. Of note is that some labels appear much more than others, but across the data sets, the distribution is uniform.

![][hist-img]



###Design and Test a Model Architecture
####1. Pre-processing

The code for this step is contained in the sixth code cell of the Jupyter notebook.

All I did for pre-processing was to normalize the image data, ie convert from 0-255 range for the pixels to 0-1. I did this because normalization was suggested as a step in the instructions. This ultimately took my validation set accuracy from ~89% to the required 93%.


####2. Model architecture

I used the standard LeNet model architecture. In training, I tried all sorts of things, such as:
* adding extra convolution layers
* changing the first layer filter size to a larger one, around 10x10
* messing with the dimensions of the middle layers of the model (increase, decrease)
* adding dropout between the fully-connected layers
* pre-processing the data to grayscale and HSV

But in the end, after all of these experiments, none of them increased the validation accuracy. In fact, each of these experiments resulted in LOWER accuracy. So I ultimately reverted on all of them and stuck with LeNet. I have a feeling the layers and dimensions were the result of many more hours tweaking parameters than I am willing to spend on this project! I figured it would be smarter to use a trusted architecture and move on.


The code for my final model is located in the seventh cell of the Jupyter notebook.

Here is a table representation of the model.

| Layer       		|    Description	|
|:--------------:|:-----------------------:|
| Input        	| 32x32x3 RGB image  		|
| Convolution 5x5| 1x1 stride and padding, outputs 28x28x6 	|
| Max pooling  	| 2x2 stride,  outputs 14x14x6 	|
| RELU					|												      |
| Convolution 5x5 | 1x1 stride and padding, outputs 10x10x16	|
| Max pooling   | 2x2 stride, outputs 5x5x16 |
| RELU          |                            |
| Flatten       | Outputs 400x1                |
| Fully connected		| Input 400, output 120   	|
| RELU          |                            |
| Fully connected		| Input 120, output 84   	|
| RELU          |                            |
| Fully connected		| Input 84, output 43   	|
| Softmax				|       									|




####3. Model training

The code for training the model is located in the ninth through twelfth cell of the Jupyter notebook.

To train the model, I used an the default AdamOptimizer routine from the LeNet lab. I have a very fast laptop, but I don't have an NVIDIA graphics card, so I trained the models on CPU, thus, I found that with a BATCH_SIZE value of 40, I achieved best epoch performance. I'm not exactly sure what the limitation was, as I have 16GB of ram in this machine and memory usage of the python process didn't seem to change based upon the BATCH_SIZE setting, but the further I increased this parameter, the slower each epoch ran.

I chose 20 epochs because that was about the point where I could tell the validation accuracy would plateau.



####4. Solution approach
The code for calculating the accuracy of the model is located in the twelfth cell of the Jupyter notebook.

My final model results were:
* validation set accuracy of 93.9%
* test set accuracy of 90.6%

As noted above, I first tried an iterative process to 'improve' upon the LeNet model architecture, but none of those iterations ended up improving the validation accuracy!

I settled upon the LeNet architecture because according to the intro videos to the lesson, they should be able to give about 93-95% accuracy, which was the minimum required. I explored potentially implementing some of the more modern architectures such as AlexNet, GoogLeNet, ResNet, etc, but they appeared to be much too complicated for this simple project. I believe the point of this project was to get us familiar with how a CNN model works, and I have a much more intimate feel for them after doing this project.



###Test a Model on New Images

####1. Acquiring new images

Here are five German traffic signs that I found on the web from the site: http://www.gettingaroundgermany.info/zeichen.shtml
I figured these would be nice images because they are very clean. I didn't intentionally choose images that would be difficult to classify. These images have much less noise and other 'real-world' artifacts in them than the test and training set (I didn't look at every image in the data sets so maybe this is a false assumption) These images also are of high contrast ratio, and that probably helped make them easier to classify.

![][image1] ![][image2] ![][image3] ![][image4] ![][image5]


####2. Performance on new images

The code for making predictions on my final model is located in the sixteenth cell of the Jupyter notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:-----------------------:|
| No passing      		| No passing   									|
| No over 3.5 tons   	| No over 3.5 tons										|
| Priority road				| Priority road									|
| Road work      		| Road work				 				|
| Wild animal crossing| Wild animal crossing     							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%!! This compares favorably to the accuracy on the test set of 90%.

####3. Model certainty

For softmax probabilities, I only output the top 3 for each image, because the certainty for each image was roughly 100%. I will omit graphs and tables showing this because it would be a waste of space. The actual and guessed index for each image is shown in cell 17 of the Jupyter notebook.
