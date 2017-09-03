**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./image2.jpg "Center Driving"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./image6.jpg "Normal Image"
[image7]: ./image7.jpg "Flipped Image"

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is based on NVidia model with modifications. It consists of convolutional layers and fully connected layers (code line 61-76)

The model includes RELU layers to introduce nonlinearity (code line 64-68), and the data is normalized in the model using a Keras lambda layer (code line 62). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (clone.py lines 71, 73, 75). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 28). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (clone.py line 79).

####4. Appropriate training data

Training data contains the following recordings: (1) the vehicle is kept driving along the center of the road (about two laps); (2) the vehicle recovers from left or right side; (3) on the curved section of the lap; (4) on the bridge; (5) reversed lap.

Each training sample is augmented into six: center, left, right cameras, and mirrored versions of these three.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to train a neural network that outputs a measurement based on input image from center camera.

My first step was to use a convolution neural network model similar to the NVidia network. I thought this model might be appropriate because it includes conv layers and fc layers with non-linearity introduced.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that dropout layers are added after each fully connected layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, e.g., the bridge, the big curves, etc. To improve the driving behavior in these cases, I collected more data in these sections.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (clone.py lines 61-76) consisted of a convolution neural network with the following layers and layer sizes.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input					| 160x320x3 RGB image   						| 
| Normalization			| image / 255.0 - 0.5							| 
| Cropping				| cropping = ((70,25), (0,0))					|
| Convolution 5x5     	| 2x2 stride, valid padding					 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding					 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding					 	|
| RELU					|												|
| Convolution 3x3     	| valid padding								 	|
| RELU					|												|
| Convolution 3x3     	| valid padding								 	|
| RELU					|												|
| Flatten				| 												|
| Fully connected		| outputs 100									|
| Dropout 				|												|
| Fully connected		| outputs 50									|
| Dropout 				|												|
| Fully connected		| outputs 10									|
| Dropout 				|												|
| Fully connected		| outputs 1										|

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to center lane. These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data sat, I also flipped images and angles thinking that this would help the model generalize. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 23946 number of data points. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 in my experiments as the training loss and validation loss converge. I used an adam optimizer so that manually training the learning rate wasn't necessary.
