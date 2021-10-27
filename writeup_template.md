# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model_summary.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

First, the model 'cut' the sky and front of car, with the function Cropping2D after that, normalize the data and centering this.

Like I said before, I use the NVIDIA neural network, which contain 5 convolutional layer and 3 Dense. The first 3 convolutional layer use 5x5 filter and 2x2 strides and the other 2 use 3x3 filters and 1x1 stride. After this, flatten the data and put 3 Dense layer.

#### 2. Attempts to reduce overfitting in the model

I use augmented data for reduce overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and take smooth and slow curve to learn better the stearing on  the road.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I choose the NVIDIA model. I put the architecture and training for 7 epochs, the training and validation loss are almost the same (0.023, 0.024).  The most important thing I can see was the amounnt of data that we need to train the model. 

The unique change on the model was, the convolutional layers had __valid__ padding, I used from the second layer __same__ padding. With this configuration I have better results.



#### 2. Final Model Architecture

Here is a visualization of the architecture.
![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I drive for 3 laps in the center of the road, sometimes I go to the edges and recover the center. I drive the last lap slow for take smooth curve.

With this amount of data I add a little more with cv2.flip, with this is "like we drive counter clockwise". 

The preprocessing step was made with lambda function.

The 80% of the data was for training, the rest was use for validation.