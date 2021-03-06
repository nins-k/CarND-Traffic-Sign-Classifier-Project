# Udacity Self-Driving Car Nanodegree

## Term 2 : Project 02 - Traffic Sign Classifier

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


---
### Writeup / README

This file is the **Write-Up**.

The **HTML report** is [here.](https://github.com/nins-k/CarND-Traffic-Sign-Classifier-Project/blob/master/report.html)

The **Jupyter Notebook** is [here.](https://github.com/nins-k/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

The baisc statistic summary of the set can be obtained with Python.

```python
# Number of training examples
n_train = len(y_train)

# Number of validation examples
n_validation = len(y_valid)

# Number of testing examples.
n_test = len(y_test)

# The shape of a traffic sign image.
image_shape = (np.array(X_train[0])).shape

# Unique classes/labels there are in the dataset.
n_classes = len(set(y_train))
```

#### 2. Include an exploratory visualization of the dataset.

First, I have displayed 25 images at random from the training set along with their labels, to familiarize myself with the kind of images in the training set. The same cell can be run repeatedly to view a number of images. 

Below is a part of the grid of random images.

![alt text](markdown_images/01_random_training_data.JPG "Random training data")
<hr>
Next, I have used bar charts to display the distribution of the Training and Validation data sets across the 43 output classes.

![alt text](markdown_images/02_data_distribution.JPG "Data distribution")
<hr>
Further, I have displayed one image from each class which I feel is under-represented. For determining these classes, I have selected **1000 images** as a threshold. Any classes with less than 1000 images are considered here to be in need of augmentation.

Below is a part of the displayed grid of images.

![alt text](markdown_images/03_classes_below_threshold.JPG "Classes below threshold")


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. 

The below helper method increases the brightness of an image at random up till a certain max value

```python
def random_brightness(img, value_range=70):
    
    rnd = np.random.randint(low=0, high=value_range+1)
    
    # Conver to HSV and isolate the V channel
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(img)
    v += rnd
    
    # Merge all channels and convery back to RGB
    img = cv2.merge((h, s, v))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = np.ndarray.astype(img, np.uint8)
    
    return img
```
<hr>

The second helper method performs rotation on the image up till a maximum given angle

```python
def random_rotation(img, angle_range=10):
    
    rnd_angle = np.random.randint(-angle_range, angle_range+1)
    img = transform.rotate(img, rnd_angle, preserve_range=True)
    img = np.ndarray.astype(img, np.uint8)
    
    return img

```

Using these two helper methods, the under-represented classes have been augmented with new images. The method **augment_data()** is responsible for identifying the classes that need to be augmented and how many images need to be created for each class. It will accordingly generate new images .

* The classes with less than 1000 images are identified. 
* For each such class, an image from that class is selected at random.
* Using this image, 5 new images (brightness and rotation modified) are created.
* Process is repeated from Step 2 till the class has 1000 images.

Below is an example of an image and an augmented version generated from it.

**Original Image**

![alt text](markdown_images/04_img_before_augmentation.JPG "Image before augmentation")

**Augmented Image**

![alt text](markdown_images/05_img_after_augmentation.JPG "Image after augmentation")
<hr>
The distribution of the training data after augmentation is shown below.

![Distribution Post Augmentation](markdown_images/06_distribution_post_augmentation.JPG "Post Augmentation")

The function also prints the below information:
```
27 classes need to be augmented
16891 images need to be generated

Images have been created
```
<hr>
Two other helper functions have been created to convert an image to grayscale and to normalize it. 

The grayscale output puts the pixel values in the range of [0,1].
```python
def grayscale(images):
    
    #Output is reshaped to 32x32x1 as that is the required input for the ConvNet
    return np.array([np.reshape(skc.rgb2gray(img), (32, 32, 1)) for img in images])
```

```python
# Normalizes pixel values using the mean and std values passed to the function

def normalize(images, m, s):    
    return (images - m) / s
```

The normalization function needs the mean and std as inputs. Here, I am calculating the mean and std of the training set and using it for all normalization operations<sup>[[1]](https://stats.stackexchange.com/questions/211436/why-do-we-normalize-images-by-subtracting-the-datasets-image-mean-and-not-the-c)[[2]](https://stats.stackexchange.com/questions/322802/per-image-normalization-vs-overall-dataset-normalization)</sup>.

```python
# Convert training, validation and test data to grayscale
X_train = grayscale(X_train)
X_valid = grayscale(X_valid)
X_test = grayscale(X_test)

# Calculate mean and std of the grayscaled training data
train_mean = np.mean(X_train)
train_std = np.std(X_train)

# Normalize training, validation and test data using mean and std of the training data
X_train = normalize(X_train, train_mean, train_std)
X_valid = normalize(X_valid, train_mean, train_std)
X_test = normalize(X_test, train_mean, train_std)
```

#### 2. Describe what your final model architecture looks like 

Starting with the LeNet Architecture, I have added an additional Convolutional Layer since additional features need to be identified for traffic signs.

The modifications include:
* One additional convolutional layer with SAME padding.
* One additional fully connected layer
* Dropout with 0.5 keep_prob on three fully connected layers<sup>[[3]](https://stats.stackexchange.com/questions/240305/where-should-i-place-dropout-layers-in-a-neural-network)</sup>.
* Exponential decay of learning rate in addition to Adam's own decay<sup>[[4]](https://www.tensorflow.org/versions/r0.12/api_docs/python/train/decaying_the_learning_rate)</sup>.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscaled Image   					| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, outputs 14x14x6 	|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x16   |
| RELU					|												|
| Convolution 5x5		| 1x1 stride, SAME padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, outputs 5x5x16		|
| Fully connected		| Dropout with keep_prob=0.5, Outputs 200      	|
| Fully connected		| Dropout with keep_prob=0.5, Outputs 120      	|
| Fully connected		| Dropout with keep_prob=0.5, Outputs 84      	|
| Fully connected		| Dropout with keep_prob=0.5, Outputs 43      	|
| Softmax				|        										|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The hyperparameters of my model are summarized below:
1. **Epochs**: 150
2. **Batch Size**: 512
3. **Learning Rate** (Initial): 0.001
4. **Decay Rate**: 0.9
5. **Decay Steps**: 100
6. **Keep Prob**: 0.5
7. **Optimizer**: Adam

Most of the parameters were arrived at through experimentation. 

* The validation accuracy stabalises after 100 **epochs**. There is a marginal improvement in the last 50 epochs and if training time was a concern, these could be avoided.
* A **batch size** larger than 512 did not provide any significant gains.
* A loss of 0.002 was more effective for **learning** with fewer epochs, but since training time was not a concern, I have trained the final model with a rate of 0.001
* The **Decay Rate** and **Decay Steps** are used to implement the exponential decay of the learning rate. This is not required for Adam but was reported to slightly improve accuracy. For my model, this helped the model to converge more smoothly.
* Adam **optimizer** was chosen since it is efficient and does not require a lot of fine tuning.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

My final model results were:
* Training set accuracy of **100%**
* Validation set accuracy of **96.8%**
* Test set accuracy of **95.2%**

<hr>

* I started out with the LeNet architecture used on the handwritten images classifier. Since the traffic sign images will have a lot more features to be identified, more complexity would be required in the network.
* As a first step, I added a convolution layer with depth 16.
* To make the network robuts, dropout was added to both the convolutional layers with a probability of 0.5. At this point, the network was showing a validation set accuracy in the range of 92% - 93%.
* To Improve the accuracy further, I focussed on preprocessing: Augmentation, Grayscale and Normalization. This improved accuracy to the range of 96%.
* Further, the tuning of hyperparameters and implementation of decaying learning rate helped the accuracy past 97%.
* On discussing the model with my peers, it was recommended that I add a fully connected layer to increase the complexity of the network - so the last change I did was to add a fully connected layer.

<hr>

The model has a fairly good accuracy on the Validation and Test sets, indicating that it works well with the properly preprocessed data set.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text](markdown_images/07_web_images.JPG)

The original full size images were obtained online here:
[Image 1](http://www.springdot.org/baustelle/02/02.jpg), [Image 2](http://bicyclegermany.com/Images/Laws/100_1607.jpg), [Image 3](http://bicyclegermany.com/Images/Laws/Arterial.jpg), [Image 4](http://bicyclegermany.com/Images/Laws/Stop%20sign.jpg), [Image 5](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSgdAJijJ-WAHAEZOomc3-NMivIYaXaCwDAakHcRTFv_YrMO_Jo)


The first image has a lot of objects in the background and was included as more of a challenge image (which the model failed to predict :( )

#### 2. Discuss the model's predictions on these new traffic signs

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work      		| Beware of ice/snow							| 
| Right-of-way at the next intersection    			| Right-of-way at the next intersection										|
| Priority road			| Priority road										|
| Stop      		| Stop					 				|
| Turn left ahead	| Turn left ahead     							|

<hr>

* Image **01**: This image is very noisy with large features present in the background. It was a challenge image for prediction. The model did not predict the image correctly.
* Images **02 and 03**: These images belong to classes that were well represented in the training data set. The background does contain other features such as clouds and buildings but the model was able to predict these images correctly.
* Images **04 and 05**: These images belong to classes that were augmented. Especially image 05 is of class "Turn left ahead" which initially had less than 400 images and the image does contain noise and extra features. This indicates that the model works well with augmented data.

<hr>

The model was able to **_correctly_ guess 4 of the 5 traffic signs**, which gives an accuracy of 80%. Given that the web set was of 5 images only, it is a fair comparison to the Test accuracy of 93.4%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction

#### Image 0: Class 25 (Road work) ####
This image was incorrectly predicted. The correct class is not in the top 5 probabilities. It has been completely misclassified, which was somewhat expected for this noisy image.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.758        			| Beware of ice/snow						| 
| 0.089				| Right-of-way at the next intersection									|
| 0.073				| Pedestrians								|
| 0.034	      			| Slippery road			 				|
| 0.021				    | Double curve    							|

<hr> 

#### Image 01: Class 11 (Right-of-way at the next intersection) ####
This image was correctly predicted with a probability of 1. The model is very confident about the prediction.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000     			| Right-of-way at the next intersection							| 
| 0.000			| General caution									|
| 0.000					| Double curve								|
| 0.000	      			| Beware of ice/snow			 				|
| 0.000				    | Roundabout mandatory
<hr> 

#### Image 02: Class 12 (Priority road) ####
This image was correctly predicted with a probability of 1. The model is very confident about the prediction.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000       			| Priority road								| 
| 0.000					| Roundabout mandatory									|
| 0.000						| Yield								|
| 0.000		      			| General caution			 				|
| 0.000					    | Speed limit (100km/h)
<hr> 

#### Image 03: Class 14 (Stop) ####
This image was correctly predicted with a probability of 1. The model is very confident about the prediction.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000      			| Stop								| 
| 0.000				| Keep right									|
| 0.000					| Speed limit (30km/h)									|
| 0.000	      			| Yield			 				|
| 0.000				    | Turn right ahead							|

<hr> 

#### Image 04: Class 34 (Turn left ahead) ####
This image was correctly predicted with a probability of 0.695. The image has other features including clouds and text at the bottom. The class of this image was also originally under-represented and had to be augmented.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.695       			| Turn left ahead							| 
| 0.110 				| General caution									|
| 0.044					| Roundabout mandatory									|
| 0.029 	      			| Turn right ahead			 				|
| 0.024				    | Priority road   							|

<hr> 


