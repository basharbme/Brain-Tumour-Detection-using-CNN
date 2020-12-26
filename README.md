# Brain-Tumour-Detection-using-CNN
Building a brain tumour detection model using a convolutional neural network in Tensorflow &amp; Keras.
Name: Yogeshwari Bahiram (MIS: 111803038)

PROJECT TOPIC :
	Brain-Tumor-Detector Using CNN (convolutional neural network)

Files included in folder:
	
	1. Research paper.pdf : Referred research paper for thid project
	2. block diagram.pdf : Block diagram of our project
	3. Data Augmentation.ipynb : Code for data augmentation
	4. Brain Tumor Detection.ipynb : Driver code of project
	5. yes and no folder : Dataset
	6.augmented folder : Created while running 'Data Augmentation.ipynb' file
	7. models : Output of CNN architecture.
	8. Accuracy.png and loss.png : Graph of loss and accuracy obtained from matplotlib
	9. Dataset link : https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection

Introduction :	
Building a detection model using a convolutional neural network in Tensorflow & Keras.

About the data :
The dataset contains 2 folders: yes and no which contains 253 Brain MRI Images. The folder yes contains 155 Brain MRI Images that are tumorous and the folder no contains 98 Brain MRI Images that are non-tumorous.

Segmentation :  
Segmentation is done using deep learning wherein skull segmentation is done first by seperating skull boundary. Then 
brain part segmentation is done by cropping the tumour part from the MRI image 
crop_brain_contour() function : Herein we are locating the top corners along a contour and then crop the image.

Data Augmentation :
Since this is a small dataset, There wasn't enough examples to train the neural network. Also, data augmentation was useful in taclking the data imbalance issue in the data.
Before data augmentation, the dataset consisted of:
155 positive and 98 negative examples, resulting in 253 example images.

hms_string_calculate() function :This function will calculate the elapsed time required for futher processing.

augment_data() : using the module of keras i.e ImageDataGenerator we assigned various parameters to the original dataset. 
data_summary() : After data augmentation, now the dataset consists of:
1085 positive and 980 examples, resulting in 2065 example images.

Note: these 2065 examples contains also the 253 original images. They are found in folder named 'augmented data'.

Data Preprocessing:
For every image, the following preprocessing steps were applied:
Crop the part of the image that contains only the brain (which is the most important part of the image).
Resize the image to have a shape of (240, 240, 3)=(image_width, image_height, number of channels): because images in 
the dataset come in different sizes. So, all images should have the same shape to feed it as an input to the neural 
network.

Load up the data (load_data() function) :
Then in order for for the images to be in the same size we performed normalization where we assign a fix size to image and store them in an array.

plot_sample_images() : Plotting the augmented images having brain tumour and also images without brain tumour.

Spliting the data (split_data() function) :
The data was split in the following way:
    70% of the data for training.
    15% of the data for validation.
    15% of the data for testing.

compute_f1_score will return the accuracy of the image.

Convolution Neural Network (CNN) Architecture (build_model() function):

Here we are using pre trained VGG-19 CNN module architecture.
Each input x (image) has a shape of (240, 240, 3) and is fed into the neural network. And, it goes through the 
following layers:

    A Zero Padding layer with a pool size of (2, 2).
    A convolutional layer with 32 filters, with a filter size of (7, 7) and a stride equal to 1. This layer will add feature like sharpening, blurring of images.
    A batch normalization layer to normalize pixel values to speed up computation.
    A ReLU activation layer.
    A Max Pooling layer with f=4 and s=4.
    A Max Pooling layer with f=4 and s=4, same as before. This will reduce the dimension of images without loosing another important features.
    A flatten layer in order to flatten the 3-dimensional matrix into a one-dimensional vector. This will convert 3D matrix image into 1D vector.
    A Dense (output unit) fully connected layer with one neuron with a sigmoid activation (since this is a binary classification task).
Then compling this all layers together into a model.

Training the model :

Here we are using validation data for training purpose.The model was trained for 24 epochs and these are the loss & accuracy plots againts number of iterations on X axis were plotted by using matplotlib (plot_metrics() function).
The best validation accuracy was achieved on the 23rd iteration that is 91%.

So the model with 91% accuracy is considered as best model and their corresponding loss and accuracy values assigned to the x and y variabls respectively.

Then we are finding the f1 score for testing dataset which is 0.88 and for validation dataset it is 0.91 which is approximately equal.

Results:
----------------------------------------------------------------------------------------
The best model (the one with the best validation accuracy) detects brain tumor with:
88.7% accuracy on the test set.
0.88 f1 score on the test set.
These resutls are very good considering that the data is balanced.

Performance table of the best model:
----------------------------------------------------------------------------------------
	    Validation set 	Test set
Accuracy    91% 		89%
F1 score    0.91 		0.88

Conclusion :
So finally we got the expected accuracy for brain tumour classification that is 88% which approximately eqauls to the accuracy for validated data i.e. 91%.

Our project is threefold as :
1.	Tumour regions from dataset are segmented through CNN model.
2.	The segmented data further augmented using several parameters to increase number of data samples.
3.	A pretrained CNN model is fine tuned for multi grade brain tumour classification.

In this system we try to improve the accuracy by utilizing data augmentation and deep learning and also tried to add more parameters in data augementation.
