# Package import
# Package import
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers, losses, callbacks,\
                             regularizers
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix 
import seaborn as sns
import warnings
# Ignore all warnings (not recommended for production code)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore",module="tensorflow")
from sklearn.metrics import classification_report


# Load images
def load_images(batch_size, image_height, image_width, testing_dir, training_dir):
    ''' This function load a Brain tumor patient images data from Brain tumor dataset jpg directory.
        Using a Tensorflow API, loading a images into a batches of 32 and images size is same for each images
        batch_size : Size of image batch.
        image_height : the image's height
        image_width : The width of the images
        testing_dir : The path to the training directory.
        training_dir : The path to the testing directory.
        training_ds : The dataset contains the same-sized and labelled training images
        testing_ds  : The dataset contains the same-sized and labelled testing images
        class_names : The name of the class(labels) that correspond to the dataset.
    '''
    training_ds = tf.keras.preprocessing.image_dataset_from_directory(training_dir,
                                               image_size = (image_height,image_width),
                                               batch_size = batch_size)
    testing_ds = tf.keras.preprocessing.image_dataset_from_directory(testing_dir,
                                               image_size = (image_height,image_width),
                                               batch_size = batch_size)
    class_names = training_ds.class_names
    return training_ds, testing_ds, class_names

# Define augmentation function
def images_augmentation(batch_size,dataset_dir):
    ''' This method is use to perform image augmentation using ImageDataGenerator class, As paremeter function
        is acceptiong batch_size and dataset directory.Creating a imgae_generator Object Using ImageDataGenerator 
        class perfoming a different technique on the image Dataset. 
        
        rescale: rescale the image pixels by 1./255.
        horizontal_flip : flip image horizontal for seeing image from different angle
        rotation_ragne : rotate image by 90
        shear_range : shearing transformation to input image
        brightness_range : adjusting a brightness of input image
        fill_mode : To fille empty pixel value for input image.
        
        After the creation of object dataset direcotory path given as input. with batch_size. and this function will return augmented
        images as result
    '''
    image_generator = ImageDataGenerator(
        rescale = 1./255,
        horizontal_flip = True,
        rotation_range = 90,
        shear_range = 0.2,
        brightness_range=[0.5, 1],
        fill_mode='nearest'
    )
    mriImage_aug_ds = image_generator.flow_from_directory(dataset_dir,
                                                          interpolation='nearest',
                                                          batch_size = batch_size)
    return mriImage_aug_ds

# Definging trainingModel to fit and compile a model
def trainingModel(model,optimizer,loss,metrics,x_train,y_train,x_test,y_test,epochs):
    ''' This function will compile a different model with configuration.
        Arguments:
        model: differnt models as arugments like CNN, VGG16
        optimizer: udpates model's weights while training process like Adam, SGD
        loss: which type of data are multiple classes or binary dataset
        x_train, y_train: training dataset
        x_test, y_test: testing dataset with class labels
        epochs: Number of time model will train
    This function will return a training history including accuray and losses.
    '''
    # Compile VGG16 modle using adam optimizer, lossfunction as caterorical crossentropy
    model.compile(optimizer, loss, metrics)
    # Fit the model
    history = model.fit(x_train, y_train, epochs = epochs, validation_data= (x_test, y_test), use_multiprocessing=True)
    return history

# Defining a funciont for check modle perfomance and accuracy
def evaluatePerfomanceOfModel(model,x_test, y_test, epochs,history):
    '''
        evaluatePerfomanceOfModel is used for analysing perfomance of trained model and checking accuracy and losses during traingin.
        arguments:
        model: model name as arugment.
        x_test,y_test: testing dataset
    '''    
    test_loss_score, test_accu_score = model.evaluate(x_test, y_test, verbose=0)
    print('CNN model Test Loss: ', test_loss_score)
    print('CNN model Test Accuracy', test_accu_score)

    epoch = range(1, epochs + 1)

    training_loss       = history.history['loss']
    validation_loss     = history.history['val_loss']
    training_accuracy   = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']

    fig , (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
    ax1.plot(epoch, training_loss, label='Training Loss')
    ax1.plot(epoch,validation_loss, label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()

    fig.suptitle('CNN Model validation and Loss')
    ax2.plot(epoch,training_accuracy, label='Training Accuracy')
    ax2.plot(epoch,validation_accuracy, label='Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    fig.tight_layout()

#Define model for preiction of from dataset
def modelPrediction(model, x_test,y_test):
    '''
    Make a prediction from training performance.
    '''
    labels_names = {
        0: "Glioma tumor",
        1: "Meningioma tumor",
        2: "No tumor",
        3: "Pituitary tumor",
    }
    pred_y = model.predict(x_test)
    pred_classes = np.argmax(pred_y, axis=1)

    plt.figure(figsize=(8,8))
    sns.set_style('white')

    for i in range(9):
        plt.subplot(3,3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_test[i])
        predicted_labels = labels_names[pred_classes[i]]
        actual_labels    = ', '.join([labels_names[y_test[i][0]] for j in np.where(label == 1)[0]])
        plt.xlabel(f'Predicted: {predicted_labels}\nActual: {actual_labels}')
    
    plt.suptitle('Classified Images')
    plt.tight_layout()
    plt.show()

def modelPerformance(model,x_test,y_test):
    ''' 
    This function is use for visulization of Confusion matrix of model and preformance analysis.
    Arguments:
    model: Trained model as argument.
    pred_classes: predicted Class name
    y_train: training Label as arguments
    y_test: testing Label as arguments
     
    '''
    true_label = np.argmax(y_test, axis=1)
    pred_classes = np.argmax(model.predict(x_test), axis=1)
    report = classification_report(true_label, pred_classes, target_names = ['Glioma tumor', 'Meningioma tumor', 'No tumor', 'Pituitary tumor'])
    print(report)
    confustion_metrix = confusion_matrix(true_label, pred_classes)
    cm_df = pd.DataFrame(confustion_metrix,
                     index = ['Glioma tumor', 'Meningioma tumor', 'No tumor', 'Pituitary tumor'],
                     columns = ['Glioma tumor', 'Meningioma tumor', 'No tumor', 'Pituitary tumor'])
    plt.figure(figsize=(10,5))
    sns.heatmap(cm_df, annot= True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Preicted Value')
    plt.tight_layout()
    plt.show()

# Declare all variables.
img_dataset_path = '/content/drive/MyDrive/ColabNotebooks/Brain_tumor_Dataset_JPG'
#img_dataset_path = 'dataset_image/Brain_tumor_Dataset_JPG'
training_dir = img_dataset_path +'/' + 'Training'
testing_dir  = img_dataset_path +'/' + 'Testing'
image_height = 150
image_width  = 150
batch_size   = 32

#from google.colab import drive
#drive.mount('/content/drive')

# Loading the image dataset using the provided direcotries and parameters
training_ds, testing_ds, class_Names = load_images(batch_size,image_height,image_width,testing_dir,training_dir)

# Print a list of classes availabe into dataset
print('Classes list : ', class_Names)

#Ploting a sample images from training dataset
plt.figure(figsize=(6,5))
for images, labels in training_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_Names[labels[i]])
        plt.axis("off")
plt.suptitle('Sample Images.')
plt.tight_layout()

# Data preprocessing uisng data generator
aug_Train_images = images_augmentation(batch_size,training_dir)
aug_Test_images  = images_augmentation(batch_size,testing_dir)
print('Classes: ',aug_Train_images.class_indices)

images, labels = next(aug_Train_images)
labels_list = tuple(labels)
labels_names = {
        0: "Glioma tumor",
        1: "Meningioma tumor",
        2: "No tumor",
        3: "Pituitary tumor",
    }

#Ploting a sample augmented images from training dataset
plt.figure(figsize=(6,5))
for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        label = labels[i]
        label_name = ', '.join([labels_names[j] for j in np.where(label == 1)[0]])
        plt.title(label_name)
        plt.suptitle('Augmented imange sample')
        plt.axis("off")
plt.tight_layout()

# Augmented images dataset assigning to a variable train and test 
train = aug_Train_images
test  = aug_Test_images

'''
Delaring X_Train_batches, Y_Train_batches, X_Test_batches, and Y_Test_batches lists to store
Augmented images train and test data loaded into baches and stored in a list. Once batches are
concatenating and assigning to x_train, y_train, x_test, and y_test and deleting a list from
memory. After deleting list data, split it using Skleran's method train_test_split with
test_size = 0.5.   
'''
X_Train_batches, Y_Train_batches = [], []
X_Test_batches, Y_Test_batches = [], []

for _ in range(len(train)):
    X_batch, Y_batch = train.next()
    X_Train_batches.append(X_batch)
    Y_Train_batches.append(Y_batch)

for _ in range(len(test)):
    X_batch, Y_batch = test.next()
    X_Test_batches.append(X_batch)
    Y_Test_batches.append(Y_batch)

# Declaring a train, test and validation dataset globally
global x_train, x_test, y_train, y_test

# Concatenate batches to get the complete x_train, y_train, x_test, and y_test arrays
x_train = np.concatenate(X_Train_batches)
y_train = np.concatenate(Y_Train_batches)
x_test = np.concatenate(X_Test_batches)
y_test = np.concatenate(Y_Test_batches)

# Deleting list from a memory.
del X_Train_batches, Y_Train_batches, X_Test_batches, Y_Test_batches

# Splitting a data into X_tran, Y_train, X_test and Y_test using skleran
x_train, x_test, y_train, y_test = train_test_split(x_train,y_train,test_size=0.5,shuffle = True, random_state=42)

#Check the shapes of the resulting arrays
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)