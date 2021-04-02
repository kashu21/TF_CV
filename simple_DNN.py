# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import tensorflow as tf

print(tf.__version__)

#explore callbacks

#Earlier, when you trained for extra epochs, you had an issue where your loss might change. 
#It might have taken a bit of time for you to wait for the training to do that and 
#you might have thought that it'd be nice if you could stop the training when you reach a desired value, such as 95% accuracy.
#for this purpose , we use callbacks!
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch, logs={}):
        if(logs.get('accuracy') > 0.97):
            print("\nReached 97% accuracy so cancelling further training!")
            self.model.stop_training = True
            
            
callbacks = myCallback()

#loading data
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels),(test_images,test_labels) = mnist.load_data()

import matplotlib.pyplot as plt
plt.imshow(training_images[0])
print("Label: ",training_labels[0])
print(training_images[0].shape)
#print(training_images[0])

#when training a neural network, it is easier to treat all values as between 0 and 1, using a process called normalization.
training_images = training_images/255.0
test_images = test_images/255.0
#print(training_images[0])

# designing the model

#    Sequential defines a sequence of layers in the neural network.
#    Flatten takes a square and turns it into a one-dimensional vector.
#    Dense adds a layer of neurons.
#    Activation functions tell each layer of neurons what to do. There are lots of options too.
#    Relu effectively means that if X is greater than 0 return X, else return 0. It only passes values of 0 or greater to the next layer in the network.
#    Softmax takes a set of values, and effectively picks the biggest one. For example, if the output of the last layer looks like [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05], then it saves you from having to sort for the largest value—it returns [0,0,0,0,1,0,0,0,0].

#the rule of thumb that the first layer in your network should be the same shape as your data
#Another rule of thumb—the number of neurons in the last layer should match the number of classes you are classifying for. In this case, it's the digits 0 through 9, 
#so there are 10 of them, and hence you should have 10 neurons in your final layer.
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),tf.keras.layers.Dense(128,activation = tf.nn.relu),
                                                            tf.keras.layers.Dense(10,activation =tf.nn.softmax)])
print(model)
print(model.dtype)

#compile and train the model

model.compile(optimizer= tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(training_images, training_labels,epochs=5,callbacks=[callbacks])

#Test the model
print("\n")
results = model.evaluate(test_images,test_labels)
print("test results: ", results)

classifications = model.predict(test_images)
print(classifications.shape)
print(classifications[0])
print(test_labels[0])

