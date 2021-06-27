#This file is used to create, train and save a new model

import tensorflow as tf
from tensorflow.keras import layers, models, datasets

#get dataset for testing
mnist = datasets.mnist # 28x28 sized images of hand writen digits 0-9

#Unpack dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Next we build a model
model = models.Sequential()

model.add(layers.Flatten())# Create input layer
model.add(layers.Dense(128, activation=tf.nn.relu))# quanlity of neyrounes, 
model.add(layers.Dense(128, activation=tf.nn.relu))# quanlity of neyrounes, 
model.add(layers.Dense(10, activation=tf.nn.softmax))# Output layer

#Preparing model for training
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',# Loss is a degree of error, function to reduce(minimize loss)
              metrics=['accuracy'])
              
# Training the model
model.fit(x_train, y_train, epochs=3)




# Summarize model's abilities
val_loss, val_acc= model.evaluate(x_test, y_test)
print("Loss and accuracy: ", val_loss, val_acc)




import matplotlib.pyplot as plt
print("Image from dataset: ")
plt.imshow(x_train[0], cmap=plt.cm.binary) # Show images, second is color map



#Normalization is a process of converting pixels from 0-255 range to 0-1 range
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
print("Normalized image from dataset: ")
plt.imshow(x_train[0], cmap=plt.cm.binary)

model.save('digits_recognizer_model') # Save on HDD
