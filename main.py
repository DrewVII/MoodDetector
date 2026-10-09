from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, RandomFlip, RandomRotation, RandomZoom
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import Sequential
from keras import utils, Input
import os

import matplotlib.pyplot as plt

MODEL_OUTPUT_DIR = "./model_save"

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# Configuring our seed
utils.set_random_seed(42)

# Contains our dataset
working_dir = 'dataset/emotions'

# All of the different moods
classes = ['angry', 'happy', 'nothing', 'sad']

data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)

training_generator = data_generator.flow_from_directory(
    working_dir,
    classes=classes,
    target_size=(128, 128),
    batch_size=100,
    subset='training',
    color_mode='grayscale'
)

validation_generator = data_generator.flow_from_directory(
    working_dir,
    classes=classes,
    target_size=(128, 128), # D'ailleurs, pourquoi est-ce en puissance de 2 ?
    batch_size=100,
    subset='validation',
    color_mode='grayscale'
)

images, labels = next(training_generator)

first_batch_img = images[0]
img_shape = first_batch_img.shape
# 256 is the default value ?
# print(img_shape)
# print(first_batch_img.min())
# print(first_batch_img.max())

convolutional_layer = Conv2D(16, (5, 5), activation='relu')

max_pool_layer = MaxPool2D()

# Adding these additional layers improved the accuracy while reducing the loss
convolutional_layer2 = Conv2D(32, (5, 5), activation='relu')

max_pool_layer2 = MaxPool2D()

convolutional_layer3 = Conv2D(64, (5, 5), activation='relu')

max_pool_layer3 = MaxPool2D()

flatten_layer = Flatten()
#global_average_pool_layer = GlobalAveragePooling2D()

postf_dense_layer = Dense(128, activation="relu") # Nombre de features max regroupés par celles extraites dans les précédentes convs layers

anti_ovf_dropout = Dropout(0.5)

dense_layer = Dense(4, activation='softmax')

optimizer = Adam(learning_rate=0.001) # 0.61 for 0.001 Best one! (Flatten tech) / 0.58 for 0.0001 / stuck somewhere around 0.2770 (how do we call something like that? Zigzag on training data) for 0.01 higher is better ? -> No, No!

cnn_model = Sequential([
    Input(shape=img_shape),
    RandomFlip('horizontal'),
    RandomRotation(0.1),
    RandomZoom(0.1),
    convolutional_layer,
    max_pool_layer,
    convolutional_layer2,
    max_pool_layer2,
    convolutional_layer3,
    max_pool_layer3,
    #global_average_pool_layer,
    flatten_layer,
    postf_dense_layer,
    anti_ovf_dropout,
    dense_layer
])

cnn_model.summary()

cnn_model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    mode='max',
    restore_best_weights=True
)

history = cnn_model.fit(
    training_generator,
    validation_data=validation_generator,
    epochs=100,
    callbacks=[early_stopping]
)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()

val_loss, val_acc = cnn_model.evaluate(validation_generator)

print(f'Here is the state of the model, accuracy: {val_acc} | loss: {val_loss}')

cnn_model.save(f'{MODEL_OUTPUT_DIR}/mood_detector.keras')

print(f'Model saved in {MODEL_OUTPUT_DIR} directory.')