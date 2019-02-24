from model.squeezenet import build_squeezenet,preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenetv2 import preprocess_input
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

SEED = 42
TARGET_SIZE = 299

model = build_squeezenet(weight_decay=1e-4, image_size=299)



model.compile(
    optimizer=optimizers.SGD(lr=1e-2, momentum=0.9, nesterov=True), 
    loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy']
)



# construct the image generator for data augmentation
'''
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=1./255)

#setup generator
train_generator = train_datagen.flow_from_directory(
    directory=r"img/train/",
    target_size=(TARGET_SIZE,TARGET_SIZE),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=SEED
)

valid_generator = test_datagen.flow_from_directory(
    directory=r"img/val/",
    target_size=(TARGET_SIZE, TARGET_SIZE),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=SEED
)

test_generator = test_datagen.flow_from_directory(
    directory=r"img/test/",
    target_size=(TARGET_SIZE, TARGET_SIZE),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=SEED
)

'''

data_generator = ImageDataGenerator(
    data_format='channels_last',
    preprocessing_function=preprocess_input
)

train_generator = data_generator.flow_from_directory(
    directory=r"img/train/",
    target_size=(TARGET_SIZE,TARGET_SIZE),
    batch_size=64
)

valid_generator = data_generator.flow_from_directory(
    directory=r"img/val/",
    target_size=(TARGET_SIZE, TARGET_SIZE),
    batch_size=64
)

test_generator = data_generator.flow_from_directory(
    directory=r"img/test/",
    target_size=(TARGET_SIZE, TARGET_SIZE),
    batch_size=1
)



#train

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=30, verbose=1 ,
                     callbacks=[
        ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=2, epsilon=0.007),
        EarlyStopping(monitor='val_acc', patience=4, min_delta=0.01)
    ]
)

#evaluate
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.evaluate_generator(generator=valid_generator ,steps=STEP_SIZE_VALID)
#save
model.save("squeezenet.h5")

#test
test_generator.reset()
pred=model.predict_generator(test_generator,verbose=1)

#predict
predicted_class_indices=np.argmax(pred,axis=1)

#labels = (train_generator.class_indices)
#labels = dict((v,k) for k,v in labels.items())
lbl = [ "airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey" , "ship", "truck" ]  
predictions = [lbl[k] for k in predicted_class_indices]

#save predict result
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("mobilenet_results.csv",index=False)


#graph
plt.plot(model.history.history['loss'], label='train')
plt.plot(model.history.history['val_loss'], label='val')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('logloss')


plt.plot(model.history.history['acc'], label='train')
plt.plot(model.history.history['val_acc'], label='val')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')

