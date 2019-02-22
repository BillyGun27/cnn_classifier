import tensorflow as tf
import numpy as np

from keras.models import Model

#important for distillation
def softmax_with_temperature(logits, temperature=1):
  logits = logits / temperature
  return np.exp(logits) / np.sum(np.exp(logits))

#import matplotlib.pyplot as plt
#%matplotlib inline

if __name__ == "__main__":
    #all the teacher and student already trained beforehand

    #prepare teacher
    teacher = load_model("darknet")

    #evaluate teacher
    test_loss, test_acc = teacher.evaluate(x_test, y_test)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_acc)


    #to get logits , we have to remove the softmax layer
    #check the teacher layer first from summary to get the name
    #there are other method as well
    model_sans_softmax = Model(inputs=teacher.input, outputs=teacher.get_layer("logits").output)
    model_logits = model_sans_softmax.predict(x_train)

    #make sure softmax removed
    model_sans_softmax.summary()

    #create graph to display diff between temperature

    #transfer the dark knowledge
    #prepare student
    student = load_model("mobilenet")

    #evaluate student
    test_loss, test_acc = student.evaluate(x_test, y_test)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_acc)

    # train a model with the exact same architecture on the softened probabilities 
    # (instead of the one hot encoded labels). 
    # The small model should also be trained with the same value of temperature. 
    # So we will have to first modify the model. Let's add a temperature layer before 
    # the softmax layer in the small model.
    new_small_model = load_model("mobilenet")
    logits = new_small_model.get_layer('logits').output
    logits = layers.Lambda(lambda x: x / temperature, name='Temperature')(logits)
    preds = layers.Activation('softmax', name='Softmax')(logits)
    
    new_small_model = Model(inputs=new_small_model.input, outputs=preds)
    new_small_model.summary()

    #train student with temperature
    new_small_model.compile(optimizer=RMSprop(lr=0.03), loss='categorical_crossentropy', metrics=['accuracy'])
    new_small_model.fit(x_train, softened_train_prob, epochs=100, batch_size=512)

    test_loss, test_acc = new_small_model.evaluate(x_test, y_test)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_acc)