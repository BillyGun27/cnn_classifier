from keras.applications.xception import Xception
from keras.layers import Dense, Input, Dropout
from keras.models import Model

def preprocess_input(x):
    x /= 255.0
    x -= 0.5
    x *= 2.0
    return x

def build_xception(target_size=299,num_class=10):
    input_tensor = Input(shape=(target_size, target_size, 3))
    base_model = Xception(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=(target_size, target_size, 3),
        pooling='avg')

    for layer in base_model.layers:
        layer.trainable = True  # trainable has to be false in order to freeze the layers
        
    op = Dense(256, activation='relu')(base_model.output)
    op = Dropout(.25)(op)
    
    ##
    # softmax: calculates a probability for every possible class.
    #
    # activation='softmax': return the highest probability;
    # for example, if 'Coat' is the highest probability then the result would be 
    # something like [0,0,0,0,1,0,0,0,0,0] with 1 in index 5 indicate 'Coat' in our case.
    ##
    output_tensor = Dense(num_class, activation='softmax')(op)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model