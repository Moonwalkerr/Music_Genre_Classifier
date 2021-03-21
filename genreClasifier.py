# importing modules 
import json
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Flatten



# function for loading the dataset
def load_data(dataset_path):
    with open(dataset_path,'r') as fp:
        data = json.load(fp)
        
        
        ## convert lists into np.arrays
        
        inputs = np.array(data['mfcc'])
        targets = np.array(data['label'])
        
        return inputs, targets

# function for training MLP
def train_mlp(inputs,targets,n_epochs=60):
    optimizer = Adam(learning_rate=0.0001)
    
    X_train, X_test, y_train, y_test = train_test_split(inputs,targets,test_size=0.33)
    model = Sequential([
        
        # input layer
        Flatten(input_shape=(inputs.shape[1],inputs.shape[2])),
        
        # hidden layers
        Dense(512,activation='relu'),
        Dense(256,activation='relu'),
        Dense(64,activation='relu'),
      
        # output layer with 10 neurons
        Dense(10,activation='softmax'),
    ])
    model.compile(optimizer=optimizer,
    metrics=['accuracy'],
    loss='sparse_categorical_crossentropy')
    print("Model Summary : \n",model.summary())
    history = model.fit(X_train,y_train,
    validation_data=(X_test,y_test),
    epochs=n_epochs)



if __name__ == '__main__':
    inputs, targets = load_data('data.json')
    train_mlp(inputs, targets)
    