import os
import keras
import tensorflow as tf
from keras.models import load_model
from keras.models import Model
from keras.layers import Average
from keras.layers import Concatenate
from keras import Input

def load_models():
    models = []
    directory = 'models'
    for file in os.listdir(directory):
        print("loading " + file)
        model = load_model(directory + "/" + file)
        model._name = file.split(".")[0]
        models.append(model)
    return models


models = load_models()
print(models)
model_input = Input(shape=(60,5))
model_outputs = [model(model_input) for model in models]
ensemble_output = Average()(model_outputs)
# ensemble_output = Concatenate()(model_outputs)
ensemble_model = Model(inputs=model_input, outputs=ensemble_output)
ensemble_model.compile(optimizer='adam',loss='mse')
ensemble_model.save('merged.h5', include_optimizer=False)