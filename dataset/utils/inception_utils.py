from keras import Model
from keras.models import load_model
from keras.applications.inception_resnet_v2 import preprocess_input
import numpy as np
import keras.backend as K
from keras.applications.inception_v3 import InceptionV3
from dataset.utils.shared import checkpoint_url
from keras.layers import Input


K.clear_session()

input_tensor = Input(shape = (299, 299, 3))


base_model = InceptionV3(weights=None, include_top=False, input_tensor=input_tensor)

x = base_model.load_weights(checkpoint_url)
# y = x.get_layer('predictions').output
y = base_model.output
model = Model(inputs=base_model.input, outputs=y)


def inception_resnet_v2_predict(images):
    images = images.astype(np.float32)
    predictions = model.predict(preprocess_input(images))
    return predictions
