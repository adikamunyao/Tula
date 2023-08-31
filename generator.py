import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Conv2DTranspose, BatchNormalization, ReLU
from tensorflow.keras.models import Model

# Parameters
filterSize = 5
numFilters = 64
numLatentInputs = 100
projectionSize = (10, 10, 1)
#projectionSize = (4, 4, 512)

# Define featureInputLayer
def featureInputLayer(input_dim):
    return Input(shape=(input_dim,))

# Define projectAndReshapeLayer
def projectAndReshapeLayer(output_shape):
    return Reshape(target_shape=output_shape)

# Create the generator architecture
latent_inputs = featureInputLayer(numLatentInputs)
x = projectAndReshapeLayer(projectionSize)(latent_inputs)
x = Conv2DTranspose(4 * numFilters, filterSize, use_bias=False, padding='valid')(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv2DTranspose(2 * numFilters, filterSize, strides=(2, 2), padding='same', use_bias=False)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv2DTranspose(numFilters, filterSize, strides=(2, 2), padding='same', use_bias=False)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
generated_output = Conv2DTranspose(3, filterSize, strides=(2, 2), padding='same', activation='tanh')(x)

# Create the generator model
generator = Model(latent_inputs, generated_output)

# Summary of the generator architecture
generator.summary()
