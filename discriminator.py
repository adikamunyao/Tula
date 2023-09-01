import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Flatten, Dropout, Dense
from tensorflow.keras.models import Model

# Parameters
dropoutProb = 0.5
numFilters = 64
alpha = 0.2  # Slope of LeakyReLU
inputSize = (64, 64, 3)
filterSize = 5

# Create the discriminator architecture
input_layer = Input(shape=inputSize)
x = Dropout(dropoutProb)(input_layer)
x = Conv2D(numFilters, filterSize, strides=(2, 2), padding='same')(x)
x = LeakyReLU(alpha=alpha)(x)
x = Conv2D(2 * numFilters, filterSize, strides=(2, 2), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=alpha)(x)
x = Conv2D(4 * numFilters, filterSize, strides=(2, 2), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=alpha)(x)
x = Conv2D(8 * numFilters, filterSize, strides=(2, 2), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=alpha)(x)
x = Conv2D(4, 1)(x)
x = Flatten()(x)
discriminator_output = Dense(1, activation='sigmoid')(x)

# Create the discriminator model
discriminator = Model(input_layer, discriminator_output)

# Compile the discriminator model
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

# Summary of the discriminator architecture
discriminator.summary()
