from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_folder = './Data/flower_photos'  

# Create an ImageDataGenerator with augmentation options
datagen_augmented = ImageDataGenerator(
    rescale=1.0/255.0,
    horizontal_flip=True  # Apply horizontal flip as augmentation
)

# Create an augmented ImageDataGenerator object
imds_augmented = datagen_augmented.flow_from_directory(
    image_folder,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)
