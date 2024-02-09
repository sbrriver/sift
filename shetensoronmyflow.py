import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import os
import cv2

image_size = (128,128) #image sizes
batch_size = 1 # val from our data set
path_name = r'C:\Users\Raksi\Documents\Code\sift\GALAXIES'

"""Set data"""
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    path_name,
    labels='inferred',
    label_mode='int', #is this correct?
    class_names=['0','1'], #Used to control the order of the classes (otherwise alphanumerical order is used)
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=image_size,
    seed=123,
    validation_split=0.2,
    subset='both',
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
    shuffle=True
) # directory name and info, initally set it so yes sn is 1, no sn is 0

# """Plot image"""
# plt.figure(figsize = (10,10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3,3,i+1)
#         plt.imshow(images[i].numpy().astype('uint8'))
#         plt.title(int(labels[i]))
#         plt.axis('off')

"""Rotate and flip images to randomize dataset - I think this is pointless"""
data_augmentation = keras.Sequential()
# data_augmentation = keras.Sequential(
#     [
#         tf.keras.layers.RandomFlip('horizontal'),
#         tf.keras.layers.RandomRotation(0.1)
#     ]
# )

# """Plot new images"""
# plt.figure(figsize = (10,10))
# for images, _ in train_ds.take(1):
#     for i in range(9):
#         augmented_images = data_augmentation(images)
#         ax = plt.subplot(3,3, i +1)
#         plt.imshow(augmented_images[0].numpy().astype('uint8'))
#         plt.axis('off')

train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls = tf.data.AUTOTUNE,
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

def make_model(input_shape, num_classes):
    """Makes model.

    Args:
        input_shape (ndarray): a 4-D array  (shape (30,4,10) means an array or tensor with 3 dimensions, containing 30 elements in the first dimension, 4 in the second and 10 in the third, totaling 30*4*10 = 1200 elements or numbers).
        num_classes (float): number of classifiers.
        
    Returns:
        ndarray: 
    """
    inputs = keras.Input(shape=input_shape)
    
    # Entry block
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs) #output same as input shape
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(inputs) #takes input shape: (batch_size, imageside1, imageside2, channels). 4+D tensor with shape: batch_shape + (channels, rows, cols) if data_format='channels_first' or 4+D tensor with shape: batch_shape + (rows, cols, channels) if data_format='channels_last'.
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    '''
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    '''

    previous_block_activation = x  # Set aside residual

    ''''''
    for size in [256]:#[256, 512, 728]: #why not 768?
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x) #I changed all the channels to 1
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)
        # Project residual
        residual = tf.keras.layers.Conv2D(size, 3, strides=2, padding="same")(
            previous_block_activation
        )
        x = tf.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = tf.keras.layers.SeparableConv2D(1024, 3, padding="same")(x) #When you see a trailing comma after a single element inside parentheses, it indicates that you are creating a tuple with a single element.
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    ''''''
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    #x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size + (1,), num_classes=2) #since our image is grayscale, there should only be one color channel at the end
# keras.utils.plot_model(model, show_shapes=True) #must install pydot


epochs = 5 # pick a value

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

def predict_img_path(folder: str):
    """"Randomly picks an image and predicts sn or non_sn"""
    path = os.path.join(path_name, folder)
    png_images = [f for f in os.listdir(path) if f.endswith('.png')]
    random_png_path = os.path.join(path, np.random.choice(png_images))
    img = keras.utils.load_img(random_png_path, target_size=image_size)

    img_array = keras.utils.img_to_array(img)

    grayscale_image = cv2.cvtColor(np.array(img_array), cv2.COLOR_RGB2GRAY)
    resized_grayscale_image = cv2.resize(grayscale_image, (image_size[1], image_size[0]))
    resized_grayscale_image = resized_grayscale_image.reshape(1, image_size[0], image_size[1], 1)

    predictions = model.predict(resized_grayscale_image)
    score = float(predictions[0][0])
    print(f"This image is {100 * (1 - score):.2f}% supernova")

print("\n\nSupernova img: ")
predict_img_path("1")

print("\nNot supernova img: ")
predict_img_path("0")
