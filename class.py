###archivo entrenada
#v2ffdsafsa
#%%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
import numpy as np

#%%
image_size = (150, 150)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'E:\DATA\orquesta\Base\paraetiquetar\solocuerdas\paradrive\cats',
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    shuffle=True,
    interpolation='bilinear',
    
    batch_size=32)
    
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'E:\DATA\orquesta\Base\paraetiquetar\solocuerdas\paradrive\cats',
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    interpolation='bilinear',
    
    batch_size=batch_size,
    
)

#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")

#%%

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)
#%%
scali=keras.layers.Lambda(lambda x: x/255)
#%%
def make_model():

    num_classes=4
    inputs=keras.Input(shape=(150,150,3))
    x=scali(inputs)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

        x = layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.GlobalAveragePooling2D()(x)
       
        activation = "softmax"
        units = num_classes

        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(units, activation=activation)(x)
        return keras.Model(inputs, outputs)


model = make_model()

#%%
model.summary()
#%%
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='d:/model{epoch:08d}.h5',
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_freq='epoch')


#%%
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), # default from_logits=False
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

#%%
model.fit(train_ds, validation_data=val_ds, epochs=20,callbacks=[model_checkpoint_callback])


# %%

model.save('d:/moditu.h5')

###############
# %%
model=tf.keras.models.load_model('d:/modi.h5')
# %%
import cv2
a=cv2.imread('d:/kalman.png')

l=cv2.resize(a,(150,150))

l=l[None,]
#%%
model.predict(l)
# %%
import numpy as np
np.argmax(model.predict(l),axis=1)
#%%
data_augmentation=keras.Sequential(
    [
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
    ]
)

#%%
##############################

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import preprocessing


#%%


image_size = (150, 150)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'E:\DATA\orquesta\Base\paraetiquetar\solocuerdas\paradrive\cats',
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    shuffle=True,
    interpolation='bilinear',
    crop_to_aspect_ratio=True,
    batch_size=32)
    
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'E:\DATA\orquesta\Base\paraetiquetar\solocuerdas\paradrive\cats',
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    interpolation='bilinear',
    crop_to_aspect_ratio=True,
    batch_size=batch_size,
    
)


#%%
scali=keras.layers.Lambda(lambda x: x/255)

# %%
def make_model():

    num_classes=4
    inputs=keras.Input(shape=(150,150,3))
    x=scali(inputs)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

        x = layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.GlobalAveragePooling2D()(x)
        if num_classes == 2:
            activation = "sigmoid"
            units = 1
        else:
            activation = "softmax"
            units = num_classes

        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(units, activation=activation)(x)
        return keras.Model(inputs, outputs)


model = make_model()
        

  


# %%
a=model()
# %%
model.summary()
# %%
