import tensorflow as tf 
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential 


#Definindo Datasets a serem usados
datatrain = tfds.load('svhn_cropped',split='train')
datateste = tfds.load('svhn_cropped',split='test')





def preprocessamento(dataset):
    
    
    x=tf.cast(dataset['image'],tf.float32)/255.0
    
    y=dataset['label']

    return x,y


datatrain=datatrain.map(preprocessamento)
datateste=datateste.map(preprocessamento)

datatrain=datatrain.batch(15)
datateste=datateste.batch(15)


cnn=Sequential([

    layers.Input(shape=(32,32,3)),
    
    layers.Conv2D(32,(3,3),activation='relu'),

    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3),activation='relu'),

    layers.MaxPooling2D((2,2)),

    layers.Reshape((6,6*64)),

    layers.Bidirectional(
        
        layers.LSTM(128,return_sequences=False)
    ),

    layers.Dense(10,activation='softmax')

])

cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'],)

cnn.fit(datatrain, epochs=20, validation_data=datateste)
