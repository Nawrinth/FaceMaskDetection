import tensorflow as tf 
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D , Dense , Dropout , Input
from tensorflow.keras.models import Model

class FaceMaskDetector:
    def __init__(self , image_size=(224,224) , freeze_layers=True):
        self.image_size = image_size
        self.freeze_layers = freeze_layers
        self.model = None

    def build_model(self):

        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.image_size + (3,)
        )

        if self.freeze_layers:
            for layers in base_model.layers:
                layers.trainable = False


        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        output = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs=base_model.input, outputs=output)

        print("Model build successfully")

    def compile_model(self , learningRate):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        print("Model compiled successfully")

    def train(self , train_ds ,val_ds , epochs):
        print("Training started")

        history = self.model.fit(
            train_ds ,
            validation_data=val_ds ,
            epochs=epochs
        )

        print("Training completed")

        return history

    def save_model(self , path):
        self.model.save(path)
        print("Model saved successfully")
