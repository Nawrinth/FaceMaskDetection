import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class PrePrecessing:
    def __init__(self , image_size=(224,224)):
        self.image_size = image_size
        print("Preprocessing class ")

    """ 
    Source => 
        /with_mask
        /without_mask
    
    Destination =>
        /train
            /with_mask
            /without_mask
        /test
            /with_mask
            /without_mask
        /val
            /with_mask
            /without_mask
    """
    def split_data(self , source , destination , train_size , val_size):
        classes = os.listdir(source)

        for cls in classes:
            path = os.path.join(source , cls)
            print(f"Class {cls} path: {path}")

            files = os.listdir(path)
            random.shuffle(files)

            train_end = int(len(files)*train_size)
            val_end = int(len(files)*(train_size+val_size))

            train_files = files[:train_end]
            val_files = files[train_end:val_end]
            test_files = files[val_end:]

            for split in ['train','val' , 'test']:
                os.makedirs(os.path.join(destination , split , cls) , exist_ok=True)

            def copy_files(img_list , split):
                for img in img_list:
                    src = os.path.join(path , img)
                    dst = os.path.join(destination , split , cls , img)
                    shutil.copy(src , dst)

            copy_files(train_files , 'train')
            copy_files(val_files , 'val')
            copy_files(test_files , 'test')


    def preprocessImage(self , base_path ):
        train = os.path.join(base_path , "train")
        val = os.path.join(base_path , "val")
        test = os.path.join(base_path , "test")

        train_datagen = ImageDataGenerator(
            preprocessing_function = preprocess_input,
            rotation_range=20,
            zoom_range=0.2,
            horizontal_flip=True,
            width_shift_range=0.2,
            height_shift_range=0.2,
        )

        val_datagen = ImageDataGenerator(
            preprocessing_function = preprocess_input,
        )

        test_datagen = ImageDataGenerator(
            preprocessing_function = preprocess_input
        )

        train_ds = train_datagen.flow_from_directory(
            train,
            target_size=self.image_size,
            batch_size=32,
            class_mode='binary',
            shuffle=True
        )

        val_ds = val_datagen.flow_from_directory(
            val,
            target_size=self.image_size,
            batch_size=32,
            class_mode='binary',
            shuffle=True
        )
        test_ds = test_datagen.flow_from_directory(
            test,
            target_size=self.image_size,
            batch_size=32,
            class_mode='binary',
            shuffle=True
        )

        return train_ds , val_ds , test_ds


