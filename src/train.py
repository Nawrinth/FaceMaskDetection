from src.model import FaceMaskDetector
from src.preprocessing import PrePrecessing
from .dataset_loader import load_dataset

source_path = load_dataset()

# Preprocess Images and Split It 

preprocessor = PrePrecessing()

preprocessor.split_data(source , "/output" , 0.7 , 0.2)
train_ds , val_ds , text_ds = preprocessor.preprocessImage("/output")


# Train the model 

detector = FaceMaskDetector()
detector.build_model()
detector.compile_model(1e-5)
history = detector.train(train_ds , val_ds , 10)
detector.save_model("../models/best_mask_detector.h5")

