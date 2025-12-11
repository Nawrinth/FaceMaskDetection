# 😷 Face Mask Detection Using CNN and OpenCV

## 📝 Description

This project is a **real-time Face Mask Detection system** built using **Convolutional Neural Networks (CNN)** and **OpenCV**. It detects faces through a webcam feed and classifies them as **“MASK”** ✅ or **“NO MASK”** ❌. The repository includes scripts for **training the model** and **running real-time detection**.

## 📂 Project Structure

```
FaceMaskDetection/
│
├── src/
│   ├── train.py            # Train the CNN model 🏋️‍♂️
│   ├── model.py            # CNN model architecture 🧠
│   ├── preprocessing.py    # Dataset preprocessing 🔄
│   └── dataset_loader.py   # Load dataset from local path or Google Drive 🌐
│
├── run.py                  # Run real-time mask detection with OpenCV 🎥
├── data/                   # Optional dataset folder 📁
├── venv/                   # Python virtual environment 🐍
└── README.md
```

## ⚙️ Setup Instructions

1. **Clone the repository**

```bash
git clone <repo_url>
cd FaceMaskDetection
```

2. **Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate  # Linux
.venv\Scripts\activate     # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt 
```

## 🏋️ How to Train the Model

The training script is located in `src/train.py`. Run:

```bash
cd FaceMaskDetection
python -m src.train
```

* The model will be saved as `.h5` 💾 for later use.
* Dataset can be loaded from a local path or Google Drive using `dataset_loader.py` 🌐.

## 🎥 How to Run Real-Time Mask Detection

The OpenCV inference script is located at `run.py` in the root folder. Run:

```bash
python run.py
```

* Opens your webcam and detects faces 🖥️.
* Classifies each face as **MASK** ✅ (green box) or **NO MASK** ❌ (red box).

## 🚀 Future Enhancements

* Multi-class mask detection (e.g., cloth, surgical, N95) 🏷️
* Integration with CCTV systems 📹
* Sign language detection or gesture recognition ✋

## 🛠️ Technologies Used

* Python 3.x 🐍
* TensorFlow / Keras (CNN) 🧠
* OpenCV (real-time detection) 🎥
* NumPy / Pandas (data handling) 📊
