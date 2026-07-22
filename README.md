# AI-Based Smart Attendance Monitoring System

A complete Smart Attendance Monitoring System built with PyTorch, ResNet18, and OpenCV. It recognizes registered students and logs their attendance automatically into CSV and Excel formats.

## Features
- **Data Preprocessing**: Automatically splits the CASIA-WebFace dataset into 70% Train, 15% Validation, and 15% Test sets.
- **Deep Learning Model**: Uses ResNet18 with transfer learning for accurate face recognition.
- **Evaluation**: Calculates Accuracy, Precision, Recall, F1-score, Confusion Matrix, and Classification Report.
- **Real-Time Attendance**: Detects faces via webcam using OpenCV and marks attendance only once per student per day.
- **Logs**: Saves attendance to `attendance_log.csv` and `attendance_log.xlsx`.

## Project Structure
```text
AI_Attendance_System/
├── attendance/          # Real-time face detection, recognition and logging modules
├── dataset/
│   ├── CASIA_WEB_FACE/  # Put raw class-wise folders here
│   └── splits/          # Automatically generated train, val, test folders
├── models/              # ResNet18 model definition
├── notebooks/           # Interactive Jupyter notebooks for all stages
├── preprocessing/       # Scripts to handle dataset preparation and augmentation
├── training/            # Training loop and evaluation scripts
├── utils/               # Helper functions
├── main.py              # Main entry point to run different parts of the system
└── requirements.txt     # Python dependencies
```

## Setup Instructions

1. **Install Anaconda or Miniconda** if you haven't already.
2. **Create a Conda Environment:**
   ```bash
   conda create -n attendance_env python=3.11 -y
   conda activate attendance_env
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Prepare Dataset:**
   Place the CASIA-WebFace dataset inside the `dataset/CASIA_WEB_FACE` directory. Each student should have their own folder named with their Student ID or Name (e.g., `dataset/CASIA_WEB_FACE/John_Doe/`).

## Usage

You can use the provided Jupyter Notebooks in the `notebooks/` folder for an interactive experience, or run the system directly using `main.py`.

### 1. Split Dataset
```bash
python main.py split
```

### 2. Train Model
```bash
python main.py train --epochs 10 --batch-size 32
```

### 3. Evaluate Model
```bash
python main.py evaluate
```

### 4. Run Real-Time Attendance
```bash
python main.py run-webcam
```
