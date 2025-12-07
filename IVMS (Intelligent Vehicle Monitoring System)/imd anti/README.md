# IVMS (Intelligent Vehicle Monitoring System)

**IVMS** is a state-of-the-art Vehicle Analytics & Automatic Number Plate Recognition (ANPR) system designed to monitor, detect, and analyze vehicles in video feeds. It leverages advanced Deep Learning models to identify vehicle attributes such as type, color, and license plate number with high accuracy.

## 🚀 Features

*   **Real-Time Vehicle Detection**: Detects multiple vehicles (Cars, Buses, Motorbikes, Trucks) using MobileNet-SSD.
*   **Automatic Number Plate Recognition (ANPR)**:
    *   **Localization**: Uses the **EAST** (Efficient and Accurate Scene Text Detector) model to pinpoint license plates.
    *   **Recognition**: Uses **TrOCR** (Transformer-based OCR) for high-accuracy text reading.
*   **Advanced Color Recognition**: Integrates **OpenAI's CLIP** model for "Zero-Shot" color classification (e.g., distinguishing "Silver/Grey" from "White").
*   **Vehicle Tracking**: Implements centroid tracking to maintain vehicle identity across frames.
*   **Search & Analytics Dashboard**: A user-friendly web interface (Streamlit) to search historical data by Plate Number, Color, or Vehicle Type.
*   **Database Integration**: Automatically logs all detections into a SQLite database for persistent storage.

## 🛠️ Tech Stack

*   **Language**: Python 3.8+
*   **Interface**: Streamlit
*   **Computer Vision**: OpenCV, Pillow
*   **Deep Learning Frameworks**: PyTorch, Transformers (Hugging Face)
*   **Models**:
    *   MobileNet-SSD (Vehicle Detection)
    *   EAST (Text Localization)
    *   TrOCR (Optical Character Recognition)
    *   CLIP (Color Classification)

## 📂 Project Structure

```
IVMS/
├── app.py                  # Main Streamlit Application
├── models/                 # Directory for AI Models (EAST, MobileNet)
├── modules/                # Core Logic Modules
│   ├── detector.py         # Text Detection (EAST)
│   ├── ocr.py              # OCR Engine (TrOCR)
│   ├── vehicle_detector.py # Vehicle Detection
│   ├── deep_color.py       # CLIP Color Detection
│   ├── database.py         # SQLite Database Handler
│   ├── tracker.py          # Object Tracker
│   └── utils.py            # Utility functions
├── requirements.txt        # Python Dependencies
├── setup_model.py          # Script to download EAST model
├── setup_vehicle_model.py  # Script to download Vehicle models
└── README.md               # Project Documentation
```

## ⚙️ Installation

1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd ivms
    ```

2.  **Install Dependencies**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Models**
    Run the setup scripts to download the necessary model weights.
    ```bash
    python setup_model.py
    python setup_vehicle_model.py
    ```

## 🖥️ Usage

1.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

2.  **Access the Interface**
    Open your browser and navigate to `http://localhost:8501`.

3.  **Using the System**
    *   **Live Processing Tab**: Upload a video or generate a test video to start detection.
    *   **Search & Analytics Tab**: View the log of detected vehicles. Filter by specific colors, types, or search for a partial plate number.

## 📝 Notes

*   **GPU Support**: The system checks for CUDA availability. Using a GPU is highly recommended for the CLIP and TrOCR models to ensure real-time performance.
*   **First Run**: The first time you run the system, it may take a moment to download the Transformer models (TrOCR/CLIP) from Hugging Face.

---
*Developed for Advanced Vehicle Monitoring & Security.*
