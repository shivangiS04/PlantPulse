# PlantPulse: Multimodal AI for Early Crop Stress Detection

## Overview
**PlantPulse** is an award-winning intelligent crop health monitoring system integrating multimodal data—plant leaf imagery, soil moisture, and environmental temperature—to detect early signs of crop stress. Leveraging advanced AI techniques and data fusion, PlantPulse enables precise, real-time diagnosis so farmers can take timely actions to reduce losses and optimize inputs.

This project won the **IIT Mandi iHub Multimodal AI Hackathon**, showcasing how the convergence of computer vision and sensor fusion can transform agricultural monitoring for sustainable farming.

---

## Problem Statement
Agriculture is vulnerable to climate variability, pests, and resource limitations. Traditional monitoring methods are often manual, expensive, and late in detecting crop issues, leading to substantial yield loss. PlantPulse solves this by automating early stress detection through fusing multimodal signals, resulting in faster, more reliable alerts and better crop management.

---

## Features Implemented (Winning Submission)
- **Deep Learning on PlantVillage Dataset:** Fine-tuned MobileNetV2 to classify 15 crop stress categories with high accuracy (95%+).
- **Simulated Sensor Data Fusion:** Soil moisture and temperature sliders simulate real sensor inputs incorporated into the AI decision pipeline.
- **Multimodal Crop Stress Alerts:** Combines image prediction confidence with sensor-derived stress scores to raise actionable alerts.
- **Interactive Streamlit Demo:** User-friendly web app with image upload, sensor controls, AI inference, confidence metrics, and Grad-CAM visual explanations.
- **Engaging UX:** Crop stress facts, actionable remedy suggestions, FAQs, and branding to educate and build trust with end users.

---

## Tech Stack
- **Programming Language:** Python 3.12
- **Deep Learning:** PyTorch, Torchvision
- **Web Framework:** Streamlit
- **Data Handling & Processing:** Pandas, Numpy, Pillow, OpenCV
- **Visualization:** Matplotlib, Seaborn
- **Data Source:** PlantVillage dataset (via Kaggle API)

---

## How It Works
1. Upload a crop leaf image via the interface.
2. Simulate sensor inputs or integrate real IoT sensor data.
3. The fine-tuned MobileNetV2 model predicts stress classes.
4. Sensor data is normalized and fused with model confidence to generate a stress score.
5. Alerts notify the user about crop health status.
6. Grad-CAM visualizations highlight leaf regions influencing classification.
7. Helpful facts and remedy recommendations guide corrective actions.

---

## Getting Started
1. Clone the repository.
2. Install dependencies (refer to `requirements.txt`).
3. Run locally or deploy on Streamlit Cloud.
4. Upload images and explore multimodal predictions interactively.

---

## Future Directions
- IoT integration for live sensor streaming.
- Expand sensor suite beyond moisture and temperature (humidity, light).
- Model enhancements using larger datasets and transformer-based architectures.
- Deployment on edge devices for farmer field use.
- Crop-specific customizations for precision agriculture.

---
## Getting Started

## Setup Instructions
1. **Clone the repository**
```bash
git clone [https://github.com/shivangiS04/PlantPulse.git](https://github.com/shivangiS04/PlantPulse.git)
cd PlantPulse
```

2. **Create and activate a virtual environment**
```bash 
python3 -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash 
pip install -r requirements.txt
```

4. **Run the Streamlit app**
```bash 
streamlit run streamlit_demo.py
```

---
## Acknowledgments
- IIT Mandi iHub Multimodal AI Hackathon
- Open-source frameworks: PyTorch, Streamlit, and more
- PlantVillage community for high-quality dataset

---

**Made with 💡 by Shivangi Singh**  
Transforming agriculture with AI for a sustainable future.
