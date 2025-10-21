# PlantPulse: Multimodal AI for Early Crop Stress Detection

## Overview
**PlantPulse** is an intelligent crop health monitoring system that integrates multimodal data—plant imagery, soil sensor readings, and environmental parameters—to detect early signs of crop stress. By fusing diverse data sources, PlantPulse enables farmers to intervene promptly, reducing yield loss and optimizing resource usage.

This project is part of the **IIT Mandi iHub Multimodal AI Hackathon** and demonstrates the power of multimodal AI in agriculture.

---

## Problem Statement
Agriculture faces challenges from climate variability, pest outbreaks, and resource constraints. Traditional crop monitoring methods are labor-intensive and often detect problems too late. PlantPulse aims to provide an automated, real-time, multimodal solution for early crop stress detection using AI.

---

## Features Implemented (Mid-Submission)
- **Image Data Loading**: Load plant leaf images from the PlantVillage dataset.
- **Sensor Data Integration**: Pair images with simulated soil moisture and temperature readings.
- **Multimodal Visualization**: Display images alongside sensor metrics for comprehensive analysis.
- **Rule-Based Alerts**: Generate crop stress warnings based on combined image and sensor thresholds.

---

## Tech Stack
- **Language**: Python 3.12
- **Libraries**: 
  - `pandas` - Data handling
  - `PIL` / `Pillow` - Image processing
  - `matplotlib` - Visualization
- **Dataset**: PlantVillage (plant disease images)
- **Tools**: Kaggle API for dataset download

---