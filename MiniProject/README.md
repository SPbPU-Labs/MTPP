# Mini-Project: Real-time Hand Sign Detection with Multi-Threading in Python

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.11%2B-orange)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-yellowgreen)](https://mediapipe.dev)

Multithreaded real-time hand gesture recognition system with comparison of single-threaded and multithreaded versions.

## 📌 Task

**Goal**: Develop a Python application that can detect hand signs in real-time video streams using multi-threading to improve responsiveness and efficiency.

**Key Technologies**: Python, OpenCV (for video processing), potentially a machine learning library (like TensorFlow or PyTorch) for sign recognition.

**Focus**: Demonstrating the benefits of multi-threading in a computer vision task.

## ✨ Features

- Recognition of 6 gestures: ✌️ Victory, 👍 Thumbs Up, 👌 OK, 🤘 Rock, 🖕 Middle Finger, 👋 Hello
- Two operating modes:
  - Single-threaded (sequential processing)
  - Multi-threaded (parallel capture and processing)
- Visualization of the finger condition (Bent/Straight)
- Performance comparison mode
- Support for real-time FPS output

## 🚀 Startup

1. Clone repo:

```bash
git clone https://github.com/SPbPU-Labs/MTPP
cd ./MiniProject
```

2. Install requirements

```bash
pip install -r requirements.txt
```

3. Run project

```bash
# Single-threaded mode
python metrics.py

# Multithreaded mode
python metrics.py --multi
```
