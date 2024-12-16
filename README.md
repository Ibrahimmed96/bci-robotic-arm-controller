# BCI-Robotic-Arm-Controller
Control a robotic arm using EEG brain signals (OpenBCI + ROS2 + Python)
# 🧠 BCI Robotic Arm Controller

This project reads EEG signals from OpenBCI, classifies brainwave patterns, and controls a robotic arm in real time using ROS2.

## 📌 Features
- Real-time EEG acquisition
- Brain-signal classification (e.g., focus vs relax)
- ROS2-based actuator control
- Modular and scalable design

## 🛠️ Tech Stack
- Python, ROS2, OpenBCI SDK
- Scikit-learn, NumPy
- STM32 (for motor controller)

## 🧪 How to Run

```bash
pip install -r requirements.txt
python openbci_reader/stream_data.py
python signal_processing/eeg_classifier.py
ros2 run ros2_publisher arm_control_node

# Updated on 2022-07-05
# Updated on 2024-06-09
# Updated on 2024-12-26
# Updated on 2024-12-16