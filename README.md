# 🧠 BCI Robotic Arm Controller

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![ROS2](https://img.shields.io/badge/ROS2-Galactic%7CHumble-green.svg)](https://docs.ros.org)
[![OpenBCI](https://img.shields.io/badge/OpenBCI-Cyton%7CDaisy-orange.svg)](https://openbci.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive brain-computer interface system that enables real-time control of robotic arms using EEG signals. This project integrates OpenBCI hardware, advanced signal processing algorithms, and ROS2 communication protocols to create a seamless brain-to-robot control pipeline.

> **🎯 Perfect for:** Research applications, assistive robotics, human-machine interfaces, and advanced BCI studies

**Authors:** Ibrahim Mediouni, Selim Ouirari  
**Date:** July 2022  
**Version:** 1.0.0

## 📌 Features

### 🧠 **Brain-Computer Interface**
- **Real-time EEG Acquisition**: High-quality signal streaming from OpenBCI devices (Cyton/Daisy)
- **Advanced Signal Processing**: Digital filtering, artifact removal, and feature extraction
- **Brain State Classification**: Machine learning-based recognition of mental states (focus, relax, motor imagery)
- **Multi-channel Support**: 8 or 16 channels with daisy chain configuration
- **Simulation Mode**: Full functionality without hardware for development and testing

### 🤖 **Robotic Control**
- **Precise Movement Control**: 6-DOF robotic arm control with inverse kinematics
- **ROS2 Integration**: Scalable and modular robot communication framework
- **Serial Communication**: Direct STM32/Arduino controller interface
- **Real-time Commands**: Sub-second response times for brain-to-robot control
- **Gripper Control**: Open/close commands for object manipulation

### 🛡️ **Safety & Reliability**
- **Emergency Stop**: Immediate motion halt via brain command or external trigger
- **Workspace Limits**: Configurable 3D boundaries prevent dangerous movements  
- **Joint Limits**: Hardware joint angle limits enforced in software
- **Signal Quality Monitoring**: Real-time EEG signal quality assessment
- **Artifact Detection**: Automatic identification and handling of EEG artifacts

### ⚙️ **System Management**
- **Flexible Configuration**: YAML-based configuration with hot-reload capability
- **Comprehensive Logging**: Detailed system monitoring and debugging tools
- **Performance Monitoring**: Real-time metrics for latency, accuracy, and throughput
- **Modular Architecture**: Easy to extend and customize for different applications
- **Cross-platform**: Linux, Windows, and macOS compatibility

## 🛠️ Tech Stack

### **Core Technologies**
- **Language**: Python 3.8+ with type hints and async support
- **Hardware**: OpenBCI (Cyton/Daisy boards), STM32/Arduino-based robotic arms
- **Robotics Framework**: ROS2 (Galactic/Humble) with custom nodes and services

### **Signal Processing & AI**
- **Machine Learning**: scikit-learn (Random Forest, SVM), custom feature extractors  
- **Signal Processing**: SciPy, NumPy, MNE-Python, PyWavelets
- **Digital Filters**: Butterworth, IIR notch, custom artifact detection
- **Feature Extraction**: Spectral analysis, Hjorth parameters, time-domain features

### **Communication & Integration**
- **Hardware Interface**: PySerial for STM32/Arduino communication
- **ROS2 Integration**: rclpy, geometry_msgs, sensor_msgs, custom message types
- **Configuration**: YAML/JSON with validation and hot-reload
- **Logging**: Python logging with rotating files and real-time monitoring

### **Development & Deployment**
- **Packaging**: setuptools with entry points and console scripts
- **Documentation**: Sphinx-compatible docstrings with comprehensive examples
- **Testing**: Built-in simulation modes and component testing
- **Cross-platform**: Linux (primary), Windows, macOS support

## 📁 Project Structure

```
bci-robotic-arm-controller/
├── openbci_reader/          # EEG data acquisition
│   ├── __init__.py
│   └── stream_data.py       # OpenBCI interface and data streaming
├── signal_processing/       # EEG signal processing and classification
│   ├── __init__.py
│   └── eeg_classifier.py    # Signal processing and ML classification
├── ros2_publisher/          # Robotic arm control
│   ├── __init__.py
│   └── arm_control_node.py  # ROS2 node for arm control
├── config/                  # Configuration files
│   ├── default_config.yaml  # Main system configuration
│   ├── arm_kinematics.yaml  # Robot kinematics parameters
│   ├── signal_processing.yaml # DSP and ML parameters
│   └── brain_commands.yaml  # Brain-to-command mapping
├── logs/                    # System logs
├── models/                  # Trained ML models
├── main.py                  # Main application entry point
├── utils.py                 # Utility functions and helpers
├── setup.py                 # Package installation script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Installation

#### **Prerequisites**
- Python 3.8 or higher
- pip package manager
- Git for cloning repository
- (Optional) ROS2 Galactic/Humble for full functionality

#### **Quick Install**
```bash
# Clone the repository
git clone https://github.com/ibrahimmediouni/bci-robotic-arm-controller.git
cd bci-robotic-arm-controller

# Create virtual environment (recommended)
python -m venv bci_env
source bci_env/bin/activate  # On Windows: bci_env\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Verify installation
python -c "import openbci_reader; print('✅ Installation successful')"
```

#### **Optional: ROS2 Installation**  
For full ROS2 functionality, install ROS2 separately:
```bash
# Ubuntu/Debian - ROS2 Galactic
sudo apt update && sudo apt install curl gnupg2 lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo apt-key add -
sudo sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'
sudo apt update
sudo apt install ros-galactic-desktop python3-colcon-common-extensions

# Source ROS2 environment
echo "source /opt/ros/galactic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 2. Hardware Setup

#### **OpenBCI EEG Setup** 🧠
```bash
# 1. Hardware Connection
# - Connect OpenBCI Cyton board via USB dongle  
# - Power on the board (blue LED should be solid)
# - Verify USB connection
ls /dev/ttyUSB* /dev/ttyACM*  # Should show device (e.g., /dev/ttyUSB0)

# 2. Electrode Placement (10-20 International System)
# Recommended 8-channel setup:
# - Fp1, Fp2 (frontal)      - F3, F4 (frontal) 
# - C3, C4 (central)        - P3, P4 (parietal)
# - Reference: right earlobe or mastoid
# - Ground: forehead (Fpz) or left earlobe

# 3. Test EEG connection
python openbci_reader/stream_data.py --port /dev/ttyUSB0 --duration 10
```

#### **Robotic Arm Setup** 🤖
```bash
# 1. Hardware Connection  
# - Connect STM32/Arduino-based arm controller via USB
# - Power on the robotic arm system
# - Verify USB connection
ls /dev/ttyACM* /dev/ttyUSB*  # Should show device (e.g., /dev/ttyACM0)

# 2. Safety Check
# - Ensure arm workspace is clear of obstacles
# - Verify emergency stop button is accessible  
# - Check joint limits and cable routing
# - Position arm in safe starting configuration

# 3. Test arm connection
python -c "
import serial
try:
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=2)
    ser.write(b'STATUS\\n')
    response = ser.readline().decode().strip()
    print(f'✅ Arm connected: {response}')
    ser.close()
except Exception as e:
    print(f'❌ Connection failed: {e}')
"
```

#### **Permissions Setup** (Linux/macOS)
```bash
# Add user to dialout group for serial access
sudo usermod -a -G dialout $USER
sudo chmod 666 /dev/ttyUSB* /dev/ttyACM*

# Restart terminal or logout/login for group changes to take effect
```

### 3. Configuration

#### **Generate Default Configuration**
```bash
# Create default configuration file
python utils.py --create-config config.yaml

# This creates a comprehensive config with all parameters documented
cat config.yaml  # Review the generated configuration
```

#### **Customize for Your Hardware**
```bash
# Edit main configuration
nano config.yaml

# Key settings to modify:
# - eeg_port: "/dev/ttyUSB0"        # Your OpenBCI port
# - arm_port: "/dev/ttyACM0"        # Your robotic arm port  
# - eeg_channels: 8                 # Number of EEG channels
# - confidence_threshold: 0.7       # Classification sensitivity
# - workspace_*_limits: [min, max]  # Safe movement boundaries

# Validate your configuration
python utils.py --validate-config config.yaml
```

### 4. System Test

#### **Complete System Test**
```bash
# Test all components with detailed output
python main.py --mode test

# Expected output:
# ✅ EEG streaming: 250.0 Hz, 8 channels
# ✅ Signal processing: Filters initialized  
# ✅ Classification: Model ready
# ✅ Robotic arm: Connected and operational
```

#### **Individual Component Tests**
```bash
# Test EEG acquisition only
python main.py --mode test --components eeg_only

# Test robotic arm only  
python main.py --mode test --components arm_only

# Test signal processing pipeline
python main.py --mode test --components processing_only

# Each test provides detailed diagnostics and performance metrics
```

### 5. Real-time Control

#### **Start BCI Control System**
```bash
# Run complete real-time BCI system
python main.py --mode realtime

# With custom configuration
python main.py --mode realtime --config my_config.yaml

# With specific components only
python main.py --mode realtime --components full
```

#### **Monitor System Status**
Open a second terminal to monitor the system:
```bash
# Watch system logs in real-time
tail -f logs/bci_system.log

# For ROS2 users - monitor topics
ros2 topic list
ros2 topic echo /brain_state
ros2 topic echo /arm_state
ros2 topic echo /system_status
```

## 🧪 Detailed Usage

### EEG Data Acquisition

```bash
# Test EEG streaming
python openbci_reader/stream_data.py --port /dev/ttyUSB0 --duration 30

# Stream with daisy chain (16 channels)
python openbci_reader/stream_data.py --daisy --duration 60
```

### Signal Processing and Classification

```bash
# Test signal processing
python signal_processing/eeg_classifier.py --mode demo --duration 30

# Train new model (requires training data)
python signal_processing/eeg_classifier.py --mode train --data-path ./training_data/
```

### Robotic Arm Control (ROS2)

```bash
# Initialize ROS2 environment
source /opt/ros/galactic/setup.bash  # or your ROS2 distribution

# Run arm control node
ros2 run bci_robotic_arm_controller arm_control_node

# Monitor arm state
ros2 topic echo /arm_state

# Send manual commands
ros2 topic pub /brain_commands std_msgs/String "data: 'focus'"
```

### Configuration Management

```bash
# Validate configuration
python utils.py --validate-config config.yaml

# Test data validation
python utils.py --test-validation
```

## 🎯 Brain Command Mapping

The system recognizes the following brain states and maps them to robotic actions:

| Brain State | Description | Arm Action | Training Protocol |
|-------------|-------------|------------|-------------------|
| `rest` | Relaxed, no task | Stop movement | Eyes closed, clear mind |
| `focus` | Concentrated attention | Move forward | Focus on specific object |
| `relax` | Deep relaxation | Move backward | Meditative state |
| `movement` | Motor imagery | Move up | Imagine hand movement |
| `left_hand` | Left hand imagery | Move left | Imagine left fist clench |
| `right_hand` | Right hand imagery | Move right | Imagine right fist clench |
| `open` | Gripper opening | Open gripper | Imagine opening hand |
| `close` | Gripper closing | Close gripper | Imagine closing hand |

## ⚙️ Configuration

### Main Configuration (`config/default_config.yaml`)

```yaml
# EEG Settings
eeg_sampling_rate: 250.0
eeg_channels: 8
eeg_port: "/dev/ttyUSB0"

# Signal Processing
filter_lowcut: 1.0
filter_highcut: 45.0
notch_frequency: 50.0

# Classification
confidence_threshold: 0.7
model_type: "rf"  # or "svm"

# Robotic Arm
arm_port: "/dev/ttyACM0"
arm_joints: 6
arm_step_size: 0.02

# Safety
workspace_x_limits: [0.1, 0.6]
workspace_y_limits: [-0.4, 0.4]
workspace_z_limits: [0.0, 0.5]
```

## 🔒 Safety Features

- **Emergency Stop**: Immediate motion halt via brain command or ROS2 topic
- **Workspace Limits**: Configurable 3D boundaries prevent dangerous movements
- **Joint Limits**: Hardware joint angle limits enforced in software
- **Velocity Limits**: Maximum joint and end-effector velocities
- **Signal Quality Monitoring**: Automatic detection of poor EEG signal quality
- **Artifact Detection**: Real-time identification and handling of EEG artifacts

## 📊 Performance Monitoring

The system provides comprehensive monitoring:

- **EEG Quality**: Real-time signal quality metrics per channel
- **Classification Performance**: Confidence scores and prediction accuracy
- **Processing Latency**: End-to-end system response times
- **Command Execution**: Success rates and error tracking
- **System Resources**: CPU and memory usage monitoring

## 🐛 Troubleshooting

### **Hardware Connection Issues**

#### **OpenBCI EEG Problems** 🧠
```bash
# Problem: Device not found
ls /dev/tty*  # List all serial devices
dmesg | grep tty  # Check kernel messages

# Problem: Permission denied
sudo chmod 666 /dev/ttyUSB*
sudo usermod -a -G dialout $USER  # Add user to dialout group
# Logout and login again

# Problem: Connection timeout
python -c "
import serial
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=5)
ser.write(b'v')  # Send version command
print('Response:', ser.readline().decode())
ser.close()
"

# Problem: No data streaming
# - Check OpenBCI battery level (>20%)
# - Verify USB dongle pairing (blue LED solid)
# - Try different USB port
# - Reset OpenBCI board (power cycle)
```

#### **Robotic Arm Issues** 🤖
```bash
# Problem: Arm not responding
# Check power supply and connections
python -c "
import serial
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=3)
ser.write(b'STATUS\\n')
response = ser.readline().decode().strip()
print(f'Arm status: {response}')
if 'READY' not in response:
    print('❌ Arm not ready - check power and firmware')
ser.close()
"

# Problem: Erratic movements
# - Verify joint calibration
# - Check for loose mechanical connections
# - Ensure proper workspace limits in config
# - Test manual control first

# Problem: Emergency stop activated
python main.py --mode test --components arm_only
# Look for error messages about joint limits or safety violations
```

### **Software & Configuration Issues**

#### **Python Environment Problems**
```bash
# Problem: Import errors
pip list | grep -E "(numpy|scipy|scikit-learn|serial)"
pip install --upgrade -r requirements.txt

# Problem: Module not found
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pip install -e .  # Reinstall in development mode

# Problem: Dependency conflicts
pip install --force-reinstall -r requirements.txt
# Or create a fresh virtual environment
```

#### **ROS2 Integration Issues**
```bash
# Problem: ROS2 not found
source /opt/ros/galactic/setup.bash  # Or your ROS2 distribution
echo $ROS_DISTRO  # Should show your ROS2 version

# Problem: Node communication failures
ros2 topic list  # Should show BCI topics
ros2 node list   # Should show bci_arm_control_node

# Problem: Message type errors
pip install --upgrade rclpy
colcon build --packages-select bci_robotic_arm_controller --symlink-install
```

#### **Configuration Problems**
```bash
# Problem: Invalid configuration
python utils.py --validate-config config.yaml
# This will show specific validation errors

# Problem: Port conflicts
sudo lsof -i :11411  # Check ROS2 port usage
ps aux | grep python  # Check for running instances

# Problem: Permission errors
chmod +x main.py
chmod -R 755 config/
```

### **Signal Quality & Performance Issues**

#### **EEG Signal Quality** 📊
```bash
# Problem: Poor signal quality
python openbci_reader/stream_data.py --duration 30
# Monitor signal quality output - should be >0.7 for good channels

# Troubleshooting steps:
# 1. Check electrode impedance (<5kΩ ideal, <25kΩ acceptable)
# 2. Apply more conductive gel
# 3. Clean electrode sites with alcohol
# 4. Ensure proper ground and reference connections
# 5. Minimize movement artifacts
# 6. Check for electrical interference (AC power, fluorescent lights)
```

#### **Classification Performance** 🎯
```bash
# Problem: Low classification accuracy
# Check confidence scores in real-time
python main.py --mode realtime
# Look for confidence values >0.7

# Improvement strategies:
# 1. Retrain model with more data
# 2. Adjust confidence threshold in config
# 3. Improve electrode placement
# 4. Reduce artifacts through better setup
# 5. Use more training sessions per mental state
```

#### **System Performance** ⚡
```bash
# Problem: High latency
# Monitor processing times
htop  # Check CPU usage
python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
"

# Optimization tips:
# - Reduce EEG buffer size for lower latency
# - Use faster machine learning models
# - Optimize digital filter parameters
# - Close unnecessary applications
# - Use SSD for faster data access
```

### **Debug Mode & Diagnostics**

#### **Enable Detailed Logging**
```bash
# Create debug configuration
cp config/default_config.yaml debug_config.yaml
# Edit debug_config.yaml:
# log_level: "DEBUG"
# log_file: "logs/debug.log"

# Run with debug output
python main.py --mode realtime --config debug_config.yaml

# Monitor detailed logs
tail -f logs/debug.log | grep -E "(ERROR|WARNING|DEBUG)"
```

#### **Component-Specific Debugging**
```bash
# Debug EEG acquisition
python openbci_reader/stream_data.py --port /dev/ttyUSB0 --duration 60
# Should show consistent sample rates and signal quality

# Debug signal processing
python signal_processing/eeg_classifier.py --mode demo --duration 30
# Should show feature extraction and classification results

# Debug robotic arm
python -c "
from ros2_publisher.arm_control_node import RobotController
controller = RobotController()
if controller.connect():
    print('✅ Arm connected successfully')  
    status = controller.get_state()
    print(f'Position: {status.end_effector_position}')
    print(f'Moving: {status.is_moving}')
    print(f'Error: {status.error_state}')
else:
    print('❌ Arm connection failed')
"
```

#### **Performance Profiling**
```bash
# Profile system performance
python -m cProfile -o profile.stats main.py --mode test
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"

# Monitor real-time performance
python -c "
import time
import psutil
while True:
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    print(f'CPU: {cpu:5.1f}% | Memory: {mem:5.1f}%', end='\\r')
    time.sleep(1)
"
```

### **Getting Help**

If you're still experiencing issues:

1. **Check System Requirements**: Ensure Python 3.8+, sufficient RAM (4GB+), and USB ports
2. **Review Logs**: Check `logs/bci_system.log` for detailed error messages  
3. **Test Components Individually**: Use `--components` flag to isolate issues
4. **Hardware Verification**: Test OpenBCI and robotic arm with their native software first
5. **Community Support**: Check GitHub issues and discussions for similar problems

#### **Reporting Bugs**
When reporting issues, please include:
```bash
# System information
python --version
pip list | grep -E "(numpy|scipy|sklearn|serial)"
uname -a  # Linux/macOS
lsusb     # USB devices

# Configuration
cat config.yaml

# Recent logs
tail -50 logs/bci_system.log

# Error reproduction
python main.py --mode test 2>&1 | tee error_output.txt
```

## 🧪 Development and Testing

### Running Tests

```bash
# System integration test
python main.py --mode test

# Component-specific tests
python -m pytest tests/  # (if test suite exists)
```

### Data Collection for Training

```bash
# Collect training data
python main.py --mode collect --output ./training_data/ --duration 300

# Follow on-screen prompts for different mental states
```

### Model Training

```bash
# Train new classification model
python main.py --mode train --data ./training_data/

# Evaluate model performance
python signal_processing/eeg_classifier.py --mode evaluate --model ./models/eeg_model.pkl
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with proper documentation
4. Add tests if applicable
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenBCI community for hardware and software tools
- ROS2 development team for robotics framework
- MNE-Python team for EEG processing tools
- All contributors and testers who helped improve this system

## 📞 Support

For questions, issues, or contributions:

- **Issues**: [GitHub Issues](https://github.com/ibrahimmediouni/bci-robotic-arm-controller/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ibrahimmediouni/bci-robotic-arm-controller/discussions)
- **Email**: ibrahim.mediouni@example.com

## ⚡ Performance Optimization

### **System Requirements**

#### **Minimum Requirements**
- **OS**: Linux (Ubuntu 20.04+), Windows 10, macOS 10.15+
- **Python**: 3.8 or higher with pip
- **RAM**: 4GB (8GB recommended for optimal performance)
- **CPU**: Dual-core 2.0GHz (Quad-core recommended)  
- **Storage**: 2GB free space (SSD recommended)
- **USB**: 2+ USB ports for OpenBCI and robotic arm

#### **Recommended Requirements**
- **OS**: Linux Ubuntu 22.04 LTS (best compatibility)
- **Python**: 3.9 or 3.10 with virtual environment
- **RAM**: 8GB+ for multiple concurrent users
- **CPU**: Quad-core 3.0GHz+ (Intel i5/AMD Ryzen 5 equivalent)
- **Storage**: 10GB+ on SSD for fast data processing
- **GPU**: Optional for future deep learning features

### **Performance Tuning**

#### **Real-time Optimization**
```yaml
# config/performance_config.yaml - Optimized for low latency
eeg_sampling_rate: 250.0        # Standard rate, good balance
classification_window: 1.5      # Reduced for faster response
confidence_threshold: 0.65      # Slightly lower for responsiveness  
buffer_size: 1024              # Smaller buffer = lower latency
max_threads: 4                 # Match your CPU cores
```

#### **High Accuracy Configuration**
```yaml
# config/accuracy_config.yaml - Optimized for precision
eeg_sampling_rate: 500.0        # Higher sampling for better signals
classification_window: 3.0      # Longer window for more features
confidence_threshold: 0.8       # Higher threshold for reliability
buffer_size: 4096              # Larger buffer for stability
filter_order: 6                # More sophisticated filtering
```

#### **System Optimization Tips**
```bash
# Linux performance tuning
# 1. Increase real-time priority
sudo echo '@audio - rtprio 99' >> /etc/security/limits.conf
sudo echo '@audio - memlock unlimited' >> /etc/security/limits.conf

# 2. Disable CPU frequency scaling
sudo cpupower frequency-set --governor performance

# 3. Optimize USB for real-time
echo 'SUBSYSTEM=="usb", ATTR{power/autosuspend}="-1"' | sudo tee /etc/udev/rules.d/50-usb-realtime.rules

# 4. Set process priorities
nice -n -10 python main.py --mode realtime  # Higher priority
```

### **Monitoring & Metrics**

#### **Real-time Performance Dashboard**
```bash
# Start performance monitoring
python -c "
import time, psutil, threading
from collections import deque

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'cpu': deque(maxlen=60),
            'memory': deque(maxlen=60), 
            'latency': deque(maxlen=60)
        }
        
    def monitor(self):
        while True:
            self.metrics['cpu'].append(psutil.cpu_percent())
            self.metrics['memory'].append(psutil.virtual_memory().percent)
            
            # Clear screen and display metrics
            print('\\033[2J\\033[H')  # Clear screen
            print('🔥 BCI Performance Monitor')
            print('=' * 40)
            print(f'CPU Usage:     {self.metrics[\"cpu\"][-1]:6.1f}%')
            print(f'Memory Usage:  {self.metrics[\"memory\"][-1]:6.1f}%')
            print(f'Avg CPU (1m):  {sum(self.metrics[\"cpu\"])/len(self.metrics[\"cpu\"]):6.1f}%')
            print(f'Avg Mem (1m):  {sum(self.metrics[\"memory\"])/len(self.metrics[\"memory\"]):6.1f}%')
            
            time.sleep(1)

monitor = PerformanceMonitor()
monitor.monitor()
" &

# Run your BCI system
python main.py --mode realtime
```

## 🔮 Future Enhancements

### **Short-term Goals (6 months)**
- [ ] **Multi-user Profiles**: Individual calibration and model storage
- [ ] **Advanced Preprocessing**: Automatic artifact rejection with ICA
- [ ] **Model Auto-tuning**: Hyperparameter optimization for user-specific models
- [ ] **Web Dashboard**: Real-time monitoring and control interface
- [ ] **Mobile App**: Remote monitoring and emergency controls

### **Medium-term Goals (1 year)**
- [ ] **Deep Learning Models**: CNN/RNN-based classification for improved accuracy
- [ ] **Adaptive Filtering**: Real-time filter adjustment based on signal quality
- [ ] **VR Integration**: Virtual reality visualization and training environments
- [ ] **Cloud Deployment**: Scalable cloud-based processing and storage
- [ ] **Multi-arm Support**: Control multiple robotic arms simultaneously

### **Long-term Vision (2+ years)**
- [ ] **Brain Plasticity Adaptation**: Long-term learning and user adaptation
- [ ] **Invasive BCI Support**: Integration with implanted electrode systems
- [ ] **Haptic Feedback**: Full sensory feedback loop for immersive control
- [ ] **AI Co-pilot**: Intelligent assistance and movement prediction
- [ ] **Standardized API**: Universal BCI-robotics interface protocol

### **Research Collaborations**
We welcome partnerships with:
- **Universities**: Academic research and student projects
- **Medical Centers**: Assistive technology and rehabilitation applications  
- **Robotics Companies**: Integration with commercial robotic platforms
- **BCI Researchers**: Advanced signal processing and machine learning techniques

### **Community Contributions**
Priority areas for community development:
- **New Hardware Support**: Additional EEG devices and robotic platforms
- **Algorithm Improvements**: Better classification and filtering methods
- **Documentation**: Tutorials, examples, and best practices
- **Testing**: Comprehensive test suites and validation protocols
- **Localization**: Multi-language support and regional adaptations

---

*Ready to revolutionize human-robot interaction? Join our growing community of BCI researchers, roboticists, and innovators building the future of neural control!*
