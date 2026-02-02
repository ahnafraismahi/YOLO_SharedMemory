# YOLO Human Detection - Shared Memory Bridge

This project demonstrates a cross-process communication system. It uses **C++ (OpenCV DNN)** to detect humans in an image and sends that data to **Python** via **Windows Shared Memory**.

## 📥 GitHub Clone
To download this project to your computer, use the following commands in your terminal:

```bash
git clone [https://github.com/LearnerMahi/YOLO_SharedMemory](https://github.com/LearnerMahi/YOLO_SharedMemory)
cd YOLO_SharedMemory
🛠 Project Components

    Producer (C++):

        Loads a YOLOv5 ONNX model.

        Performs inference on a target image.

        Writes image pixels and bounding box coordinates into Windows Shared Memory.

    Consumer (Python):

        Attaches to the shared memory.

        Reads raw bytes and reconstructs the image using NumPy.

        Displays the final result with bounding boxes.

🚀 Setup & Installation
1. Saving the Files

    Save the C++ code as: main.cpp

    Save the Python code as: reader.py

    Save this text as: README.md

2. Visual Studio Configuration (C++)

    Include Directories: Add your OpenCV build/include path.

    Library Directories: Add your OpenCV build/x64/vc16/lib path.

    Linker Input: Add opencv_worldXXX.lib (replace XXX with your version).

    Build Configuration: Set to x64.

3. Python Requirements

Install the necessary libraries via your terminal or command prompt:
Bash

pip install numpy opencv-python

📋 Running the Detection

    Compile and Run C++: Open your terminal in the folder where your .exe is located and run: YourProject.exe <image_path> <model_path> (Example: Detection.exe test.jpg yolov5s.onnx)

    Keep C++ Open: The C++ program will say "Press Enter to close shared memory." Do not press Enter yet, or the memory will be deleted.

    Run Python Reader: In a second terminal window, run: python reader.py

⚠️ Important Notes

    Shared Memory Name: Both C++ and Python must use the exact same name: YOLODetectionSharedMemory.

    Data Alignment: If you change the image size or box structure, you must update the constants in both files to prevent crashes.
