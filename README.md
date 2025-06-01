Vehicle Detection and Tracking Application
Overview
This project, developed by Group 5, is a real-time object detection and tracking application using the YOLOv8 model. It is designed for tasks such as road safety, surveillance, and medical assistance, focusing on detecting and tracking vehicles (e.g., ambulances, cars, buses). The application features an interactive web interface built with Streamlit, video processing with OpenCV, and heatmap generation for visualization.
Features

Real-time detection and tracking of objects in videos.
Customizable selection of object classes (e.g., cars, ambulances).
Optional emergency vehicle detection based on color and audio cues.
Generation of heatmaps and downloadable PDF reports/CSV logs.
Support for uploaded videos and sample videos.

Installation
Prerequisites

Python 3.8 or higher
Git (for cloning the repository)

Steps

Clone the repository:git clone https://github.com/ton-utilisateur/your-depot.git
cd your-depot


Install the required dependencies:pip install -r requirements.txt


Ensure the YOLOv8 model file (yolov8n.pt) is in the project directory or adjust the path in detector.py.

Usage

Run the application:streamlit run app.py


Use the web interface to:
Upload a video or select a sample video.
Choose object classes to detect.
Enable/disable emergency mode.
View real-time results and download reports.



Dependencies

streamlit
opencv-python
ultralytics
scipy
numpy
reportlab
pandas

List is included in requirements.txt.
Contributors

Anna NIANE
Ousmane TIENTA
Fatou Kine TOURE

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Future Work

Optimize performance for low-power devices.
Enhance tracking robustness with advanced algorithms.
Add critical event detection (e.g., accidents).

