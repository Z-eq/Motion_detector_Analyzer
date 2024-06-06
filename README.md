# Motion_detector_Analyzer
This project uses OpenCV and YOLO (You Only Look Once) to detect humans and animals in video files. It identifies objects like "person, or animals and saves the screenshots and the tagname of video.

## Features

- Human and animal detection in videos that has been recorded which you want to analyze if there was an event without need of looking through all videos.
- Motion detection using background subtraction
- Configurable frame skipping for performance tuning 
- Logging of detection results and processed videos
- Good for analyzing survelliance cameras which has no motiondetection.

## Installation

1. **Open Command Prompt**: Press `Win + R`, type `cmd`, and press Enter.
2. **Navigate to Project Directory**: 

    cd /your_path/
   
3. **Create Virtual Environment (Optional but Recommended)**:
   
    python -m venv venv
    venv\Scripts\activate
  
4. **Install cv2 (opencv-python)

    pip install opencv-python
   
6. **Run the Script**:
    
    python motion_detection.py
   
Ensure YOLO model files (weights, config, class names) are in the project directory.. If not you can download them manually using the links below.
"yolov3.weights":"https://pjreddie.com/media/files/yolov3.weights"
"yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
"coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

Or use the script download_yolo.py that is located in the root directory, it will automaticly download the weights,cfg adn coco names.
