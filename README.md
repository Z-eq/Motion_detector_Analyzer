# Motion_detector_Analyzer
This project uses OpenCV and YOLO (You Only Look Once) to detect humans and animals in video files. It identifies objects like "person, or animals and saves the screenshots and the tagname of video.

## Requirements

-   Python 3.x
-   OpenCV
-   NumPy

## Installation

1.  **Clone the repository:**

    `git clone https://github.com/YOUR_USERNAME/Motion_detector_Analyzer.git
    cd Motion_detector_Analyzer` 
    
2.  **Install the required packages:**
    
        
    `pip install -r requirements.txt` 
    
3.  **Download YOLO weights:**
    
    `python download_yolo.py` 
    

## Usage

To use this script, you need to prepare your video files and place them in a directory. You will also need the YOLO configuration and weights files (`yolov3.cfg`, `yolov3.weights`) and the COCO names file (`coco.names`).

### Parameters

-   **video_directory**: The path to your directory containing video files.
-   **output_directory**: The directory where detected frames will be saved.
-   **log_file_path**: The path to the log file where processed videos will be recorded.
 
-   **frame_skip**: Number of frames to skip while processing. Lower value means more frames are processed (default is 25).
-   **confidence_threshold**: Minimum confidence for detections. Adjust this based on how confident you want the model to be before accepting a detection (default is 0.6).
-   **nms_threshold**: Non-max suppression threshold. Adjust this based on how much overlap you allow between bounding boxes before they are considered the same object (default is 0.4).
-   **batch_size**: Number of videos to process in each batch (default is 5).

### Example Command

`python motion_detection.py` 

### Sample Configuration

Edit the script to set your paths and parameters:

```py
`if __name__ == "__main__":
    # Paths to the YOLO files
    weights_path = "yolov3.weights"
    config_path = "yolov3.cfg"
    names_path = "coco.names"

    # Directory paths
    video_directory = "/path/to/your/videos"
    output_directory = "/path/to/output"
    log_file_path = "/path/to/output/PROCESSED_Videos.log"
    
    # Parameters
    frame_skip = 5
    confidence_threshold = 0.6
    nms_threshold = 0.4
    batch_size = 5
    
    # Process videos and log results
    process_videos(video_directory, output_directory, weights_path, config_path, names_path, log_file_path, frame_skip, confidence_threshold, nms_threshold, batch_size)` 
```

### Functions

-   **load_yolo(weights_path, config_path, names_path)**: Loads the YOLO model with given configuration, weights, and class names.
-   **detect_objects(frame, net, output_layers)**: Detects objects in a single frame.
-   **get_bounding_boxes(outs, width, height, classes, confidence_threshold=0.5, nms_threshold=0.4)**: Gets bounding boxes for detected objects.
-   **detect_motion(frame, background_subtractor)**: Detects motion in the frame using background subtraction.
-   **detect_humans_and_animals(video_path, weights_path, config_path, names_path, frame_skip=15, confidence_threshold=0.5, nms_threshold=0.4)**: Detects humans and animals in a video.
-   **process_single_video(args)**: Processes a single video file.
-   **process_videos(video_directory, output_directory, weights_path, config_path, names_path, log_file_path, frame_skip=15, confidence_threshold=0.5, nms_threshold=0.4, batch_size=5)**: Processes videos in batches.

## Notes

-   Adjust the `frame_skip`, `confidence_threshold`, and `nms_threshold` parameters based on your requirements for accuracy and performance.
-   Ensure that the output directory exists or will be created by the script.
-   The script uses parallel processing to speed up video processing. Adjust the `batch_size` to fit your system's capabilities.


You an contact me if you have questions  pm or mail zeq.alidemaj @ gmail.com
