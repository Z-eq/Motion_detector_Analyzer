import cv2
import os
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_yolo(weights_path: str, config_path: str, names_path: str) -> Tuple[cv2.dnn_Net, List[str], List[str]]:
   
    try:
        net = cv2.dnn.readNet(weights_path, config_path)
        with open(names_path, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return net, classes, output_layers
    except Exception as e:
        logging.error(f"Failed to load YOLO model: {e}")
        raise

def detect_objects(frame: cv2.Mat, net: cv2.dnn_Net, output_layers: List[str]) -> Tuple[List[cv2.Mat], int, int]:
    """
    Detect objects in a frame using YOLO.
    
    Parameters:
    - frame: The input frame for object detection.
    - net: Loaded YOLO network.
    - output_layers: Names of the output layers.
    
    Returns:
    - outs: YOLO detection results.
    - width: Frame width.
    - height: Frame height.
    """
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    return outs, width, height

def get_bounding_boxes(outs: List[cv2.Mat], width: int, height: int, classes: List[str], 
                       confidence_threshold: float = 0.5, nms_threshold: float = 0.4) -> Tuple[List[int], List[float], List[List[int]]]:
    """
    Extract bounding boxes from YOLO detection results.
    
    Parameters:
    - outs: Detection results.
    - width: Frame width.
    - height: Frame height.
    - classes: List of class names.
    - confidence_threshold: Minimum confidence for a detection to be considered valid.
    - nms_threshold: Threshold for non-max suppression to remove redundant boxes.
    
    Returns:
    - final_class_ids: List of class IDs for final detections.
    - final_confidences: List of confidence scores for final detections.
    - final_boxes: List of bounding boxes for final detections.
    """
    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]
            if confidence > confidence_threshold and classes[class_id] in ["person", "cat", "dog", "bird"]:
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression to remove redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    final_boxes, final_class_ids, final_confidences = [], [], []
    for i in indices:
        final_boxes.append(boxes[i])
        final_class_ids.append(class_ids[i])
        final_confidences.append(confidences[i])
    
    return final_class_ids, final_confidences, final_boxes

def detect_motion(frame: cv2.Mat, background_subtractor: cv2.BackgroundSubtractor) -> bool:
    """
    Detect motion in a frame using background subtraction.
    
    Parameters:
    - frame: Video frame.
    - background_subtractor: Background subtractor object.
    
    Returns:
    - bool: True if motion is detected, False otherwise.
    """
    # Apply Gaussian blur to reduce noise
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    fgmask = background_subtractor.apply(blurred_frame)
    _, fgmask = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)
    fgmask = cv2.erode(fgmask, None, iterations=2)
    fgmask = cv2.dilate(fgmask, None, iterations=2)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Adjusted threshold for significant motion
            return True
    return False

def detect_humans_and_animals(video_path: str, net: cv2.dnn_Net, output_layers: List[str], classes: List[str], 
                              background_subtractor: cv2.BackgroundSubtractor, frame_skip: int = 15, 
                              confidence_threshold: float = 0.5, nms_threshold: float = 0.4) -> Tuple[List[Tuple[int, List[List[int]]]], List[Tuple[int, cv2.Mat]]]:
    """
    Detect humans and animals in a video.
    
    Parameters:
    - video_path: Path to the video file.
    - net: Loaded YOLO network.
    - output_layers: Names of output layers.
    - classes: List of class names.
    - background_subtractor: Background subtractor object.
    - frame_skip: Number of frames to skip between detections.
    - confidence_threshold: Minimum confidence for detections.
    - nms_threshold: Threshold for non-max suppression.
    
    Returns:
    - detections: List of frames with detected objects.
    - output_frames: List of frames with annotations.
    """
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    detections = []
    output_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        if frame_number % frame_skip != 0:
            continue

        if detect_motion(frame, background_subtractor):
            outs, width, height = detect_objects(frame, net, output_layers)
            class_ids, confidences, boxes = get_bounding_boxes(outs, width, height, classes, confidence_threshold, nms_threshold)

            if boxes:
                detections.append((frame_number, boxes))
                for i, box in enumerate(boxes):
                    x, y, w, h = box
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                output_frames.append((frame_number, frame))
                # Log detection to console
                logging.info(f"Detected {len(boxes)} objects in frame {frame_number}")

    cap.release()
    return detections, output_frames

def process_videos(video_directory: str, output_directory: str, net: cv2.dnn_Net, output_layers: List[str], 
                   classes: List[str], log_file_path: str, frame_skip: int = 15, 
                   confidence_threshold: float = 0.5, nms_threshold: float = 0.4) -> None:
    """
    Process all videos in a directory for human and animal detection and log processed videos.
    
    Parameters:
    - video_directory: Directory containing video files.
    - output_directory: Directory to save frames with detections.
    - net: Loaded YOLO network.
    - output_layers: Names of output layers.
    - classes: List of class names.
    - log_file_path: Path to the log file for recording processed videos.
    - frame_skip: Number of frames to skip between detections.
    - confidence_threshold: Minimum confidence for detections.
    - nms_threshold: Threshold for non-max suppression.
    """
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
    video_files = [f for f in os.listdir(video_directory) if f.endswith(".mp4")]

    with open(log_file_path, 'w') as log_file:
        for video_file in video_files:
            video_path = os.path.join(video_directory, video_file)
            detections, output_frames = detect_humans_and_animals(video_path, net, output_layers, classes, background_subtractor, frame_skip, confidence_threshold, nms_threshold)
            
            if detections:
                output_path = os.path.join(output_directory, os.path.splitext(video_file)[0])
                os.makedirs(output_path, exist_ok=True)
                for frame_number, frame in output_frames:
                    output_frame_path = os.path.join(output_path, f"frame_{frame_number}.jpg")
                    cv2.imwrite(output_frame_path, frame)
                logging.info(f"Detections in {video_file}: {detections}")

            log_file.write(f"{video_file}\n")
            logging.info(f"Processed video: {video_file}")

if __name__ == "__main__":
    # Paths to the YOLO files
    weights_path = "yolov3.weights"
    config_path = "yolov3.cfg"
    names_path = "coco.names"

    # Load YOLO
    net, classes, output_layers = load_yolo(weights_path, config_path, names_path)
    
    # Directory paths
    video_directory = "/folder_of_you_videos"  # Path to your videos directory
    output_directory = "/your_output_folder"  # Directory to save detected frames
    log_file_path = "/your_output_folder_/PROCESSED_Videos.log"  # Log file to record processed videos
    
    # Create output directory if not exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Parameters
    frame_skip = 25 # Process every 25th frame ( The lower frames the more accurate but slower)
    confidence_threshold = 0.6  # Minimum confidence for detections (adjust this for sensitivity)
    nms_threshold = 0.4  # Non-max suppression threshold (adjust this to remove noise)
    
    # Process videos and log results
    process_videos(video_directory, output_directory, net, output_layers, classes, log_file_path, frame_skip, confidence_threshold, nms_threshold)
