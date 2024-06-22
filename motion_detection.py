import cv2
import os
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_yolo(weights_path, config_path, names_path):
    try:
        net = cv2.dnn.readNet(weights_path, config_path)
        #IMPROVED: Use GPU if available
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        with open(names_path, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return net, classes, output_layers
    except Exception as e:
        logging.error(f"Failed to load YOLO model: {e}")
        raise

def detect_objects(frame, net, output_layers):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    return outs, width, height

def get_bounding_boxes(outs, width, height, classes, confidence_threshold=0.5, nms_threshold=0.4):
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

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    final_boxes, final_class_ids, final_confidences = [], [], []
    for i in indices:
        final_boxes.append(boxes[i])
        final_class_ids.append(class_ids[i])
        final_confidences.append(confidences[i])
    
    return final_class_ids, final_confidences, final_boxes

def detect_motion(frame, background_subtractor):
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    fgmask = background_subtractor.apply(blurred_frame)
    _, fgmask = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)
    fgmask = cv2.erode(fgmask, None, iterations=2)
    fgmask = cv2.dilate(fgmask, None, iterations=2)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            return True
    return False

def detect_humans_and_animals(video_path, weights_path, config_path, names_path, frame_skip=15, confidence_threshold=0.5, nms_threshold=0.4):
    net, classes, output_layers = load_yolo(weights_path, config_path, names_path)
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
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
                print(f"Detected {len(boxes)} objects in frame {frame_number}")

    cap.release()
    return detections, output_frames

def process_single_video(args):
    video_file, video_directory, output_directory, weights_path, config_path, names_path, frame_skip, confidence_threshold, nms_threshold = args
    video_path = os.path.join(video_directory, video_file)
    detections, output_frames = detect_humans_and_animals(video_path, weights_path, config_path, names_path, frame_skip, confidence_threshold, nms_threshold)
    
    if detections:
        output_path = os.path.join(output_directory, os.path.splitext(video_file)[0])
        os.makedirs(output_path, exist_ok=True)
        for frame_number, frame in output_frames:
            output_frame_path = os.path.join(output_path, f"frame_{frame_number}.jpg")
            cv2.imwrite(output_frame_path, frame)
        print(f"Detections in {video_file}: {detections}")
    
    return video_file

def process_videos(video_directory, output_directory, weights_path, config_path, names_path, log_file_path, frame_skip=15, confidence_threshold=0.5, nms_threshold=0.4, batch_size=5):
    video_files = [f for f in os.listdir(video_directory) if f.endswith(".mp4")]
    
    def process_batch(batch):
        args = [(video_file, video_directory, output_directory, weights_path, config_path, names_path, frame_skip, confidence_threshold, nms_threshold) for video_file in batch]
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_single_video, arg) for arg in args]
            for future in as_completed(futures):
                video_file = future.result()
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f"{video_file}\n")
                print(f"Processed video: {video_file}")

    # Process videos in batches
    for i in range(0, len(video_files), batch_size):
        batch = video_files[i:i + batch_size]
        process_batch(batch)

if __name__ == "__main__":
    # Paths to the YOLO files
    weights_path = "yolov3.weights"
    config_path = "yolov3.cfg"
    names_path = "coco.names"

    # Directory paths
    video_directory = "/YOUR_Folder_TO_videos"  # Path to your videos directory
    output_directory = "/YOUR_detects_OUTOUT"  # Directory to save detected frames
    log_file_path = "/YOUR_LOG_FOLDER/PROCESSED_Videos.log"  # Log file to record processed videos
    
    # Create output directory if not exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Parameters
    frame_skip = 25 # Process every 5th frame (The lower frames the more accurate but slower)
    
    confidence_threshold = 0.6  # Minimum confidence for detections, Adjust this based on how confident you want the model to be before accepting a detection.
    
    nms_threshold = 0.4  # Non-max suppression threshold, Adjust this based on how much overlap you allow between bounding boxes before they are considered the same object.
    
    batch_size = 5  # Number of videos to process in each batch, 
    
    # Process videos and log results
    process_videos(video_directory, output_directory, weights_path, config_path, names_path, log_file_path, frame_skip, confidence_threshold, nms_threshold, batch_size)
