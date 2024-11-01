## Train YOLOv11 Model on  a Custom Dataset 
from ultralytics import YOLO
import cv2
import time
import os
from deep_sort_realtime.deepsort_tracker import DeepSort

#*******************************************************************
def detection(model, image, confidence: int = 0.7, allowed_classes : list = [1,2]):
    """
    Performs object detection on an input image using a YOLO model, with specified confidence threshold
    and class filtering.
    Args:
        model: The YOLO model object used for prediction, loaded with ultralytics.
        image: The input image for object detection, which can be a file path, URL, or image array.
        confidence (float): Confidence threshold for detections, with a default of 0.7. Only detections 
                            with confidence scores above this threshold will be considered.
        allowed_classes (list): List of class IDs to include in the detection. Only objects belonging to 
                                these classes will be returned in the output. Default is [1, 2].
                                Label of classes are : ball= 0,  goalkeeper=1, player=2, referee = 3  


    Returns:
        list: A list of detected objects, where each object is represented as a tuple:
            - Bounding box coordinates (x, y, width, height) as a list of integers.
            - Class ID (int): The numerical class identifier for the detected object.
            - Confidence score (float): The confidence score of the detection.
        
    Notes:
        - `result.boxes` provides bounding box coordinates as box.xyxy (in both `xyxy` and `xywh` format), confidence scores as box.conf,
          and class IDs as box.cls. 
          The `xyxy` format includes the top-left and bottom-right coordinates of each bounding box.
        - Only the specified `allowed_classes` are included in the returned detections list.
        
        - Coordinates (box.xyxy):  The bounding box coordinates are given as [x1, y1, x2, y2], where 
        - x1, y1 are the top-left corner coordinates while x2, y2 are the bottom-right corner coordinates.
        
        - Confidence Score (box.conf): This is the model's confidence level for that bounding box, typically ranging from 0 to 1.
        - Class ID (box.cls): The predicted class ID for that box, indicating which class the box corresponds to based on the model's training.

    """
    results= model.predict(source= image, conf=0.7, save= False,  classes = allowed_classes)
    result = results[0]
    # result.boxes : Boxes object for bounding box outputs
    # result.masks : Masks object for segmentation masks outputs
    # result.keypoints : Keypoints object for pose outputs
    # result.probs  : Probs object for classification outputs
    # result.obb  : Oriented boxes object for OBB outputs
    boxes=result.boxes
    detection=[]
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2-x1, y2-y1
        class_number = int(box.cls[0])
        conf = box.conf[0]
        detection.append(  ( ([x1,y1, w, h]), class_number, conf ) )
    return detection

#*******************************************************************
def track( object_tracker, detections, frame):
    """
    Tracks detected objects across frames using the DeepSort object tracker.

    Args:
        object_tracker (DeepSort): An instance of the DeepSort tracker configured with desired parameters.
                                   This tracker maintains unique identities for objects across frames.
        detections (list): A list of detected objects from the current frame. Each detection should be a tuple 
                           containing bounding box coordinates and relevant information in the format:
                           (bbox, class_id, confidence).
        frame: The current frame (image) used for tracking, typically a BGR image as expected by DeepSort.


    Returns:
        tuple: A tuple containing:
            - tracking_ids (list): A list of unique tracking IDs for each confirmed object track.
            - boxes (list): A list of bounding boxes (as lists of coordinates in `[left, top, right, bottom]` format)
                            for each confirmed track, where each bounding box corresponds to the track ID at the same index.

    Notes:
        - Only confirmed tracks are returned, which have met DeepSort's criteria for reliability.
        - The `update_tracks` method in DeepSort takes detections and the current frame, updates internal tracking states,
          and returns a list of active tracks.
        - The `to_ltrb` method is used to retrieve each track's bounding box in `[left, top, right, bottom]` format.

    """
    tracks = object_tracker.update_tracks(detections, frame= frame)
    tracking_ids = []
    boxes = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        tracking_ids.append(track.track_id)
        ltrb = track.to_ltrb()
        boxes.append(ltrb)
    return tracking_ids, boxes

#*******************************************************************
def main():
    ##  Load yolo11 model
    model = YOLO("yolo11m.pt")
    Home= os.getcwd()

    ##  Fin-tuning the model 
    # Football_Players_Detection_Dataset has been downloaded from  https://universe.roboflow.com/
    # download it and put it in datasets folder
    model.train(data=f'{Home}/datasets/Football_Players_Detection_Dataset/data.yaml',  epochs = 50, imgsz=640, lr0 = 1e-3, device = 0, batch = 8, workers= 0)

    ## Test the performance of the fine-tuned YOLOm 11 model
    trained_model = YOLO(f'{Home}/runs/detect/train/weights/best.pt')
    trained_model.predict(source= f'{Home}/datasets/Football_Players_Detection_Dataset/test/images', conf=0.7, save= True, classes = [1,2])

    # Test the performance of the fine-tuned model on a video
    object_tracker = DeepSort(
            max_age=20,
            n_init=2,
            nms_max_overlap=0.3,
            max_cosine_distance=0.8,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )

    VIDEO_PATH = f'{Home}/football.mp4'
    OUTPUT_VIDEO_PATH = f'{Home}/output_football.mp4'
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        exit()

    # Video Writer settings
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    frame_count = 0  # Initialize frame counter
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start_time = time.perf_counter()
        # Save the original frame before processing
        original_frame = frame.copy()  # Make a copy of the original frame
        
        # Run detection and tracking
        detections = detection(trained_model, frame, confidence=0.7, allowed_classes=[1,2])
        tracking_ids, boxes = track(object_tracker, detections, frame)
        
        # Draw bounding boxes and tracking IDs
        for tracking_id, bounding_box in zip(tracking_ids, boxes):
            cv2.rectangle(frame, (int(bounding_box[0]), int(bounding_box[1])), 
                        (int(bounding_box[2]), int(bounding_box[3])), (0, 0, 255), 2)
            cv2.putText(frame, f"{str(tracking_id)}", (int(bounding_box[0]), int(bounding_box[1] - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save original frame and output frame as .jpg
        if frame_count % 30 == 0:  # Change 30 to save every N frames
            original_frame_path = f"{Home}/output_frames/original_frame_{frame_count}.jpg"
            output_frame_path = f"{Home}/output_frames/output_frame_{frame_count}.jpg"
            
            # Create a directory to save the frames if it doesn't exist
            os.makedirs(f"{Home}/output_frames", exist_ok=True)

            # Save the original frame
            cv2.imwrite(original_frame_path, original_frame)  # Save the output frame (with bounding boxes)
            cv2.imwrite(output_frame_path, frame)  # Save the original frame
    
        # Save frame to output video
        out.write(frame)
    
        # Calculate and print FPS
        end_time = time.perf_counter()
        fps = 1 / (end_time - start_time)
        print(f"Current fps: {fps}")
        
        frame_count += 1  # Increment the frame count

    cap.release()
    out.release()
#*******************************************************************    
if  __name__ == "__main__":
    main()
