import os
import cv2
import numpy as np
import sys
import importlib.util
import pytesseract
import time
import threading
import re

# CONFIG START HERE
# Set model directory and video path
MODE=2 # 1=Webcam Capture, 2=Video Capture
IPCAMERAPATH = "http://192.168.0.159:8080/video"
MODEL_NAME = "/home/adrian/Desktop/ProjektInzynierski/custom_model_lite"
min_conf_threshold = 0.5
if MODE==2:
    VIDEO_NAME = "/home/adrian/Desktop/ProjektInzynierski/test1.mp4"
if MODE==1:
    imW, imH = 640, 480  # Desired webcam resolution
mainPath="/home/adrian/Desktop/ProjektInzynierski/ClassesImages"
image_path_A_7=os.path.join(mainPath, "A-7.png")
image_path_B_20=os.path.join(mainPath, "B-20.png")
image_path_B_34_B_42=os.path.join(mainPath, "B-34-B-42.png")
image_path_D_1=os.path.join(mainPath, "D-1.png")
image_path_D_2=os.path.join(mainPath, "D-2.png")
image_path_D_42=os.path.join(mainPath, "D-42.png")
image_path_D_43=os.path.join(mainPath, "D-43.png")
image_path_B_33_30=os.path.join(mainPath, "B-33-30.png")
image_path_B_33_40=os.path.join(mainPath, "B-33-40.png")
image_path_B_33_50=os.path.join(mainPath, "B-33-50.png")
image_path_B_33_70=os.path.join(mainPath, "B-33-70.png")
image_path_unknown=os.path.join(mainPath, "Unknown.png")
unknown_trigger_objects = ["A-7", "D-1", "D-2", "B-34-B-42", "B-20"]
ocr_classes = ['B-33-30', 'B-33-40', 'B-33-50', 'B-33-70']
# Outside the loop, initialize a dictionary to store images and timestamps for each class
roi_data = {'B-33-30': {'image': None, 'timestamp': 0}, 'B-33-40': {'image': None, 'timestamp': 0}, 'B-33-50': {'image': None, 'timestamp': 0}, 'B-33-70': {'image': None, 'timestamp': 0}}
# CONFIG END HERE

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
use_TPU = False
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate
# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If the user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if GRAPH_NAME == 'detect.tflite':
        GRAPH_NAME = 'edgetpu.tflite'
        
# Get path to the current working directory        
CWD_PATH = os.getcwd()
# Path to video file
if MODE==2:
    VIDEO_PATH = os.path.join(CWD_PATH, VIDEO_NAME)
# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)
# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
    
# Load the TensorFlow Lite model.
# If using Edge TPU, use a special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)   

if MODE==1:
    # Define VideoStream class to handle streaming of video from webcam in a separate processing thread
    class VideoStream:

        """Camera object that controls video streaming from the Picamera"""

        def __init__(self, resolution=(640, 480), framerate=30):
            index = 0
            i = 10
            while i > 0:
                cap = cv2.VideoCapture(index)
                if cap.isOpened():
                    print("Found camera at index: ", index , "please change it accordingly")
                    cap.release()
                index += 1
                i -= 1
            # Initialize the PiCamera and the camera image stream
            print("Initializing VideoCapture Device..")
            self.stream = cv2.VideoCapture(0)

            ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

            ret = self.stream.set(3, resolution[0])

            ret = self.stream.set(4, resolution[1])



            # Read the first frame from the stream

            (self.grabbed, self.frame) = self.stream.read()



            # Variable to control when the camera is stopped

            self.stopped = False



        def start(self):

            # Start the thread that reads frames from the video stream

            threading.Thread(target=self.update, args=()).start()

            return self



        def update(self):

            # Keep looping indefinitely until the thread is stopped

            while True:

                # If the camera is stopped, stop the thread

                if self.stopped:

                    # Close camera resources

                    self.stream.release()

                    return



                # Otherwise, grab the next frame from the stream

                (self.grabbed, self.frame) = self.stream.read()



        def read(self):

            # Return the most recent frame

            return self.frame



        def stop(self):

            # Indicate that the camera and thread should be stopped

            self.stopped = True
        
# Outside the loop, set the fixed resolution for the window
window_resolution = (200, 200)
# Create the window at the start of the program
cv2.namedWindow("test", cv2.WINDOW_NORMAL)
# Initialize last_ocr_timestamp outside the loop
last_ocr_timestamp = 0
# Create a new window for displaying the PNG image
cv2.namedWindow("RoadSignDetector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RoadSignDetector", 1024, 600)
cv2.setWindowProperty("RoadSignDetector", cv2.WND_PROP_FULLSCREEN, 1)

# Specify the path to your PNG image
image_path_top1 = image_path_unknown #speed limit
image_path_bottom1 = image_path_unknown #max left
image_path_bottom2 = image_path_unknown #middle
image_path_bottom3 = image_path_unknown #max right = oldest

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if 'StatefulPartitionedCall' in outname:  # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:  # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Open video file
if MODE==2:
    video = cv2.VideoCapture(VIDEO_PATH)
    imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    paused = False
    speedup_factor = 1
    slowdown_factor = 1
    default_speedup_factor = 2
    default_slowdown_factor = 2

# Function to update the image in the OpenCV window
def update_images(image_paths_top, image_paths_bottom):
    # Create a blank background with the original window dimensions
    background = np.zeros((600, 1024, 3), dtype=np.uint8)

    # Calculate the position to paste the images in the top and bottom halves
    num_images_top = len(image_paths_top)
    num_images_bottom = len(image_paths_bottom)
    space_between_images = 20
    total_width = background.shape[1]
    total_height_top = background.shape[0] // 2  # Use the top 50%
    total_height_bottom = background.shape[0] - total_height_top  # Use the bottom 50%

    # Calculate the width and height for each image based on the number of images
    image_width_top_percentage = 0.33  # Set the desired percentage of the top space's width
    image_width_top = int(total_width * image_width_top_percentage)
    image_height_top = int(total_height_top)  # Use the full height for the top image

    image_width_bottom = int((total_width - (num_images_bottom - 1) * space_between_images) / num_images_bottom)
    image_height_bottom = int(total_height_bottom)  # Use the full height for the bottom images

    # Place the top image in the center of the top space
    x_offset_top = (total_width - image_width_top) // 2
    y_offset_top = (total_height_top - image_height_top) // 2  # Center the top image vertically

    for idx, image_path in enumerate(image_paths_top):
        # Read the image using OpenCV
        image = cv2.imread(image_path)

        if image is not None:
            # Resize the top image to fit its designated space
            resized_image = cv2.resize(image, (image_width_top, image_height_top))

            # Paste the resized image onto the background
            background[y_offset_top:y_offset_top + image_height_top, x_offset_top:x_offset_top + image_width_top] = resized_image

        else:
            print(f"Error: Unable to read the image at path: {image_path}")

    for idx, image_path in enumerate(image_paths_bottom):
        # Read the image using OpenCV
        image = cv2.imread(image_path)

        if image is not None:
            # Resize each image to fit its designated space
            resized_image = cv2.resize(image, (image_width_bottom, image_height_bottom))

            # Calculate the position to paste the resized image
            x_offset = idx * (image_width_bottom + space_between_images)
            y_offset = total_height_top  # Images are aligned at the bottom of the top 50%

            # Paste the resized image onto the background
            background[y_offset:y_offset + image_height_bottom, x_offset:x_offset + image_width_bottom] = resized_image

        else:
            print(f"Error: Unable to read the image at path: {image_path}")

    # Display the composite image in the "RoadSignDetector" window
    cv2.imshow("RoadSignDetector", background)

    
def perform_ocr(roi):
    text = pytesseract.image_to_string(roi, config='--psm 8 --oem 3 -c tessedit_char_whitelist=03457')
    
    # Extract only 2-character numbers using regular expression
    numbers = re.findall(r'\b[3457]0\b', text)
    if numbers:
        print(f"OCR Result: {numbers[0]}")
        #time.sleep(1)  # Adjust the sleep duration as needed

detected_models_list = []

# Function to update the detected models list
def update_detected_models_list(new_model):
    global detected_models_list  # Add this line
    # Ensure that the model is not added more than once
    if new_model not in detected_models_list:
        detected_models_list.insert(0, new_model)

        # Keep only the last three detected models
        detected_models_list = detected_models_list[:3]

# Function to update image paths based on the detected models list
def update_image_paths(object_name):
    global image_path_bottom3, image_path_bottom2, image_path_bottom1, image_path_top1
    
    if len(detected_models_list) >= 3:
        image_path_bottom3 = os.path.join(mainPath, f"{detected_models_list[2]}.png")
    
    if len(detected_models_list) >= 2:
        image_path_bottom2 = os.path.join(mainPath, f"{detected_models_list[1]}.png")
    
    if len(detected_models_list) >= 1:
        image_path_bottom1 = os.path.join(mainPath, f"{detected_models_list[0]}.png")
        
    if object_name in ocr_classes:
        image_path_top1 = os.path.join(mainPath, f"{object_name}.png")

    if object_name in unknown_trigger_objects:
        image_path_top1 = os.path.join(mainPath, "Unknown.png")

frame_rate_calc = 1
freq = cv2.getTickFrequency() 

if MODE==1:
    videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
    time.sleep(1)
    
while True:
    t1 = cv2.getTickCount()
    if MODE==2:
        if not paused:
            # Adjust the frame rate based on speedup and slowdown factors
            for _ in range(int(speedup_factor)):
                ret, frame = video.read()
                if not ret:
                    print('Reached the end of the video!')
                    break
    if MODE==1:
        # Grab frame from video stream
        frame1 = videostream.read()
        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if the model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std


    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]  # Confidence of detected objects

    # Loop over all detections and draw detection box if confidence is above the minimum threshold
    for i in range(len(scores)):
        if min_conf_threshold < scores[i] <= 1.0:
            # Get bounding box coordinates and draw box
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            # Get class name
            object_name = labels[int(classes[i])]

            # Draw bounding box and label
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
           
            
            update_detected_models_list(object_name)
            update_image_paths(object_name)
            update_images([image_path_top1], [image_path_bottom1, image_path_bottom2, image_path_bottom3])
            
            # Perform OCR for specified classes
            if object_name in ocr_classes:
                # Extract text from the bounding box area
                ymin, ymax, xmin, xmax = map(int, [ymin, ymax, xmin, xmax])

                # Define a margin to ensure the entire bounding box is within the frame
                margin = 20

                # Check if the bounding box is fully within the frame with margin
                if margin <= ymin < ymax - margin <= imH and margin <= xmin < xmax - margin <= imW:
                    # Extract the entire width of the bounding box
                    roi = frame[ymin:ymax, xmin:xmax]

                    # Scale the image to the fixed resolution
                    roi = cv2.resize(roi, window_resolution)

                    # Update the image and timestamp for the current class in the dictionary
                    roi_data[object_name]['image'] = roi
                    roi_data[object_name]['timestamp'] = time.time()

                    # Display the latest image in the 'test ROI' window
                    #cv2.imshow("test", roi_data[object_name]['image'])

                    # Start a new thread for OCR, passing the image from 'test ROI' window
                    if roi_data[object_name]['timestamp'] > last_ocr_timestamp:
                        # Pass only the roi to the perform_ocr function
                        ocr_thread = threading.Thread(target=perform_ocr, args=(roi_data[object_name]['image'],))
                        ocr_thread.start()
                        last_ocr_timestamp = roi_data[object_name]['timestamp']

    # Draw framerate in the corner of the frame
    cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)
    
    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1
    
    # Wait for key events
    key = cv2.waitKey(1)
    if MODE==2:
        # Toggle pause state when spacebar is pressed
        if key == ord(' '):
            paused = not paused

        # Speed up video when 'm' is pressed
        elif key == ord('m'):
            speedup_factor *= 2
            slowdown_factor = default_slowdown_factor

        # Slow down video when 'n' is pressed
        elif key == ord('n'):
            slowdown_factor *= 2
            speedup_factor = default_speedup_factor

    # Press 'q' to quit
    if key == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
if MODE==2:
    video.release()
if MODE==1:
    videostream.stop()