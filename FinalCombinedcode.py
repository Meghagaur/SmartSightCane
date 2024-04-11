import argparse
import sys
import time
import os

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from picamera2 import Picamera2
from utils import visualize
import serial

ser = serial.Serial("/dev/serial0", 115200, timeout=1)

COUNTER, FPS = 0, 30
START_TIME = time.time()
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640,480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

def speak (sentence):

    sentence = sentence.replace(" " "_")

    os.system(f"espeak '{sentence}'")

def read_distance():
    time.sleep(1)
    counter = ser.in_waiting
    if counter > 8:
        bytes_serial = ser.read(9)
        ser.reset_input_buffer()
        if bytes_serial [0] == 0x59 and bytes_serial [1] == 0x59:
            distance = bytes_serial [2] + bytes_serial [3]*256
            ser.reset_input_buffer()
            return distance
    return None

def run (model: str, max_results: int, score_threshold: float,
        camera_id: int, width: int, height: int) -> None:

        """Continuously run inference on images acquired from the camera.
        Args:
            model: Name of the TFLite object detection model.
            max_results: Max number of detection results.
            score_threshold: The score threshold of detection results.
            camera_id: The camera id to be passed to OpenCV.
            width: The width of the frame captured from the camera.
            height: The height of the frame captured from the camera.
        """

        row_size = 30 # pixels
        left_margin = 60 # pixels
        text_color = (0, 0, 255) # black
        font_size = 1
        font_thickness = 3
        fps_avg_frame_count = 10

        boxColor=(255,0,0)
        boxWeight=2

        labelHeight=1.5
        labelColor=(0,255,0)
        labelWeight=2


    font = cv2.FONT_HERSHEY_SIMPLEX

    detection frame = None
    detection_result_list = []

    def save_result(result: vision. ObjectDetectorResult, unused_output_image: mp.Image,timestamp_ms: int):
        global FPS, COUNTER, START_TIME

            # Calculate the FPS
            if COUNTER % fps_avg_frame_count == 0:
                FPS = fps_avg_frame_count / (time.time() START TIME)
                START TIME = time.time()

            detection_result_list.append(result)
            COUNTER += 1

    # Initialize the object detection model
    base_options = python. BaseOptions (model_asset_path=model)|
    options = vision. ObjectDetectorOptions (base_options=base_options,
                                            running_mode=vision.RunningMode.LIVE_STREAM, 
                                            score_threshold=score_threshold,
                                            max_results=max_results,
                                            result_callback=save result) 
    Detector = vision.ObjectDetector.create_from_options (options)

    while True:
        image= picam2.capture_array() 
        image=cv2.resize(image, (640,480))
        image = cv2.flip(image, -1)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp. Image (image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run object detection using the model. 
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Read distance and speak
        distance = read_distance()
        if distance is not None:
            #if detector.has_result():
              #result = detector.get_result()
              #for detection in result.detections:
                 #obj_name = detection.categories [0].category_name
            if detection_result_list:
        # print(detection_result_list[0].detections)
                if distance > 100 and distance <= 1000:
                    for myDetects in detection_result_list[0].detections: 
                        objName = myDetects.categories[0].category_name
                        speak(f"{objName} detected at a distance of {distance} centimeteter")
                        #print(f"{objName} detected at a distance of {distance} centimeteter")
                else: speak("stop stop stop")
        # Show the FPS
        #fps_text = 'FPS = {:.1f}'.format(FPS)
        #text_location = (left_margin, row_size)
        #current_frame = image
        #cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_SIMPLEX
        #font_size, text_color, font_thickness, cv2.LINE_AA)

        #if detector.has_result():
            #for myDetects in detector.get_result().detections:
                #UL = (myDetects.bounding_box.origin_x, myDetects.bounding_box.origin_y
                #LR = (myDetects.bounding_box.origin_x + myDetects.bounding_box.width,
                #myDetects.bounding_box.origin_y + myDetects.bounding_box.height)
            #objName = myDetects.categories [0].category_name
            #current_frame = cv2.rectangle(image, UL, LR, boxColor, boxWeight)
            #cv2.putText(current_frame, objName, UL, font, labelHeight, labelCol

        #cv2.imshow('object_detection', current_frame)

        """if detection result list:
         # print(detection_result_list[0].detections)
          for myDetects in detection_result_list[0].detections:
            #print(myDetects)
            UL=(myDetects.bounding_box.origin_x, myDetects.bounding_box.origin_y)
            LR=(myDetects.bounding_box.origin_x+myDetects.bounding_box.width,myDetec 
             objName = myDetects.categories[0].category_name
             current_frame=cv2.rectangle (image, UL, LR, boxColor, boxWeight)
             cv2.putText(current_frame, objName, UL, font, labelHeight, labelColor, labelWe

            detection frame = current frame
            

            if detection frame is not None:
                cv2.imshow('object_detection', detection_frame)"""

            detection result list.clear()
            # Stop the program if the q key is pressed.
            if cv2.waitKey(1) == ord('q'):
                break

    detector.close()
    picam2.stop()
    cv2.destroyAllWindows()

    def main():
        parser = argparse.ArgumentParser (
           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument(
           '--model',
            help='Path of the object detection model.',
            required=False,
            default='efficientdet liteo.tflite')
            #default='best.tflite')
        parser.add_argument(
           '-- maxResults',
            help='Max number of detection results.',
            required=False,
            default=5)
        parser.add_argument(
            '--scoreThreshold',
            help='The score threshold of detection results.',
            required=False,
            type=float,
            default=0.3)
        # Finding the camera ID can be very reliant on platform-dependent methods. 
        # One common approach is to use the fact that camera IDs are usually indexed sequentially by the OS, starting from 0. 
        # Here, we use OpenCV and create a VideoCapture object for each potential ID with 'cap = cv2.VideoCapture(i)'.
        # If 'cap' is None or not 'cap.isOpened()', it indicates the camera ID is not available.
        parser.add_argument(
            '--cameraId', help='Id of camera.', required=False, type=int,default=0)
        parser.add_argument(
            '--frameWidth',
            help='Width of frame to capture from camera.',
            required=False,
            type=int,
            default=640)
        parser.add_argument(
            '--frameHeight',
            help='Height of frame to capture from camera.',
            required=False,
            type=int,
            default=480)
        args = parser.parse_args()

        run(args.model, int(args.maxResults),
            args.scoreThreshold, int(args.cameraId), args.frameWidth, args.frameHeight)

if _name_ == '_main_':
   main()
