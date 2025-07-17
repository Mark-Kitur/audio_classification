import cv2
import subprocess
import numpy as np
import tensorflow as tf
import pickle
import time
import RPi.GPIO as GPIO

led_pin = 8
GPIO.setmode(GPIO.BCM)

# Set the pin as output
GPIO.setup(led_pin, GPIO.OUT)


# Load label names
with open("unique_labels.plk", 'rb') as f:
    unique_labels = pickle.load(f)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="sign_lang_1.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_height, input_width = input_shape[1], input_shape[2]

# libcamera-vid command
frame_width = 640
frame_height = 480
cmd = [
    "libcamera-vid",
    "--width", str(frame_width),
    "--height", str(frame_height),
    "--nopreview",
    "--codec", "yuv420",
    "-t", "0",
    "-o", "-"
]

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)

frame_size = frame_width * frame_height * 3 // 2  # YUV420 format

# Helper functions
def set_input_tensor(image):
    image_resized = cv2.resize(image, (input_width, input_height))
    input_data = image_resized.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)

def detect_objects(image):
    set_input_tensor(image)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    pred_class= unique_labels[np.argmax(classes)]
    scores= np.max(classes)
    return boxes, pred_class, scores

# Main loop
try:
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            print("Incomplete frame captured")
            break

        yuv = np.frombuffer(raw, dtype=np.uint8).reshape((frame_height * 3 // 2, frame_width))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

        box, clas_name,confidence= detect_objects(bgr)

        if confidence>0.65:
            if clas_name =="C":
                GPIO.output(led_pin,GPIO.HIGH)
                time.sleep(5)
                
            elif clas_name =="O":
                GPIO.output(led_pin,GPIO.LOW)
                time.sleep(1)
            xmin, ymin,xmax,ymax= box
            x1= int(xmin*frame_width)
            y1= int(ymin*frame_height)
            x2= int(xmax*frame_width)
            y2= int(ymax*frame_height)

            # Draw bounding box and label on the frame
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(bgr, clas_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        else:
            cv2.putText(bgr, "poor image", (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


        cv2.imshow("Object Detection", bgr)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break
finally:
    proc.terminate()
    GPIO.cleanup()
    cv2.destroyAllWindows()
