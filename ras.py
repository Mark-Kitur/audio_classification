import cv2
import subprocess
import numpy as np
import tensorflow as tf
import pickle

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
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    return boxes, classes, scores

# Main loop
try:
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            print("⚠️ Incomplete frame captured")
            break

        yuv = np.frombuffer(raw, dtype=np.uint8).reshape((frame_height * 3 // 2, frame_width))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

        boxes, classes, scores = detect_objects(bgr)

        for i in range(len(scores)):
            if scores[i] > 0.5:
                y_min, x_min, y_max, x_max = boxes[i]
                start_point = (int(x_min * frame_width), int(y_min * frame_height))
                end_point = (int(x_max * frame_width), int(y_max * frame_height))
                cv2.rectangle(bgr, start_point, end_point, (0, 255, 0), 2)
                
                class_id = int(classes[i])
                label = f"{unique_labels[class_id]}: {scores[i]:.2f}"
                cv2.putText(bgr, label, (start_point[0], start_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Object Detection", bgr)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break
finally:
    proc.terminate()
    cv2.destroyAllWindows()
