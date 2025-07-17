import cv2
import subprocess
import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="sign_lang_1.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input shape
input_shape = input_details[0]['shape']
input_height, input_width = input_shape[1], input_shape[2]

# Start libcamera-vid with raw YUV420 stream piped to stdout
cmd = [
    "libcamera-vid",
    "--width", "640",
    "--height", "480",
    "--nopreview",
    "--codec", "yuv420",
    "-t", "0",  # Run indefinitely
    "-o", "-"
]

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)

frame_width = 640
frame_height = 480
frame_size = frame_width * frame_height * 3 // 2  # YUV420

def set_input_tensor(image):
    image = cv2.resize(image, (input_width, input_height))
    input_data = np.expand_dims(image, axis=0)
    input_data = input_data.astype(np.float32) / 255.0
     # or float32 depending on model
    interpreter.set_tensor(input_details[0]['index'], input_data)

def detect_objects(image):
    set_input_tensor(image)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding boxes
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index
    scores = np.max(classes)  # Confidence
    return boxes, classes, scores

try:
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break

        yuv = np.frombuffer(raw, dtype=np.uint8).reshape((frame_height * 3 // 2, frame_width))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

        # Run detection
        boxes, classes, scores = detect_objects(bgr)

        # Draw boxes above threshold
        for i in range(len(scores)):
            if scores[i] > 0.5:
                y_min, x_min, y_max, x_max = boxes[i]
                start_point = (int(x_min * frame_width), int(y_min * frame_height))
                end_point = (int(x_max * frame_width), int(y_max * frame_height))
                cv2.rectangle(bgr, start_point, end_point, (0, 255, 0), 2)
                label = f"ID:{int(classes[i])} {scores[i]:.2f}"
                cv2.putText(bgr, label, (start_point[0], start_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Live Object Detection", bgr)

        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    proc.terminate()
    cv2.destroyAllWindows()
