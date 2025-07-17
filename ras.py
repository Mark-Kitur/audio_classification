import numpy as np
import tensorflow as tf
import cv2
import pickle
import time

# === Load Labels ===
with open("unique_labels.plk", 'rb') as t:
    unique_labels = pickle.load(t)

# === Load TFLite Model ===
interpreter = tf.lite.Interpreter(model_path='sign_lang_1.tflite')
interpreter.allocate_tensors()

# Get model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Camera Setup ===
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("❌ Error: Could not open camera.")
    exit()

# Get camera resolution
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer to save output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (frame_width, frame_height))

# === Helper Function to Map Prediction ===
def get_label(prob):
    return unique_labels[np.argmax(prob)]

# === Main Loop ===
print("✅ Starting video stream. Press 'q' to quit.")
while True:
    ret, frame = cam.read()
    if not ret:
        print('❌ Error: Failed to capture frame.')
        break

    # Resize to model input size (assumed 480x480 here)
    resized = cv2.resize(frame, (480, 480))
    input_tensor = resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    pred_box = interpreter.get_tensor(output_details[0]['index'])
    pred_prob = interpreter.get_tensor(output_details[1]['index'])

    # Postprocessing
    pred_class = get_label(pred_prob)
    confidence = np.max(pred_prob)
    original_height, original_width, _ = frame.shape

    if confidence > 0.5:
        xmin, ymin, xmax, ymax = pred_box[0]
        x1 = int(xmin * original_width)
        y1 = int(ymin * original_height)
        x2 = int(xmax * original_width)
        y2 = int(ymax * original_height)

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{pred_class} - {confidence:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No Object Detected", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Save and show frame
    out.write(frame)
    cv2.imshow('Real-Time Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cam.release()
out.release()
cv2.destroyAllWindows()
print("🛑 Video stream ended.")
