import numpy as np
import tensorflow as tf
import cv2  # Using OpenCV for better performance
import pickle

# Load labels
with open("unique_labels.plk", 'rb') as t:
    unique_labels = pickle.load(t)

# Initialize TFLite interpreter
interpreter = tf.lite.Interpreter(model_path='sign_lang_1.tflite')
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Check input shape (typically [1, height, width, 3])
input_shape = input_details[0]['shape']
_, height, width, _ = input_shape

def preprocess_frame(frame):
    """Process camera frame for model input"""
    # Resize and normalize
    frame = cv2.resize(frame, (width, height))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frame = frame.astype(np.float32) / 255.0  # Normalize to [0,1]
    return np.expand_dims(frame, axis=0)  # Add batch dimension

def main():
    cap = cv2.VideoCapture(0)  # Use default camera
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess input
            input_data = preprocess_frame(frame)
            
            # Run inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            # Get results
            bbox = interpreter.get_tensor(output_details[0]['index'])[0]
            class_probs = interpreter.get_tensor(output_details[1]['index'])[0]
            
            # Process outputs
            class_id = np.argmax(class_probs)
            sign_label = unique_labels[class_id]
            confidence = class_probs[class_id]

            # Draw bounding box (if model outputs normalized coordinates)
            if len(bbox) == 4:  # [ymin, xmin, ymax, xmax]
                h, w, _ = frame.shape
                ymin, xmin, ymax, xmax = bbox
                xmin = int(xmin * w)
                xmax = int(xmax * w)
                ymin = int(ymin * h)
                ymax = int(ymax * h)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)

            # Display prediction
            cv2.putText(frame, f"{sign_label} ({confidence:.2f})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            cv2.imshow('Sign Language Detection', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()