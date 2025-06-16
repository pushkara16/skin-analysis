from ultralytics import YOLO
import cv2

# Load the model
model_path = r"C:\Users\Lekhana\Downloads\tionbest (1).pt"
model = YOLO(model_path)

# Load the test image (use the same image you used in Colab)
image_path = r"C:\Users\Lekhana\Downloads\kevu.jpg"
img = cv2.imread(image_path)

# Run inference
results = model( r"C:\Users\Lekhana\Downloads\kevu.jpg", device=0, conf=0.25)
results[0].show()
  # set a low confidence threshold

# Print detections
