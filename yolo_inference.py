# This file is going to be used to test ultralytics ( library that contains YOLO)

from ultralytics import YOLO


model = YOLO('models/best.pt') # Importing the model // yolov8x is the largest one with the best accuracy and uses the most PC ressources

results = model.predict("input_vids/08fd33_4.mp4", save=True)
print(results[0])
print("---------------------------------------------------------------------------------------------------------------")
for box in results[0].boxes:
    print(box)

print("---------------------------------------------------------------------------------------------------------------")
