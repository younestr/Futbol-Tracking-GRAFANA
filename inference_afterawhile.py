from ultralytics import YOLO
import os

def main():
    model_path = 'models/best.pt'
    input_video = 'input_vids/08fd33_4.mp4'
    
    # Check if the model and input video exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    if not os.path.exists(input_video):
        print(f"Error: Input video file not found at {input_video}")
        return
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)  # YOLOv8x is the largest model
    
    # Run predictions
    print(f"Processing video {input_video}...")
    results = model.predict(input_video, save=True)
    print("Predictions complete.")
    
    # Output directory
    output_dir = "runs/detect/predict"
    print(f"Results saved in {output_dir}")
    
    # Print summary for the first frame
    print("\nFirst frame summary:")
    print(results[0])
    print("-" * 120)
    
    # Print detailed bounding box information
    print("Bounding boxes:")
    for box in results[0].boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf:.2f}, Coordinates: {box.xyxy}")
    print("-" * 120)

if __name__ == "__main__":
    main()
