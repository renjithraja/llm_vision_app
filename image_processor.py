from PIL import Image
import torch

def detect_objects(image_path):
    try:
        # Load YOLOv5 model
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
        model.eval()

        # Perform inference
        results = model(image_path)
        detections = results.pandas().xyxy[0]

        objects = []
        for _, row in detections.iterrows():
            objects.append({
                'name': row['name'],
                'confidence': round(row['confidence'], 2)
            })

        return objects
    except Exception as e:
        print(f"Error in object detection: {e}")
        return []
