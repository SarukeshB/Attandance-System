import cv2
import numpy as np
import torch
import base64
from datetime import datetime
from django.shortcuts import render
from .models import RecognizedFace
from facenet_pytorch import InceptionResnetV1, MTCNN
import threading
import face_recognition 

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Shared frame variable
latest_frame_base64 = None

def detect_and_encode(image):
    """Detect faces in an image and return their encodings and bounding boxes."""
    with torch.no_grad():
        boxes, _ = mtcnn.detect(image)
        if boxes is not None:
            encodings = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = image[y1:y2, x1:x2]
                
                if face.size == 0:
                    continue
                
                face = cv2.resize(face, (160, 160))
                face = np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0
                face_tensor = torch.tensor(face).unsqueeze(0).to(device)
                encoding = resnet(face_tensor).detach().cpu().numpy().flatten()
                encodings.append((encoding, box))
            return encodings
    return []

def encode_known_faces(known_faces):
    """Encode faces of known individuals."""
    known_face_encodings = []
    known_face_names = []

    for name, image_paths in known_faces.items():
        for image_path in image_paths:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Unable to load image for {name} at {image_path}")
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encodings = detect_and_encode(image_rgb)
            if encodings:
                encoding, _ = encodings[0]  # Assuming one face per image
                known_face_encodings.append(encoding)
                known_face_names.append(name)

    return known_face_encodings, known_face_names

def encode_frame_to_base64(frame):
    """Encode a frame to base64 format to send to frontend."""
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return frame_base64

def process_frame(cap, known_encodings, known_names):
    """Process frames for face recognition."""
    global latest_frame_base64
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_encodings = detect_and_encode(frame_rgb)

        # Compare the frame encodings with the known encodings
        recognized_faces = compare_faces(known_encodings, known_names, frame_encodings)

        for name, box in recognized_faces:
            if name != "Unknown":
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Save recognized face to the database
                RecognizedFace.objects.create(name=name, timestamp=datetime.now())

        # Encode the frame to base64
        latest_frame_base64 = encode_frame_to_base64(frame)

def compare_faces(known_encodings, known_names, frame_encodings):
    """Compare the detected faces with known encodings."""
    recognized_faces = []
    for encoding, box in frame_encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding)  # Using the correct module
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
        recognized_faces.append((name, box))
    return recognized_faces

def capture_faces(request):
    """Captures faces using the webcam and processes them."""
    known_faces = {
        "Sarukesh": [
            "images/sarukesh.jpg", "images/sarukesh1.jpg", "images/sarukesh2.jpg", 
            "images/sarukesh3.jpg", "images/sarukesh4.jpg", "images/sarukesh5.jpg", 
            "images/sarukesh6.jpg"
        ],
        "Niaz": [
            "images/niaz.jpg", "images/niaz1.jpg", "images/niaz2.jpg", 
            "images/niaz3.jpg", "images/niaz4.jpg", "images/niaz5.jpg", 
            "images/niaz6.jpg"
        ]
    }
    
    known_encodings, known_names = encode_known_faces(known_faces)

    if len(known_encodings) == 0:
        return render(request, 'camo/index.html', {"error": "No known faces were encoded."})

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        return render(request, 'camo/index.html', {"error": "Unable to access the camera."})

    # Run the video capture in a separate thread
    capture_thread = threading.Thread(target=process_frame, args=(cap, known_encodings, known_names))
    capture_thread.start()

    return render(request, 'camo/index.html', {"frame_base64": latest_frame_base64})

def face_recognition_view(request):
    return capture_faces(request)
