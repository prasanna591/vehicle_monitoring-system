import cv2
import mediapipe as mp
import numpy as np
import threading
import torch
import os
import time
from playsound import playsound
from ultralytics import YOLO
import simpleaudio as sa 


# OS-based sound module
import platform
if platform.system() == "Windows":
    import winsound  # Windows beep sound
    sa = None   #prevents undefined variable issues

else:
    import simpleaudio as sa  # Linux/Mac alternative

# ------------------------------- DRIVER MONITORING SETUP -------------------------------
print("Initializing Driver Monitoring...")

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Alert thresholds
EYE_CLOSED_FRAMES = 0
EYE_CLOSED_THRESHOLD = 30  # Adjust based on FPS (~1 sec if FPS=30)
HEAD_TILT_THRESHOLD = 25  # Maximum angle for head tilt alert
ALARM_ON = False  # Prevent multiple alarms

# Open driver camera (First Camera - Index 0)
driver_cam = cv2.VideoCapture(0)
if not driver_cam.isOpened():
    print("Error: Could not open driver camera!")
    exit()
print("Driver camera opened successfully!")

def play_alarm():
    """Plays alarm sound in a separate thread"""
    global ALARM_ON
    try:
        if not ALARM_ON:
            ALARM_ON = True
            threading.Thread(target=lambda: playsound("beep.mp3"), daemon=True).start()
    except Exception as e:
        print(f"Error playing alarm: {e}")
        # Fallback to system beep if sound file fails
        print('\a')    

# Eye Aspect Ratio Calculation
def eye_aspect_ratio(eye_landmarks, frame_shape):
    """Calculate Eye Aspect Ratio (EAR) to detect blinking"""
    eye = np.array([(int(landmark.x * frame_shape[1]), int(landmark.y * frame_shape[0])) for landmark in eye_landmarks])
    
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])

    return (A + B) / (2.0 * C)

# Head Tilt Calculation
def head_tilt_angle(face_landmarks, frame_shape):
    """Calculate head tilt based on left and right eye midpoint"""
    left_eye = np.array([int(face_landmarks[LEFT_EYE[0]].x * frame_shape[1]), int(face_landmarks[LEFT_EYE[0]].y * frame_shape[0])])
    right_eye = np.array([int(face_landmarks[RIGHT_EYE[0]].x * frame_shape[1]), int(face_landmarks[RIGHT_EYE[0]].y * frame_shape[0])])

    return np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

# ------------------------------- TRAFFIC MONITORING SETUP -------------------------------
print("Initializing Traffic Monitoring...")

# Ensure YOLO model exists
model_path = "yolov8n.pt"
if not os.path.exists(model_path):
    print(f"Error: YOLO model '{model_path}' not found. Download it first!")
    exit()

# Load YOLOv8 model
traffic_model = YOLO(model_path)

# Open traffic camera (Second Camera - Index 1)
traffic_cam = cv2.VideoCapture(1)
if not traffic_cam.isOpened():
    print("Error: Could not open traffic camera!")
    exit()
print("Traffic camera opened successfully!")

# Objects to monitor
ALERT_CLASSES = {0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Distance estimation
def estimate_distance(bbox_width, known_width=1.8, focal_length=600):
    return float('inf') if bbox_width == 0 else (known_width * focal_length) / bbox_width

# ------------------------------- MULTITHREADING FUNCTIONS -------------------------------

def driver_monitoring():
    global EYE_CLOSED_FRAMES, ALARM_ON, driver_cam

    while True:
        ret, frame = driver_cam.read()

        # If camera fails, try to reinitialize
        if not ret:
            print("Warning: Driver camera disconnected! Reinitializing...")
            driver_cam.release()
            time.sleep(2)  # Wait before reinitializing
            driver_cam = cv2.VideoCapture(0)
            continue  # Skip this loop iteration
        
        # Rest of the code remains unchanged...


        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye_ear = eye_aspect_ratio([face_landmarks.landmark[i] for i in LEFT_EYE], frame.shape)
                right_eye_ear = eye_aspect_ratio([face_landmarks.landmark[i] for i in RIGHT_EYE], frame.shape)
                avg_ear = (left_eye_ear + right_eye_ear) / 2

                if avg_ear < 0.2:
                    EYE_CLOSED_FRAMES += 1
                    if EYE_CLOSED_FRAMES > EYE_CLOSED_THRESHOLD:
                        cv2.putText(frame, "ALERT! WAKE UP!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        play_alarm()
                else:
                    EYE_CLOSED_FRAMES = 0
                    ALARM_ON = False

                tilt_angle = head_tilt_angle(face_landmarks.landmark, frame.shape)
                if abs(tilt_angle) > HEAD_TILT_THRESHOLD:
                    cv2.putText(frame, "WARNING: HEAD TILT!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    play_alarm()

        cv2.imshow("Driver Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def traffic_monitoring():
    while traffic_cam.isOpened():
        ret, frame = traffic_cam.read()
        if not ret:
            break

        results = traffic_model(frame)
        annotated_frame = results[0].plot()

        for result in results:
            for obj in result.boxes.data:
                x1, y1, x2, y2, conf, cls = obj.tolist()
                cls = int(cls)

                if cls in ALERT_CLASSES:
                    obj_name = ALERT_CLASSES[cls]
                    bbox_width = x2 - x1
                    distance = estimate_distance(bbox_width)

                    if distance < 2.0:
                        cv2.putText(annotated_frame, f"⚠ {obj_name} Too Close!", 
                                    (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        play_alarm()

        cv2.imshow("Traffic Monitoring", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ------------------------------- RUNNING BOTH MONITORS SIMULTANEOUSLY -------------------------------
t1 = threading.Thread(target=driver_monitoring)
t2 = threading.Thread(target=traffic_monitoring)

t1.start()
t2.start()

t1.join()
t2.join()

# Release resources
driver_cam.release()
traffic_cam.release()
cv2.destroyAllWindows()
