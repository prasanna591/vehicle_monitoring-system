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
    sa = None   # prevents undefined variable issues
else:
    import simpleaudio as sa  # Linux/Mac alternative

# ------------------------------- MODE SWITCH MENU -------------------------------
print("\n" + "="*40)
print("     VEHICLE MONITORING SYSTEM CONFIG     ")
print("="*40)
print("1. Driver Monitoring Only (Webcam)")
print("2. Traffic Monitoring Only (Webcam)")
print("3. Both (Requires 2 Cameras or Configured Streams)")
print("="*40)

choice = input("Select operation mode (1, 2, or 3): ").strip()

# Configure active components based on choice
DRIVE_MONITOR_ACTIVE = choice in ["1", "3"]
TRAFFIC_MONITOR_ACTIVE = choice in ["2", "3"]

# Dynamically assign the single available camera index (0) based on choice
DRIVER_CAM_INDEX = 0 if choice == "1" else (0 if choice == "3" else -1)
TRAFFIC_CAM_INDEX = 0 if choice == "2" else (1 if choice == "3" else -1)
print("="*40 + "\n")

# Global Alarm Trackers (Separated to fix muting issues)
DRIVER_ALARM_ON = False
LAST_DRIVER_ALARM_TIME = 0

TRAFFIC_ALARM_ON = False
LAST_TRAFFIC_ALARM_TIME = 0

# Track positions of objects across frames to calculate speed
tracked_objects = {} 

# Real-world ADAS Configurations
TTC_THRESHOLD = 2.5  # Alert if collision path is less than 2.5 seconds away
CRITICAL_DISTANCE = 3.0  # Absolute fallback alert if something is under 3 meters away

# Objects to monitor
ALERT_CLASSES = {0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# ------------------------------- AUDIO SEPARATE CONTROLLERS -------------------------------

def play_driver_alarm():
    """Independent audio controller for driver face mesh alerts"""
    global DRIVER_ALARM_ON, LAST_DRIVER_ALARM_TIME
    current_time = time.time()
    
    if current_time - LAST_DRIVER_ALARM_TIME < 2.0:
        return
        
    try:
        if not DRIVER_ALARM_ON:
            DRIVER_ALARM_ON = True
            LAST_DRIVER_ALARM_TIME = current_time
            
            def play():
                global DRIVER_ALARM_ON
                os.system("paplay beep.mp3 > /dev/null 2>&1 || mpg123 beep.mp3 > /dev/null 2>&1 || aplay beep.wav > /dev/null 2>&1")
                time.sleep(1.5)
                DRIVER_ALARM_ON = False
                
            threading.Thread(target=play, daemon=True).start()
    except Exception as e:
        print(f"Driver audio error: {e}")

def play_traffic_alarm():
    """Independent audio controller for traffic YOLO tracking alerts"""
    global TRAFFIC_ALARM_ON, LAST_TRAFFIC_ALARM_TIME
    current_time = time.time()
    
    if current_time - LAST_TRAFFIC_ALARM_TIME < 2.0:
        return
        
    try:
        if not TRAFFIC_ALARM_ON:
            TRAFFIC_ALARM_ON = True
            LAST_TRAFFIC_ALARM_TIME = current_time
            
            def play():
                global TRAFFIC_ALARM_ON
                os.system("paplay beep.mp3 > /dev/null 2>&1 || mpg123 beep.mp3 > /dev/null 2>&1 || aplay beep.wav > /dev/null 2>&1")
                time.sleep(1.0)
                TRAFFIC_ALARM_ON = False
                
            threading.Thread(target=play, daemon=True).start()
    except Exception as e:
        print(f"Traffic audio error: {e}")


# ------------------------------- DRIVER MONITORING SETUP -------------------------------
if DRIVE_MONITOR_ACTIVE:
    print("Initializing Driver Monitoring...")
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        EYE_CLOSED_FRAMES = 0
        EYE_CLOSED_THRESHOLD = 30  
        HEAD_TILT_THRESHOLD = 25  

        driver_cam = cv2.VideoCapture(DRIVER_CAM_INDEX)
        if not driver_cam.isOpened():
            print("Warning: Could not open driver camera! Driver monitoring disabled.")
            DRIVE_MONITOR_ACTIVE = False
        else:
            print("Driver camera opened successfully!")
    except Exception as e:
        print(f"Driver monitoring initialization failed: {e}")
        DRIVE_MONITOR_ACTIVE = False

def eye_aspect_ratio(eye_landmarks, frame_shape):
    eye = np.array([(int(landmark.x * frame_shape[1]), int(landmark.y * frame_shape[0])) for landmark in eye_landmarks])
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def head_tilt_angle(face_landmarks, frame_shape):
    left_eye = np.array([int(face_landmarks[LEFT_EYE[0]].x * frame_shape[1]), int(face_landmarks[LEFT_EYE[0]].y * frame_shape[0])])
    right_eye = np.array([int(face_landmarks[RIGHT_EYE[0]].x * frame_shape[1]), int(face_landmarks[RIGHT_EYE[0]].y * frame_shape[0])])
    return np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))


# ------------------------------- TRAFFIC MONITORING SETUP -------------------------------
if TRAFFIC_MONITOR_ACTIVE:
    print("Initializing Traffic Monitoring...")
    try:
        model_path = "yolov8n.pt"
        if not os.path.exists(model_path):
            print(f"Warning: YOLO model '{model_path}' not found. Traffic monitoring disabled.")
            TRAFFIC_MONITOR_ACTIVE = False
        else:
            traffic_model = YOLO(model_path)
            traffic_cam = cv2.VideoCapture(TRAFFIC_CAM_INDEX)
            if not traffic_cam.isOpened():
                print(f"Warning: Could not open traffic camera (Index {TRAFFIC_CAM_INDEX})! Traffic monitoring disabled.")
                TRAFFIC_MONITOR_ACTIVE = False
            else:
                print("Traffic camera opened successfully!")
    except Exception as e:
        print(f"Traffic monitoring initialization failed: {e}")
        TRAFFIC_MONITOR_ACTIVE = False

def estimate_distance(bbox_width, known_width=1.8, focal_length=600):
    return float('inf') if bbox_width == 0 else (known_width * focal_length) / bbox_width


# ------------------------------- MULTITHREADING RUN TIME -------------------------------

def driver_monitoring():
    global EYE_CLOSED_FRAMES, DRIVER_ALARM_ON, driver_cam
    if not DRIVE_MONITOR_ACTIVE:
        return

    while True:
        ret, frame = driver_cam.read()
        if not ret:
            print("Warning: Driver camera disconnected! Reinitializing...")
            driver_cam.release()
            time.sleep(2)  
            driver_cam = cv2.VideoCapture(DRIVER_CAM_INDEX) 
            continue  
        
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
                        play_driver_alarm()
                else:
                    EYE_CLOSED_FRAMES = 0
                    DRIVER_ALARM_ON = False

                tilt_angle = head_tilt_angle(face_landmarks.landmark, frame.shape)
                if abs(tilt_angle) > HEAD_TILT_THRESHOLD:
                    cv2.putText(frame, "WARNING: HEAD TILT!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    play_driver_alarm()

        cv2.imshow("Driver Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def traffic_monitoring():
    global tracked_objects
    if not TRAFFIC_MONITOR_ACTIVE:
        return

    while traffic_cam.isOpened():
        ret, frame = traffic_cam.read()
        if not ret:
            print("Warning: Traffic camera stream stopped.")
            break

        current_time = time.time()
        results = traffic_model(frame)
        annotated_frame = results[0].plot()

        for result in results:
            for idx, obj in enumerate(result.boxes.data):
                x1, y1, x2, y2, conf, cls = obj.tolist()
                cls = int(cls)

                if cls in ALERT_CLASSES:
                    obj_name = ALERT_CLASSES[cls]
                    bbox_width = x2 - x1
                    
                    # 1. Distance Calculation
                    current_distance = estimate_distance(bbox_width)
                    
                    # 2. Velocity Tracking / Approach Calculations
                    obj_key = f"{cls}_{idx}"
                    ttc = float('inf')
                    
                    if obj_key in tracked_objects:
                        last_distance, last_time = tracked_objects[obj_key]
                        distance_delta = last_distance - current_distance
                        time_delta = current_time - last_time
                        
                        if time_delta > 0:
                            approach_velocity = distance_delta / time_delta
                            if approach_velocity > 0.5:  # Target moving closer
                                ttc = current_distance / approach_velocity

                    tracked_objects[obj_key] = [current_distance, current_time]

                    # 3. Decision Matrix
                    is_imminent_threat = (0 < ttc <= TTC_THRESHOLD)
                    is_too_close = (current_distance <= CRITICAL_DISTANCE)

                    if is_imminent_threat or is_too_close:
                        label_text = f"CRASH RISK: {ttc:.1f}s!" if is_imminent_threat else "TOO CLOSE!"
                        
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                        cv2.putText(annotated_frame, f"⚠ {obj_name} {label_text}", 
                                    (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                        play_traffic_alarm()

        if len(tracked_objects) > 25:
            tracked_objects.clear()

        cv2.imshow("Traffic Monitoring", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ------------------------------- MULTITHREADING THREAD EXECUTOR -------------------------------
threads = []

if DRIVE_MONITOR_ACTIVE:
    t1 = threading.Thread(target=driver_monitoring)
    threads.append(t1)
    t1.start()

if TRAFFIC_MONITOR_ACTIVE:
    t2 = threading.Thread(target=traffic_monitoring)
    threads.append(t2)
    t2.start()

for t in threads:
    t.join()

if DRIVE_MONITOR_ACTIVE:
    try: driver_cam.release()
    except: pass
if TRAFFIC_MONITOR_ACTIVE:
    try: traffic_cam.release()
    except: pass

cv2.destroyAllWindows()
print("System shutdown cleanly.")