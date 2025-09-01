import os
import sys
import time
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import RPi.GPIO as GPIO

# ------------------------ Argument Parsing ------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model (.pt)')
parser.add_argument('--source', default='0', help='Camera source: 0, 1, or "picamera2"')
parser.add_argument('--resolution', default='640x480', help='Resolution: WIDTHxHEIGHT')
args = parser.parse_args()

model_path = args.model
source = args.source
resolution = args.resolution

# Parse resolution
try:
    res_w, res_h = map(int, resolution.lower().split('x'))
except ValueError:
    print("Invalid resolution format. Use WIDTHxHEIGHT")
    sys.exit(1)

# ------------------------ Servo Setup ------------------------
servo_pin_sorter = 19
servo_pin_feeder = 26

GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin_sorter, GPIO.OUT)
GPIO.setup(servo_pin_feeder, GPIO.OUT)

pwm_sorter = GPIO.PWM(servo_pin_sorter, 50)
pwm_feeder = GPIO.PWM(servo_pin_feeder, 50)

pwm_sorter.start(0)
pwm_feeder.start(0)

def set_servo_angle(pwm, angle):
    duty = 2.5 + (angle / 180.0) * 10
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)
    pwm.ChangeDutyCycle(0)

def set_feeder_servo_angle(angle):
    duty = 2.5 + (angle / 180.0) * 10
    pwm_feeder.ChangeDutyCycle(duty)
    time.sleep(0.5)
    pwm_feeder.ChangeDutyCycle(0)

# Feeder logic with sorter trigger at step 2
def activate_feeder(last_class=None):
    print("Feeder: Moving object into frame in 3 steps.")
    
    # Step 1
    set_feeder_servo_angle(90)
    time.sleep(5)
    
    # Step 2
    set_feeder_servo_angle(50)
    
    # --- Move sorter while feeder is in step 2 ---
    if last_class == "intact":
        sorter_left()
    elif last_class == "damaged":
        sorter_right()
    
    time.sleep(5)  # wait for sorter + feeder to move together
    
    # Step 3 - reset feeder
    set_feeder_servo_angle(0)
    time.sleep(5)

# Sorter positions
def sorter_neutral():
    set_servo_angle(pwm_sorter, 90)

def sorter_left():
    set_servo_angle(pwm_sorter, 45)

def sorter_right():
    set_servo_angle(pwm_sorter, 135)

# ------------------------ Load YOLO Model ------------------------
if not os.path.exists(model_path):
    print(f"Model path '{model_path}' is invalid.")
    sys.exit(1)

model = YOLO(model_path)
CONF_THRESHOLD = 0.50
labels = model.names

# ------------------------ Camera Setup ------------------------
is_picamera = False
if source.lower() == 'picamera2':
    try:
        from picamera2 import Picamera2
        picam = Picamera2()
        picam.configure(picam.create_video_configuration(main={"format": 'XRGB8888', "size": (res_w, res_h)}))
        picam.start()
        is_picamera = True
    except Exception as e:
        print(f"Failed to initialize PiCamera2: {e}")
        GPIO.cleanup()
        sys.exit(1)
else:
    cap = cv2.VideoCapture(int(source))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_h)

# ------------------------ Main Loop ------------------------
prev_time = 0
sorting_state = "idle"
last_class = None
fps_buffer = []
fps_avg_len = 20

try:
    sorter_neutral()
    set_feeder_servo_angle(0)  # Reset feeder at start

    while True:
        current_time = time.time()

        # FPS limiter ~5 FPS
        if (current_time - prev_time) < 0.20:
            continue
        prev_time = current_time

        # Get frame
        if is_picamera:
            frame_bgra = picam.capture_array()
            frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
        else:
            ret, frame = cap.read()
            if not ret:
                print("Camera not detected.")
                break

        # ---------------- Step 1: Bring bean ----------------
        if sorting_state == "idle":
            activate_feeder(last_class)
            sorting_state = "classifying"
            last_class = None
            continue

        # ---------------- Step 2: Classification ----------------
        if sorting_state == "classifying":
            results = model(frame, verbose=False)[0]
            detected_class = None
            detected_conf = 0.0

            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > detected_conf and conf >= CONF_THRESHOLD:
                    detected_class = cls
                    detected_conf = conf

                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{labels[cls]}: {int(conf * 100)}%"
                color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Set last_class for feeder + sorter
            if detected_class is not None:
                last_class = "intact" if detected_class == 0 else "damaged"
                # Feed + move sorter together
                activate_feeder(last_class)
                sorting_state = "reset"
            else:
                print("No confident detection.")

        # ---------------- Step 3: Reset sorter ----------------
        elif sorting_state == "reset":
            sorter_neutral()
            set_feeder_servo_angle(0)  # Reset feeder to initial position
            sorting_state = "idle"

        # FPS Calculation
        end_time = time.time()
        fps = 1 / (end_time - current_time)
        fps_buffer.append(fps)
        if len(fps_buffer) > fps_avg_len:
            fps_buffer.pop(0)
        avg_fps = sum(fps_buffer) / len(fps_buffer)

        # Show FPS
        cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Show window
        cv2.imshow("Sorting System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pwm_sorter.stop()
    pwm_feeder.stop()
    GPIO.cleanup()
    if not is_picamera:
        cap.release()
    else:
        picam.stop()
    cv2.destroyAllWindows()
