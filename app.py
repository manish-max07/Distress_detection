from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import time
import logging
import pygame  # For audio alerts

app = Flask(__name__)

# Initialize pygame for audio
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.mp3")  # Add a alarm.wav file to your static folder

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Hand detector setup
detector = HandDetector(detectionCon=0.8, maxHands=1)
alert_active = False
last_alert_time = 0

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def is_distress_signal(hand):
    # Check for "Help" signal: open palm with thumb folded across
    fingers = detector.fingersUp(hand)
    
    # Alternative 1: All fingers extended except thumb (might need adjustment)
    # return fingers == [0, 1, 1, 1, 1]
    
    # Alternative 2: Check thumb position using landmarks
    lmList = hand["lmList"]
    thumb_tip = lmList[4]
    thumb_mcp = lmList[2]
    
    # Check if thumb is folded across palm (X coordinate comparison)
    if hand["type"] == "Right":
        return thumb_tip[0] < thumb_mcp[0]
    else:
        return thumb_tip[0] > thumb_mcp[0]

def generate_frames():
    global alert_active, last_alert_time
    while True:
        try:
            success, img = cap.read()
            if not success:
                logging.error("Failed to read from video capture")
                continue

            # Mirror the image
            img = cv2.flip(img, 1)
            
            # Detect hands
            hands, img = detector.findHands(img, draw=True)

            # Check for distress signal
            if hands:
                hand = hands[0]
                if is_distress_signal(hand):
                    if time.time() - last_alert_time > 5:  # 5 second cooldown
                        alert_active = True
                        last_alert_time = time.time()
                        alarm_sound.play()
                else:
                    if time.time() - last_alert_time > 5:
                        alert_active = False

            # Add alert overlay
            if alert_active:
                cvzone.putTextRect(img, "DISTRESS SIGNAL DETECTED!", 
                                [200, 100], 3, 4, (0,0,255), (255,255,255))
                cv2.rectangle(img, (50, 50), (1230, 670), (0, 0, 255), 10)

            ret, buffer = cv2.imencode('.jpg', img)
            if not ret:
                logging.error("Failed to encode image")
                continue

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            logging.error(f"Error in generate_frames: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_alert')
def check_alert():
    global alert_active
    return jsonify({'alertActive': alert_active})

@app.route('/acknowledge_alert', methods=['POST'])
def acknowledge_alert():
    global alert_active
    alert_active = False
    alarm_sound.stop()
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Remove ssl_context if not needed