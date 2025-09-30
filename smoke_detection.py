import cv2
import torch
import time
import smtplib
import ssl
import os

# --- 1. USER CONFIGURATION ---
# --- Email Settings ---q
SENDER_EMAIL = "lab.alert@gmail.com"
# IMPORTANT: Replace this with your 16-character Gmail App Password
SENDER_PASSWORD = "snfkhbnsavhqcvzd" 
RECIPIENT_EMAIL = "jyotishankarsahoo20@gmail.com"

# --- Model Settings ---
MODEL_PATH = "firesmoke.pt"

# --- Detection & Alert Settings ---
CONFIDENCE_THRESHOLD = 0.60
ALERT_COOLDOWN_SECONDS = 60

# --- 2. ALERT FUNCTION (Cleaned Up) ---
def send_alert_email():
    """Sends a fire/smoke alert email using the configuration variables."""
    subject = "AAAAGG.....!" # Your custom subject line
    body = "A potential fire or smoke hazard has been detected by the monitoring system. Please check immediately."
    message = f"Subject: {subject}\n\n{body}"
    
    context = ssl.create_default_context()
    try:
        print("Sending alert email...")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            # Use the variables from the top for login and sending
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, message)
        print("Email sent successfully!")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

# --- 3. MODEL LOADING ---
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Ensure your custom model '{MODEL_PATH}' is in the same folder as the script.")
    exit()

# --- 4. REAL-TIME MONITORING ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

last_alert_time = 0
print("Starting fire/smoke monitoring... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.pandas().xyxy[0]
    hazard_detected = False
    
    for _, det in detections.iterrows():
        if det['name'] in ['fire', 'smoke'] and det['confidence'] > CONFIDENCE_THRESHOLD:
            hazard_detected = True
            xmin, ymin, xmax, ymax = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
            label = f"{det['name'].upper()}: {det['confidence']:.2f}"
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # --- 5. ALERTING SYSTEM LOGIC (WITH DEBUGGING) ---
    if hazard_detected:
        print("-> Hazard Detected! Checking cooldown...")
        current_time = time.time()
        
        if current_time - last_alert_time > ALERT_COOLDOWN_SECONDS:
            print(f"   -> Cooldown passed ({int(current_time - last_alert_time)}s > {ALERT_COOLDOWN_SECONDS}s). Attempting to send alert.")
            if send_alert_email():
                last_alert_time = current_time
        else:
            print(f"   -> Cooldown active. {int(ALERT_COOLDOWN_SECONDS - (current_time - last_alert_time))}s remaining.")

    cv2.imshow('Fire & Smoke Detection System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print("Monitoring stopped.")