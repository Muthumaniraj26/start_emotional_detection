import cv2
import numpy as np
from datetime import datetime

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Create windows with adjusted properties
    cv2.namedWindow('Thermal Simulation', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Thermal Simulation', 800, 600)
    cv2.resizeWindow('Original', 800, 600)
    
    # Parameters for motion detection
    prev_frame = None
    motion_threshold = 5000
    motion_detected = False
    motion_counter = 0
    
    # Parameters for temperature simulation
    min_temp = 20  # Minimum simulated temperature (°C)
    max_temp = 40  # Maximum simulated temperature (°C)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # ----------------------------
        # Motion Detection
        # ----------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if prev_frame is None:
            prev_frame = gray
            continue
        
        frame_diff = cv2.absdiff(prev_frame, gray)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > motion_threshold:
                motion_detected = True
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        prev_frame = gray
        
        # ----------------------------
        # Thermal Simulation
        # ----------------------------
        # Convert to grayscale and normalize
        gray_thermal = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Simulate temperature variations
        # - Face region will be warmer
        # - Edge regions will be cooler
        
        # Detect faces for temperature simulation
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_thermal, 1.1, 4)
        
        # Create temperature mask
        temp_mask = np.zeros_like(gray_thermal, dtype=np.float32)
        
        # Base temperature (cool)
        temp_mask.fill(min_temp)
        
        # Warmer areas around edges (simulating body heat)
        edges = cv2.Canny(gray_thermal, 100, 200)
        edges_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        temp_mask[edges_dilated > 0] = (min_temp + max_temp) / 2
        
        # Hot areas for faces
        for (x, y, w, h) in faces:
            # Create oval mask for face
            center = (x + w//2, y + h//2)
            axes = (w//2, h//2)
            face_mask = np.zeros_like(gray_thermal)
            cv2.ellipse(face_mask, center, axes, 0, 0, 360, 255, -1)
            temp_mask[face_mask > 0] = max_temp
        
        # Normalize temperature to 0-255 range
        temp_normalized = ((temp_mask - min_temp) / (max_temp - min_temp) * 255).astype(np.uint8)
        
        # Apply thermal colormap
        thermal_frame = cv2.applyColorMap(temp_normalized, cv2.COLORMAP_JET)
        
        # ----------------------------
        # Add HUD (Heads-Up Display) Information
        # ----------------------------
        # Current time
        current_time = datetime.now().strftime("%H:%M:%S")
        cv2.putText(thermal_frame, current_time, (width - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Temperature scale
        cv2.putText(thermal_frame, f"{max_temp}C", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(thermal_frame, f"{min_temp}C", (20, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Motion detection status
        motion_status = "MOTION DETECTED" if motion_detected else "STATIC SCENE"
        motion_color = (0, 0, 255) if motion_detected else (0, 255, 0)
        cv2.putText(thermal_frame, motion_status, (width // 2 - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, motion_color, 2)
        
        # Face detection count
        cv2.putText(thermal_frame, f"Faces: {len(faces)}", (width - 150, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # ----------------------------
        # Display Results
        # ----------------------------
        # Combine original and thermal views
        combined = np.hstack((frame, thermal_frame))
        
        # Show the combined view
        cv2.imshow('Thermal Camera Simulation', combined)
        
        # Exit on 'q' key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"thermal_simulation_{timestamp}.jpg", combined)
            print("Screenshot saved!")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()