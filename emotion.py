import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from collections import deque
from datetime import datetime

# Set global matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
mpl.rcParams['font.family'] = 'DejaVu Sans'

# Load emotion model
model = load_model('emotion_model_2.h5')

# Emotion labels and colors (updated colormap access)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_colors = plt.colormaps['RdYlGn'](np.linspace(0.1, 0.9, len(emotion_labels)))

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

# History tracking for trend visualization
history_length = 20
emotion_history = {label: deque([0]*history_length, maxlen=history_length) for label in emotion_labels}

# Thermal simulation parameters
min_temp = 20  # Minimum simulated temperature (°C)
max_temp = 40  # Maximum simulated temperature (°C)

# Create figure with multiple visualization panels
plt.ion()
fig = plt.figure(figsize=(16, 9), facecolor='#f5f5f5')
fig.canvas.manager.set_window_title('Advanced Emotion & Thermal Analysis Dashboard')

gs = GridSpec(3, 3, figure=fig)

# 1. Main bar chart
ax1 = fig.add_subplot(gs[0:2, 0:2])
bars = ax1.bar(emotion_labels, [0]*len(emotion_labels), 
              color=emotion_colors,
              edgecolor='white',
              linewidth=1.2)
ax1.set_ylim(0, 1)
ax1.set_title("Current Emotion Distribution", fontsize=12, fontweight='bold')
ax1.set_ylabel("Confidence Score")
plt.sca(ax1)
plt.xticks(rotation=15, ha='right')

# 2. Pie chart
ax2 = fig.add_subplot(gs[0, 2])
initial_pie_values = [0.01]*len(emotion_labels)
pie_wedges, pie_texts = ax2.pie(initial_pie_values, 
                               colors=emotion_colors,
                               radius=1.3,
                               startangle=90)
ax2.set_title("Emotion Proportion", fontsize=12, fontweight='bold')

# 3. Trend lines
ax3 = fig.add_subplot(gs[1, 2])
trend_lines = []
for i, label in enumerate(emotion_labels):
    line, = ax3.plot(range(history_length), [0]*history_length, 
                    color=emotion_colors[i], label=label, alpha=0.7)
    trend_lines.append(line)
ax3.set_xlim(0, history_length-1)
ax3.set_ylim(0, 1)
ax3.set_title("Emotion Trends Over Time", fontsize=12, fontweight='bold')
ax3.set_xlabel("Time (frames)")
ax3.legend(loc='upper right', bbox_to_anchor=(1.4, 1))

# 4. Radar chart
ax4 = fig.add_subplot(gs[2, 0:2], polar=True)
angles = np.linspace(0, 2*np.pi, len(emotion_labels), endpoint=False).tolist()
angles += angles[:1]
radar_line, = ax4.plot(angles, [0.01]*(len(emotion_labels)+1), 'o-', linewidth=2)
ax4.fill(angles, [0.01]*(len(emotion_labels)+1), alpha=0.25)
ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(emotion_labels)
ax4.set_title("Emotion Radar Profile", fontsize=12, fontweight='bold', y=1.1)
ax4.set_rlabel_position(30)
ax4.set_ylim(0, 1)

# 5. Dominant emotion display
ax5 = fig.add_subplot(gs[2, 2])
ax5.axis('off')
dominant_text = ax5.text(0.5, 0.5, "No Face Detected", 
                        ha='center', va='center', 
                        fontsize=16, fontweight='bold',
                        color='gray')
ax5.set_title("Current Dominant Emotion", fontsize=12, fontweight='bold')

plt.tight_layout()

def update_visualizations(prediction):
    global pie_wedges, pie_texts
    
    prediction = np.nan_to_num(prediction, nan=0.001)
    prediction = np.clip(prediction, 0.001, 1.0)
    
    for bar, h in zip(bars, prediction):
        bar.set_height(h)
    
    pie_values = np.maximum(prediction, 0.01)
    ax2.clear()
    pie_wedges, pie_texts = ax2.pie(pie_values, 
                                   colors=emotion_colors,
                                   radius=1.3,
                                   startangle=90)
    ax2.set_title("Emotion Proportion", fontsize=12, fontweight='bold')
    
    for wedge, val in zip(pie_wedges, pie_values):
        wedge.set_width(0.3 + 0.2 * (val / max(pie_values)))
    
    for i, label in enumerate(emotion_labels):
        emotion_history[label].append(prediction[i])
        trend_lines[i].set_ydata(emotion_history[label])
    
    values = prediction.tolist()
    values += values[:1]
    radar_line.set_ydata(values)
    
    while ax4.collections:
        ax4.collections[0].remove()
    ax4.fill(angles, values, alpha=0.25)
    
    dominant_idx = np.argmax(prediction)
    dominant_label = emotion_labels[dominant_idx]
    dominant_conf = prediction[dominant_idx]
    dominant_text.set_text(f"{dominant_label}\n({dominant_conf*100:.1f}%)")
    dominant_text.set_color(emotion_colors[dominant_idx])
    
    fig.canvas.draw()
    fig.canvas.flush_events()

def create_thermal_image(frame, faces):
    # Convert to grayscale and normalize
    gray_thermal = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Create temperature mask
    temp_mask = np.zeros_like(gray_thermal, dtype=np.float32)
    temp_mask.fill(min_temp)
    
    # Warmer areas around edges
    edges = cv2.Canny(gray_thermal, 100, 200)
    edges_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    temp_mask[edges_dilated > 0] = (min_temp + max_temp) / 2
    
    # Hot areas for faces
    for (x, y, w, h) in faces:
        center = (x + w//2, y + h//2)
        axes = (w//2, h//2)
        face_mask = np.zeros_like(gray_thermal)
        cv2.ellipse(face_mask, center, axes, 0, 0, 360, 255, -1)
        temp_mask[face_mask > 0] = max_temp
    
    # Normalize temperature to 0-255 range
    temp_normalized = ((temp_mask - min_temp) / (max_temp - min_temp) * 255).astype(np.uint8)
    
    # Apply thermal colormap
    thermal_frame = cv2.applyColorMap(temp_normalized, cv2.COLORMAP_JET)
    
    # Add HUD information
    height, width = frame.shape[:2]
    current_time = datetime.now().strftime("%H:%M:%S")
    cv2.putText(thermal_frame, current_time, (width - 150, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(thermal_frame, f"{max_temp}C", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(thermal_frame, f"{min_temp}C", (20, height - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return thermal_frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Create thermal image
    thermal_frame = create_thermal_image(frame, faces)
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (48, 48))
        roi_normalized = roi_resized.astype('float32') / 255.0
        roi_reshaped = np.expand_dims(roi_normalized, axis=0)
        roi_reshaped = np.expand_dims(roi_reshaped, axis=-1)

        prediction = model.predict(roi_reshaped, verbose=0)[0]
        
        # Draw face bounding box on both frames
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 255), 3)
        cv2.rectangle(thermal_frame, (x, y), (x+w, y+h), (255, 255, 255), 3)
        
        emotion_index = np.argmax(prediction)
        text = f'{emotion_labels[emotion_index]} ({prediction[emotion_index]*100:.1f}%)'
        cv2.putText(frame, text, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, 
                   (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(thermal_frame, text, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, 
                   (255, 255, 255), 2, cv2.LINE_AA)
        
        update_visualizations(prediction)
    else:
        dominant_text.set_text("No Face Detected")
        dominant_text.set_color('gray')
        update_visualizations(np.array([0.001]*len(emotion_labels)))

    # Combine original and thermal views
    combined = np.hstack((frame, thermal_frame))
    cv2.imshow('Emotion Recognition & Thermal Simulation', combined)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"emotion_thermal_{timestamp}.jpg", combined)
        print("Screenshot saved!")

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.close()