import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from gtts import gTTS
from playsound import playsound
import os

import time
from typing import Optional

# Popup configuration
POPUP_DURATION = 2.0  # seconds to display the transient popup

def speak(text):
    """Converts text to speech and plays it."""
    try:
        tts = gTTS(text=text, lang='en')
        # Use a temporary file for the audio
        audio_file = "temp_audio.mp3"
        tts.save(audio_file)
        playsound(audio_file)
        os.remove(audio_file)
    except Exception as e:
        st.error(f"Error in TTS: {e}")

def show_detection_popup(label: str, conf: float, severity: str = 'warning') -> None:
    """Record a detection in session state so the UI can show a transient popup.

    severity: 'warning'|'info' - determines whether the UI shows a warning or info popup.
    Also records a `pending_tts` text that can be consumed by a later text-to-speech step.
    """
    now = time.time()
    st.session_state['last_detection'] = {
        'label': label,
        'conf': conf,
        'time': now,
        'severity': severity,
    }
    
    # Cooldown to avoid speaking too often for the same object
    COOLDOWN_SECONDS = 5.0
    
    last_spoken_time = st.session_state.get('last_spoken_time', 0.0)
    last_spoken_label = st.session_state.get('last_spoken_label')

    if label != last_spoken_label or (now - last_spoken_time) > COOLDOWN_SECONDS:
        tts_text = f"{label} is prohibited, please collect"
        st.session_state['pending_tts'] = tts_text
        speak(tts_text)
        st.session_state['last_spoken_label'] = label
        st.session_state['last_spoken_time'] = now


# Ensure session state key exists
if 'last_detection' not in st.session_state:
    st.session_state['last_detection'] = None
if 'pending_tts' not in st.session_state:
    st.session_state['pending_tts'] = None
if 'last_spoken_label' not in st.session_state:
    st.session_state['last_spoken_label'] = None
if 'last_spoken_time' not in st.session_state:
    st.session_state['last_spoken_time'] = 0.0
if 'collected_items' not in st.session_state:
    st.session_state['collected_items'] = []
if 'session_ended' not in st.session_state:
    st.session_state['session_ended'] = False
if 'collecting_item' not in st.session_state:
    st.session_state['collecting_item'] = None

import collections

# Load your YOLO model
model = YOLO("best.pt")

st.title("Exam Classroom Scanner")

if st.button("End Exam Scan"):
    st.session_state.session_ended = True

if st.session_state.session_ended:
    st.header("Collected Items Report")
    if not st.session_state.collected_items:
        st.write("No prohibited items were collected during the scan.")
    else:
        item_counts = collections.Counter(item['label'] for item in st.session_state.collected_items)
        st.table(item_counts)
else:
    st.write("This application uses a YOLO model to scan for prohibited items in an exam setting.")

    # Placeholders
    spoken_text_placeholder = st.empty()
    collect_button_placeholder = st.empty()
    collection_status_placeholder = st.empty()

    # Handle collection timer and status
    if st.session_state.collecting_item:
        collection_start_time = st.session_state.collecting_item['collection_start_time']
        elapsed_time = time.time() - collection_start_time
        if elapsed_time < 5:
            collection_status_placeholder.info(f"Collecting {st.session_state.collecting_item['label']}... {5 - int(elapsed_time)}s remaining")
        else:
            st.session_state.collected_items.append(st.session_state.collecting_item)
            collection_status_placeholder.success(f"Collected {st.session_state.collecting_item['label']}!")
            st.session_state.collecting_item = None
            st.session_state.last_detection = None # Clear last detection to hide collect button
            collection_status_placeholder.empty()
            st.experimental_rerun()

    # Start/Stop checkbox
    run = st.checkbox("Start Camera")

    # Placeholder with dummy image
    frame_window = st.image(np.zeros((480, 640, 3), dtype=np.uint8), caption="Live Camera Detection")

    # Render transient popup if a recent detection was recorded
    last = st.session_state.get('last_detection')
    if last is not None and (time.time() - last.get('time', 0.0)) < POPUP_DURATION:
        if last.get('severity', 'info') == 'warning':
            st.warning(f"Detected: {last['label']} ({last['conf']:.2f})", icon="⚠️")
        else:
            st.info(f"Detected: {last['label']} ({last['conf']:.2f})", icon="📣")

    # Initialize webcam
    camera = cv2.VideoCapture(0)

    while run:
        success, frame = camera.read()
        if not success:
            st.write("Failed to access webcam.")
            break

        # Update spoken text display
        if st.session_state.get('pending_tts'):
            spoken_text_placeholder.write(f"Speaking: {st.session_state['pending_tts']}")

        # Run YOLO prediction
        results = model.predict(frame, conf=0.5)

        # Plot detections
        annotated_frame = results[0].plot()

        # If there are detections, extract the first label+confidence and show popup
        try:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                # Attempt robust extraction of class and confidence
                try:
                    cls_list = boxes.cls.cpu().numpy().astype(int).tolist()
                    conf_list = boxes.conf.cpu().numpy().tolist()
                except Exception:
                    try:
                        cls_list = boxes.cls.numpy().astype(int).tolist()
                        conf_list = boxes.conf.numpy().tolist()
                    except Exception:
                        cls_list = [int(x) for x in boxes.cls]
                        conf_list = [float(x) for x in boxes.conf]

                cls0 = cls_list[0]
                conf0 = conf_list[0]
                label = model.names.get(cls0, str(cls0)) if hasattr(model, 'names') else str(cls0)
                # This will update st.session_state.last_detection
                show_detection_popup(label, conf0, severity='warning')

                # Display collect button if an item is detected
                if not st.session_state.collecting_item:
                    if collect_button_placeholder.button(f"Collect {label}"):
                        st.session_state.collecting_item = {
                            'label': label,
                            'collection_start_time': time.time()
                        }
                        collect_button_placeholder.empty()
                        st.experimental_rerun()

        except Exception:
            # Non-fatal: if extraction fails, silently continue
            pass

        # Convert BGR to RGB
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Update streamlit window
        frame_window.image(annotated_frame)

    camera.release()
