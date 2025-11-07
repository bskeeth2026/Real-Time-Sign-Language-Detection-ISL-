# final.py
import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow import keras
from collections import deque
import time

class ISLRealTimeDetector:
    def __init__(self, model_path='isl_model_best.h5', label_encoder_path='label_encoder.pkl', scaler_path='processed_data/scaler.pkl'):
        print("Loading model...")
        self.model = keras.models.load_model(model_path)

        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

        # Load scaler for normalization
        try:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("✅ Scaler loaded successfully!")
        except FileNotFoundError:
            print("⚠️ Warning: Scaler not found. Predictions may be less accurate.")
            self.scaler = None

        # MediaPipe with optimized parameters for speed
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Optimized for faster detection
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,  # Increased for more confident detections
            min_tracking_confidence=0.6,
            model_complexity=0  # Use lighter model for speed
        )

        # Improved smoothing with shorter buffer for faster response
        self.prediction_buffer = deque(maxlen=5)  # Reduced from 7
        self.confidence_threshold = 0.65  # Lowered for quicker detection
        self.fps_buffer = deque(maxlen=30)
        
        # Temporal confidence tracking
        self.last_predictions = deque(maxlen=3)  # Track last 3 predictions with confidence

        # constants matching preprocess
        self.LANDMARKS_PER_HAND = 21
        self.COORDS_PER_LANDMARK = 3
        self.FEATURES_PER_HAND = self.LANDMARKS_PER_HAND * self.COORDS_PER_LANDMARK
        self.TOTAL_FEATURES = self.FEATURES_PER_HAND * 2

        print("✅ Model loaded successfully!")
        print(f"Detecting {len(self.label_encoder.classes_)} alphabets: {', '.join(self.label_encoder.classes_)}")

    def _pad_or_order_hands(self, results):
        """Return fixed-length vector order: Left then Right (pad if missing)."""
        left = np.zeros(self.FEATURES_PER_HAND, dtype=np.float32)
        right = np.zeros(self.FEATURES_PER_HAND, dtype=np.float32)

        if not results or not results.multi_hand_landmarks:
            return np.concatenate([left, right])

        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # 'Left' or 'Right'
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            coords = np.array(coords, dtype=np.float32)
            if label == 'Left':
                left = coords
            else:
                right = coords

        return np.concatenate([left, right])

    def extract_landmarks(self, image):
        """Extract ordered/padded landmarks from frame; also return results to draw"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        features = self._pad_or_order_hands(results)
        return features if not np.all(features == 0) else None, results

    def predict_sign(self, landmarks):
        """Predict sign with normalized features"""
        if landmarks is None:
            return None, 0.0
        
        landmarks_reshaped = landmarks.reshape(1, -1)
        
        # Apply normalization if scaler is available
        if self.scaler is not None:
            landmarks_reshaped = self.scaler.transform(landmarks_reshaped)
        
        prediction = self.model.predict(landmarks_reshaped, verbose=0)
        confidence = float(np.max(prediction))
        predicted_class = int(np.argmax(prediction))
        predicted_label = self.label_encoder.classes_[predicted_class]
        
        # Track predictions with confidence
        self.last_predictions.append((predicted_label, confidence))
        
        return predicted_label, confidence

    def smooth_prediction(self, label, confidence):
        """Improved smoothing with temporal confidence averaging"""
        if confidence > self.confidence_threshold and label is not None:
            self.prediction_buffer.append(label)
        
        if len(self.prediction_buffer) > 0:
            # Get most common prediction
            unique, counts = np.unique(list(self.prediction_buffer), return_counts=True)
            most_common = unique[np.argmax(counts)]
            
            # Calculate average confidence for the most common prediction
            if len(self.last_predictions) > 0:
                recent_confs = [conf for pred, conf in self.last_predictions if pred == most_common]
                if recent_confs:
                    avg_conf = np.mean(recent_confs)
                    # Only return if average confidence is good
                    if avg_conf > self.confidence_threshold:
                        return most_common
        
        return None

    def draw_info(self, image, predicted_label, confidence, fps):
        """Draw information panel with improved visuals"""
        h, w, _ = image.shape
        panel_height = 120
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)

        if predicted_label:
            # Draw predicted letter
            cv2.putText(panel, predicted_label, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.5,
                        (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > self.confidence_threshold else (0, 165, 255), 4)
            
            # Confidence text and bar
            conf_text = f"Confidence: {confidence * 100:.1f}%"
            cv2.putText(panel, conf_text, (150, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            bar_width = int(300 * confidence)
            cv2.rectangle(panel, (150, 55), (450, 75), (60, 60, 60), -1)
            
            # Color-coded confidence bar
            bar_color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > self.confidence_threshold else (0, 165, 255)
            cv2.rectangle(panel, (150, 55), (150 + bar_width, 75), bar_color, -1)
            cv2.rectangle(panel, (150, 55), (450, 75), (255, 255, 255), 2)
        else:
            cv2.putText(panel, "No reliable prediction", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

        # FPS display
        cv2.putText(panel, f"FPS: {fps:.1f}", (w - 150, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Instructions
        cv2.putText(panel, "Press 'q' to quit | 'r' to reset buffer", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1)
        
        combined = np.vstack([panel, image])
        return combined

    def run(self):
        """Run real-time detection with optimizations"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Set desired FPS

        print("\n🎥 Starting webcam...")
        print("Press 'q' to quit")
        print("Press 'r' to reset prediction buffer")
        print("\n⚡ Optimizations enabled: Faster detection with improved accuracy!")

        frame_skip = 0  # Process every frame for responsiveness
        frame_count = 0

        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # mirror
            frame_count += 1

            predicted_label = None
            confidence = 0.0

            # Process every frame for better responsiveness
            if frame_count % (frame_skip + 1) == 0:
                landmarks, results = self.extract_landmarks(frame)

                if landmarks is not None:
                    # Draw all detected hands
                    if results.multi_hand_landmarks:
                        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                            # Draw hand landmarks
                            self.mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style()
                            )
                            
                            # Show which hand (Left/Right)
                            xs = [lm.x for lm in hand_landmarks.landmark]
                            ys = [lm.y for lm in hand_landmarks.landmark]
                            h, w = frame.shape[:2]
                            min_x = int(min(xs) * w)
                            min_y = int(min(ys) * h)
                            label_text = handedness.classification[0].label
                            cv2.putText(frame, label_text, (min_x, max(20, min_y - 10)), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    raw_label, confidence = self.predict_sign(landmarks)
                    predicted_label = self.smooth_prediction(raw_label, confidence)

            # Calculate FPS
            elapsed = time.time() - start_time
            self.fps_buffer.append(1 / elapsed if elapsed > 0 else 0)
            fps = np.mean(self.fps_buffer)

            display_frame = self.draw_info(frame, predicted_label, confidence, fps)
            cv2.imshow('ISL Alphabet Detection - Enhanced', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.prediction_buffer.clear()
                self.last_predictions.clear()
                print("Prediction buffer reset!")

        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Detection stopped!")

if __name__ == "__main__":
    try:
        detector = ISLRealTimeDetector(
            model_path='isl_model_best.h5',
            label_encoder_path='label_encoder.pkl',
            scaler_path='processed_data/scaler.pkl'
        )
        detector.run()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("1) Trained the model (run preprocess.py then train.py)")
        print("2) Files exist: isl_model_best.h5, label_encoder.pkl, and processed_data/scaler.pkl")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your webcam connection and try again.")
