# preprocess.py
import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

class ISLDataPreprocessor:
    def __init__(self, dataset_path='sign_datasets'):
        self.dataset_path = dataset_path
        self.mp_hands = mp.solutions.hands
        # allow up to 2 hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        # constants
        self.LANDMARKS_PER_HAND = 21
        self.COORDS_PER_LANDMARK = 3
        self.FEATURES_PER_HAND = self.LANDMARKS_PER_HAND * self.COORDS_PER_LANDMARK
        self.TOTAL_FEATURES = self.FEATURES_PER_HAND * 2  # left + right
        self.scaler = StandardScaler()

    def _pad_or_order_hands(self, results):
        """
        Return a flattened array of length TOTAL_FEATURES.
        Order is: Left hand (21*3), then Right hand (21*3).
        If a hand is missing, its part is zero-padded.
        """
        left = np.zeros(self.FEATURES_PER_HAND, dtype=np.float32)
        right = np.zeros(self.FEATURES_PER_HAND, dtype=np.float32)

        if not results or not results.multi_hand_landmarks:
            return np.concatenate([left, right])

        # results.multi_hand_landmarks and results.multi_handedness are parallel lists
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

    def extract_landmarks(self, image_path):
        """Extract ordered/padded landmarks from image (returns 126-length vector or None)"""
        image = cv2.imread(image_path)
        if image is None:
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        # Always return fixed-length vector (padding if necessary)
        features = self._pad_or_order_hands(results)
        if np.all(features == 0):
            return None
        return features

    def augment_image(self, image_path, num_augmentations=25):
        """Create augmented versions of the image with more diverse transformations"""
        image = cv2.imread(image_path)
        if image is None:
            return []

        augmented_landmarks = []
        h, w = image.shape[:2]

        for _ in range(num_augmentations):
            aug_img = image.copy()
            
            # Random rotation (-20 to 20 degrees)
            angle = np.random.uniform(-20, 20)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            # Random brightness/contrast (wider range)
            alpha = 1.0 + np.random.uniform(-0.2, 0.2)
            beta = np.random.uniform(-30, 30)
            aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)

            # Random horizontal flip (50%)
            if np.random.rand() > 0.5:
                aug_img = cv2.flip(aug_img, 1)

            # Add Gaussian noise for robustness
            if np.random.rand() > 0.5:
                noise = np.random.randn(h, w, 3) * 10
                aug_img = np.clip(aug_img + noise, 0, 255).astype(np.uint8)

            # Random scaling (zoom in/out slightly)
            scale = np.random.uniform(0.9, 1.1)
            new_w, new_h = int(w * scale), int(h * scale)
            aug_img = cv2.resize(aug_img, (new_w, new_h))
            
            # Crop or pad back to original size
            if scale > 1.0:
                start_x = (new_w - w) // 2
                start_y = (new_h - h) // 2
                aug_img = aug_img[start_y:start_y+h, start_x:start_x+w]
            else:
                pad_x = (w - new_w) // 2
                pad_y = (h - new_h) // 2
                aug_img = cv2.copyMakeBorder(aug_img, pad_y, h-new_h-pad_y, pad_x, w-new_w-pad_x, cv2.BORDER_REFLECT)

            image_rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            features = self._pad_or_order_hands(results)
            if not np.all(features == 0):
                augmented_landmarks.append(features)

        return augmented_landmarks

    def process_dataset(self, augment=True, num_augmentations=25):
        """Process entire dataset and extract landmarks"""
        X = []
        y = []
        print("Processing dataset...")
        alphabet_folders = sorted([f for f in os.listdir(self.dataset_path)
                                   if os.path.isdir(os.path.join(self.dataset_path, f))])

        for alphabet in alphabet_folders:
            folder_path = os.path.join(self.dataset_path, alphabet)
            print(f"Processing alphabet: {alphabet}")
            for image_name in os.listdir(folder_path):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder_path, image_name)
                    landmarks = self.extract_landmarks(image_path)
                    if landmarks is not None:
                        X.append(landmarks)
                        y.append(alphabet)
                    if augment:
                        augmented = self.augment_image(image_path, num_augmentations)
                        for aug in augmented:
                            X.append(aug)
                            y.append(alphabet)

        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        
        # Normalize features for better training
        print("\nNormalizing features...")
        X = self.scaler.fit_transform(X)
        
        print(f"\nDataset processed!")
        print(f"Total samples: {len(X)}")
        print(f"Feature shape: {X.shape}  (expected features per sample: {self.TOTAL_FEATURES})")
        print(f"Unique labels: {len(np.unique(y))}")
        return X, y

    def save_processed_data(self, X, y, output_dir='processed_data'):
        """Save processed data"""
        os.makedirs(output_dir, exist_ok=True)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'y_val.npy'), y_val)

        unique_labels = sorted(np.unique(y))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        with open(os.path.join(output_dir, 'label_map.json'), 'w') as f:
            json.dump(label_map, f)

        # Save scaler for inference
        with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)

        print(f"\nData saved to '{output_dir}' directory")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        return X_train, X_val, y_train, y_val, label_map

if __name__ == "__main__":
    preprocessor = ISLDataPreprocessor(dataset_path='sign_datasets')
    X, y = preprocessor.process_dataset(augment=True, num_augmentations=25)
    X_train, X_val, y_train, y_val, label_map = preprocessor.save_processed_data(X, y)
    # Save an example of feature length for reference
    with open('processed_data/feature_info.json', 'w') as f:
        json.dump({"features_per_sample": preprocessor.TOTAL_FEATURES}, f)
    print("\n✅ Step 1 Complete! Ready for model training.")
