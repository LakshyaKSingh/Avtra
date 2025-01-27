import cv2
import numpy as np
import torch
import mediapipe as mp
from pathlib import Path
import time
import sys


class AnimeGANAvatar:
    def __init__(self):
        Path("captured_images").mkdir(exist_ok=True)

        # Use local model path
        model_path = Path("E:/Avatar/models/bryandlee_animegan2-pytorch_main")

        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {self.device}")

            # Load model directly from local files
            sys.path.append(str(model_path))
            from model import Generator

            self.model = Generator()
            weights = torch.load(
                model_path / "weights" / "face_paint_512_v2.pt",
                map_location=self.device
            )
            self.model.load_state_dict(weights)
            self.model.to(self.device).eval()
            print("Model loaded successfully from local files!")

        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"Make sure model files exist in: {model_path}")
            raise

        # Initialize Mediapipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_eye_regions(self, frame):
        """Extract and enhance both eye regions."""
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)

            left_eye, right_eye = None, None
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Left and right eye landmarks
                    left_eye_indices = [33, 160, 158, 133, 153, 144]
                    right_eye_indices = [362, 385, 387, 263, 373, 380]

                    left_eye = [
                        (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                        for idx, landmark in enumerate(face_landmarks.landmark)
                        if idx in left_eye_indices
                    ]
                    right_eye = [
                        (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                        for idx, landmark in enumerate(face_landmarks.landmark)
                        if idx in right_eye_indices
                    ]

            return left_eye, right_eye
        except Exception as e:
            print(f"Error extracting eye regions: {e}")
            return None, None

    def enhance_eye(self, frame, eye_points):
        """Apply enhancements to a single eye region."""
        try:
            if eye_points is not None:
                # Create a mask for the eye
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(eye_points, np.int32)], 255)

                # Extract and enhance the eye region
                eye_region = cv2.bitwise_and(frame, frame, mask=mask)
                eye_region = cv2.GaussianBlur(eye_region, (15, 15), 30)  # Enhance eyes
                enhanced_eye = cv2.addWeighted(frame, 0.7, eye_region, 0.3, 0)

                # Combine back into the original frame
                frame[mask > 0] = enhanced_eye[mask > 0]
            return frame
        except Exception as e:
            print(f"Error enhancing eye: {e}")
            return frame

    def process_frame(self, frame):
        """Process a single video frame."""
        left_eye, right_eye = self.extract_eye_regions(frame)
        if left_eye is not None and right_eye is not None:
            frame = self.enhance_eye(frame, left_eye)
            frame = self.enhance_eye(frame, right_eye)

        try:
            # Resize frame directly to 512x512 for model input
            img = cv2.resize(frame, (512, 512))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Convert to tensor for model inference
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
            img = img.to(self.device) / 127.5 - 1.0

            # Generate anime style
            with torch.no_grad():
                out = self.model(img)

            # Convert back to image
            out = out[0].permute(1, 2, 0).cpu().numpy()
            out = (out + 1) * 127.5
            out = np.clip(out, 0, 255).astype(np.uint8)
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

            return out

        except Exception as e:
            print(f"Error in process_frame: {e}")
            return frame

    def process_video(self):
        """Capture video and apply processing."""
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            anime_frame = self.process_frame(frame)

            cv2.imshow('Original', frame)
            cv2.imshow('Anime Style', anime_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                original_path = f"captured_images/original_{timestamp}.jpg"
                anime_path = f"captured_images/anime_{timestamp}.jpg"
                cv2.imwrite(original_path, frame)
                cv2.imwrite(anime_path, anime_frame)
                print(f"Images captured at timestamp: {timestamp}")
                print(f"Original image saved at: {original_path}")
                print(f"Anime image saved at: {anime_path}")

        cap.release()
        cv2.destroyAllWindows()

    def list_saved_images(self):
        """List all saved images in the captured_images directory."""
        return list(Path("captured_images").glob("*.jpg"))


if __name__ == "__main__":
    try:
        avatar = AnimeGANAvatar()
        avatar.process_video()
    except Exception as e:
        print(f"Error: {e}")
