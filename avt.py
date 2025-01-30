import streamlit as st
import cv2
import numpy as np
import torch
from AnimeGANAvatar import AnimeGANAvatar
from pathlib import Path
import mediapipe as mp

# Load the robot face, sunglasses, thug life, spiderman, and alien images
robot_face_image = cv2.imread("robot_face.png", cv2.IMREAD_UNCHANGED)
sunglasses_image = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED)
thug_life_image = cv2.imread("image.png", cv2.IMREAD_UNCHANGED)
spiderman_image = cv2.imread("spiderman.png", cv2.IMREAD_UNCHANGED)
alien_image = cv2.imread("alien.png", cv2.IMREAD_UNCHANGED)

# Initialize the AnimeGANAvatar class
avatar_generator = AnimeGANAvatar()

# Streamlit app title and description
st.title("AnimeGAN Avatar Generator")
st.write("Generate anime-style avatars with fun filters. Upload images or use your webcam to see live previews and capture your filtered avatar.")

# Sidebar options
filter_option = st.sidebar.selectbox("Choose filter:", ["Avatar", "Robot", "Sunglasses", "Thug Life", "Spiderman", "Alien", "B/W"])
option = st.sidebar.selectbox("Choose an option:", ["Upload Image", "Use Webcam"])

# Create a directory to save images if it doesn't exist
output_dir = Path("captured_images")
output_dir.mkdir(exist_ok=True)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Function to apply black and white filter
def apply_bw_filter(image):
    """Apply enhanced black and white filter to the image."""
    bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply a contrast adjustment
    bw_image = cv2.convertScaleAbs(bw_image, alpha=1.5, beta=0)  # Increase contrast
    return cv2.cvtColor(bw_image, cv2.COLOR_GRAY2BGR)  # Convert back to 3 channels

# Function to overlay images on detected faces
def overlay_with_autofit(image, overlay_image, face_coordinates, is_lens_filter=False):
    """Overlay an image on the detected face region with autofit."""
    for (x, y, w, h) in face_coordinates:
        if is_lens_filter:
            # Resize overlay to fit only the eye region
            overlay_resized = cv2.resize(overlay_image, (w, int(h / 4)), interpolation=cv2.INTER_AREA)  # Resize to fit the eye region
            y_offset = y + int(h / 4)  # Position overlay on the eyes
            for i in range(overlay_resized.shape[0]):
                for j in range(overlay_resized.shape[1]):
                    if overlay_resized[i, j][3] != 0:  # Check alpha channel
                        image[y_offset + i, x + j] = overlay_resized[i, j][:3]  # Overlay image
        else:
            # Resize overlay to cover the entire face
            overlay_resized = cv2.resize(overlay_image, (w, h + int(h / 4)), interpolation=cv2.INTER_AREA)  # Resize to fit the face
            for i in range(h + int(h / 4)):
                for j in range(w):
                    if overlay_resized[i, j][3] != 0:  # Check alpha channel
                        image[y + i - int(h / 4), x + j] = overlay_resized[i, j][:3]  # Overlay image
    return image

# Real-time image processing and display
def display_images_side_by_side(original, anime):
    """Display original and anime-style images side-by-side."""
    original_resized = cv2.resize(original, (anime.shape[1], anime.shape[0]))  # Resize original to match anime size
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_resized, caption="Original Image", channels="BGR", use_container_width=True)
    with col2:
        st.image(anime, caption=filter_option, channels="BGR", use_container_width=True)

# Process uploaded image
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Detect faces using MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img_rgb)

        # Initialize img_with_overlay
        img_with_overlay = img.copy()

        # Apply the selected option
        if results.detections:
            face_coordinates = []
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = img.shape
                x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                face_coordinates.append((x, y, width, height))

            # Apply filters based on the selected option
            if filter_option == "Robot":
                img_with_overlay = overlay_with_autofit(img.copy(), robot_face_image, face_coordinates)
            elif filter_option == "Sunglasses":
                img_with_overlay = overlay_with_autofit(img.copy(), sunglasses_image, face_coordinates, is_lens_filter=True)
            elif filter_option == "Thug Life":
                img_with_overlay = overlay_with_autofit(img.copy(), thug_life_image, face_coordinates, is_lens_filter=True)
            elif filter_option == "Spiderman":
                img_with_overlay = overlay_with_autofit(img.copy(), spiderman_image, face_coordinates)
            elif filter_option == "Alien":
                img_with_overlay = overlay_with_autofit(img.copy(), alien_image, face_coordinates)
            elif filter_option == "B/W":
                img_with_overlay = apply_bw_filter(img.copy())

            # Process the image with anime style
            anime_image = avatar_generator.process_frame(img_with_overlay)

        # Add download button for processed image
        st.download_button("Download Processed Image", data=cv2.imencode('.png', anime_image)[1].tobytes(), file_name="processed_image.png", mime="image/png")

        # Display images side-by-side
        display_images_side_by_side(img, anime_image)  # Use the original img for display

# Webcam processing
elif option == "Use Webcam":
    st.write("Capture an image from your webcam to generate an anime-style avatar.")
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        # Read the captured image
        file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Detect faces using MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img_rgb)

        # Initialize img_with_overlay
        img_with_overlay = img.copy()

        # Apply the selected option
        if results.detections:
            face_coordinates = []
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = img.shape
                x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                face_coordinates.append((x, y, width, height))

            # Apply filters based on the selected option
            if filter_option == "Robot":
                img_with_overlay = overlay_with_autofit(img.copy(), robot_face_image, face_coordinates)
            elif filter_option == "Sunglasses":
                img_with_overlay = overlay_with_autofit(img.copy(), sunglasses_image, face_coordinates, is_lens_filter=True)
            elif filter_option == "Thug Life":
                img_with_overlay = overlay_with_autofit(img.copy(), thug_life_image, face_coordinates, is_lens_filter=True)
            elif filter_option == "Spiderman":
                img_with_overlay = overlay_with_autofit(img.copy(), spiderman_image, face_coordinates)
            elif filter_option == "Alien":
                img_with_overlay = overlay_with_autofit(img.copy(), alien_image, face_coordinates)
            elif filter_option == "B/W":
                img_with_overlay = apply_bw_filter(img.copy())

            # Process the image with anime style
            anime_image = avatar_generator.process_frame(img_with_overlay)

        # Add download button for processed image
        st.download_button("Download Processed Image", data=cv2.imencode('.png', anime_image)[1].tobytes(), file_name="processed_image.png", mime="image/png")

        # Display images side-by-side
        display_images_side_by_side(img, anime_image)  # Use the original img for display
