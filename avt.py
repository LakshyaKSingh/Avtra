import streamlit as st
import cv2
import numpy as np
import torch
from AnimeGANAvatar import AnimeGANAvatar
from pathlib import Path

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
option = st.sidebar.selectbox("Choose an option:", ["Upload Image", "Use Webcam"])

# Create a directory to save images if it doesn't exist
output_dir = Path("captured_images")
output_dir.mkdir(exist_ok=True)

# Function to apply black and white filter
def apply_bw_filter(image):
    """Apply enhanced black and white filter to the image."""
    bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply a contrast adjustment
    bw_image = cv2.convertScaleAbs(bw_image, alpha=1.5, beta=0)  # Increase contrast
    return cv2.cvtColor(bw_image, cv2.COLOR_GRAY2BGR)  # Convert back to 3 channels

# Function to create a face mask
def create_face_mask(image, face_coordinates):
    """Create a mask for the detected face region."""
    mask = np.zeros_like(image)
    for (x, y, w, h) in face_coordinates:
        mask[y:y+h, x:x+w] = image[y:y+h, x:x+w]
    return mask

# Real-time image processing and display
def display_images_side_by_side(original, anime):
    """Display original and anime-style images side-by-side."""
    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption="Original Image", channels="BGR")
    with col2:
        st.image(anime, caption="Anime Style", channels="BGR")

def overlay_with_autofit(image, overlay_image, face_coordinates):
    """Overlay an image on the detected face region with autofit."""
    for (x, y, w, h) in face_coordinates:
        overlay_resized = cv2.resize(overlay_image, (w, h), interpolation=cv2.INTER_AREA)  # Resize to fit the face
        for i in range(h):
            for j in range(w):
                if overlay_resized[i, j][3] != 0:  # Check alpha channel
                    image[y + i, x + j] = overlay_resized[i, j][:3]  # Overlay image
    return image

def overlay_sunglasses(image, face_coordinates):
    """Overlay the sunglasses image on the detected eye region."""
    for (x, y, w, h) in face_coordinates:
        sunglasses_resized = cv2.resize(sunglasses_image, (w, int(h / 4)), interpolation=cv2.INTER_AREA)  # Resize sunglasses
        y_offset = y + int(h / 4)  # Position sunglasses on the eyes
        for i in range(sunglasses_resized.shape[0]):
            for j in range(sunglasses_resized.shape[1]):
                if sunglasses_resized[i, j][3] != 0:  # Check alpha channel
                    image[y_offset + i, x + j] = sunglasses_resized[i, j][:3]
    return image

def overlay_thug_life(image, face_coordinates):
    """Overlay the thug life image on the detected eye region."""
    for (x, y, w, h) in face_coordinates:
        thug_life_resized = cv2.resize(thug_life_image, (w, int(h / 4)), interpolation=cv2.INTER_AREA)  # Resize thug life image
        y_offset = y + int(h / 4)  # Position thug life image on the eyes
        for i in range(thug_life_resized.shape[0]):
            for j in range(thug_life_resized.shape[1]):
                if thug_life_resized[i, j][3] != 0:  # Check alpha channel
                    image[y_offset + i, x + j] = thug_life_resized[i, j][:3]
    return image

# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Filter selection before processing
filter_option = st.sidebar.selectbox("Choose filter:", ["Avatar", "Thug Life", "Spiderman", "Sunglasses", "Alien", "Robot", "B/W"])

# Process uploaded image
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Detect faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Initialize img_with_overlay
        img_with_overlay = img.copy()

        # Apply the selected option
        if len(faces) > 0:
            if filter_option == "B/W":
                anime_image = apply_bw_filter(img.copy())
            elif filter_option == "Robot":
                img_with_overlay = overlay_with_autofit(img.copy(), robot_face_image, faces)
                anime_image = img_with_overlay
            elif filter_option == "Sunglasses":
                img_with_overlay = overlay_sunglasses(img.copy(), faces)
                anime_image = img_with_overlay
            elif filter_option == "Thug Life":
                img_with_overlay = overlay_thug_life(img.copy(), faces)
                anime_image = img_with_overlay
            elif filter_option == "Spiderman":
                img_with_overlay = overlay_with_autofit(img.copy(), spiderman_image, faces)
                anime_image = img_with_overlay
            elif filter_option == "Alien":
                img_with_overlay = overlay_with_autofit(img.copy(), alien_image, faces)
                anime_image = img_with_overlay
            else:
                img_with_overlay = img.copy()
                anime_image = avatar_generator.process_frame(img_with_overlay)
        else:
            img_with_overlay = img
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

        # Detect faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Initialize img_with_overlay
        img_with_overlay = img.copy()

        # Apply the selected option
        if len(faces) > 0:
            if filter_option == "Robot":
                img_with_overlay = overlay_with_autofit(img.copy(), robot_face_image, faces)
                anime_image = img_with_overlay
            elif filter_option == "Sunglasses":
                img_with_overlay = overlay_sunglasses(img.copy(), faces)
                anime_image = img_with_overlay
            elif filter_option == "Thug Life":
                img_with_overlay = overlay_thug_life(img.copy(), faces)
                anime_image = img_with_overlay
            elif filter_option == "Spiderman":
                img_with_overlay = overlay_with_autofit(img.copy(), spiderman_image, faces)
                anime_image = img_with_overlay
            elif filter_option == "Alien":
                img_with_overlay = overlay_with_autofit(img.copy(), alien_image, faces)
                anime_image = img_with_overlay
            elif filter_option == "B/W":
                anime_image = apply_bw_filter(img.copy())
            else:
                img_with_overlay = img.copy()
                anime_image = avatar_generator.process_frame(img_with_overlay)
        else:
            img_with_overlay = img
            anime_image = avatar_generator.process_frame(img_with_overlay)

        # Add download button for processed image
        st.download_button("Download Processed Image", data=cv2.imencode('.png', anime_image)[1].tobytes(), file_name="processed_image.png", mime="image/png")

        # Display images side-by-side
        display_images_side_by_side(img, anime_image)  # Use the original img for display
