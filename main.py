import streamlit as st
import numpy as np
import os
import requests
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
import imutils

# Create directories for storing images
os.makedirs('pan_card_tampering/image', exist_ok=True)

# Function to load image from URL or file
def load_image(uploaded_file, url=None):
    if uploaded_file:
        return Image.open(uploaded_file)
    elif url:
        return Image.open(requests.get(url, stream=True).raw)
    else:
        return None

# Title and description
st.title('PAN Card Tampering Detection')
st.write("""
Upload the original and tampered PAN card images to detect differences.
""")

# File uploaders for original and tampered images
uploaded_original = st.file_uploader("Upload Original Image", type=['jpg', 'jpeg', 'png'])
uploaded_tampered = st.file_uploader("Upload Tampered Image", type=['jpg', 'jpeg', 'png'])

# Default URLs for images
original_url = 'https://www.thestatesman.com/wp-content/uploads/2019/07/pan-card.jpg'
tampered_url = 'https://assets1.cleartax-cdn.com/s/img/20170526124335/Pan4.png'

# Load images
original = load_image(uploaded_file=uploaded_original, url=original_url if not uploaded_original else None)
tampered = load_image(uploaded_file=uploaded_tampered, url=tampered_url if not uploaded_tampered else None)

if original and tampered:
    # Display images
    st.subheader('Original Image')
    st.image(original, caption='Original Image', use_column_width=True)
    st.subheader('Tampered Image')
    st.image(tampered, caption='Tampered Image', use_column_width=True)

    # Print image details
    st.write("Original image format: ", original.format)
    st.write("Tampered image format: ", tampered.format)
    st.write("Original image size: ", original.size)
    st.write("Tampered image size: ", tampered.size)

    # Resize images
    original = original.resize((250, 160))
    tampered = tampered.resize((250, 160))
    original.save('pan_card_tampering/image/original.png')
    tampered.save('pan_card_tampering/image/tampered.png')

    # Convert images to OpenCV format
    original = cv2.imread('pan_card_tampering/image/original.png')
    tampered = cv2.imread('pan_card_tampering/image/tampered.png')

    # Convert to grayscale
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two images
    (score, diff) = ssim(original_gray, tampered_gray, full=True)
    diff = (diff * 255).astype("uint8")
    st.write("SSIM: {}".format(score))

    # Threshold the difference image
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Find contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Draw rectangles on the images to highlight differences
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(tampered, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Convert images to PIL format for display in Streamlit
    original_image = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    tampered_image = Image.fromarray(cv2.cvtColor(tampered, cv2.COLOR_BGR2RGB))
    diff_image = Image.fromarray(diff)
    thresh_image = Image.fromarray(thresh)

    # Display images with differences highlighted
    st.subheader('Original Image with Differences Highlighted')
    st.image(original_image, caption='Original Image with Differences Highlighted', use_column_width=True)
    st.subheader('Tampered Image with Differences Highlighted')
    st.image(tampered_image, caption='Tampered Image with Differences Highlighted', use_column_width=True)
    st.subheader('Difference Image')
    st.image(diff_image, caption='Difference Image', use_column_width=True)
    st.subheader('Threshold Image')
    st.image(thresh_image, caption='Threshold Image', use_column_width=True)
else:
    st.warning("Please upload both the original and tampered images.")
