import streamlit as st
from PIL import Image
import torch
from transformers import pipeline
import tempfile
import os

# Load object detection model once
@st.cache_resource
def load_detection_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# Load text generation model once
@st.cache_resource
def load_text_model():
    return pipeline("text-generation", model="gpt2")

# Object detection function
def detect_objects(image_path, model):
    results = model(image_path)
    detections = results.pandas().xyxy[0]

    objects = []
    for _, row in detections.iterrows():
        objects.append({
            'name': row['name'],
            'confidence': round(row['confidence'], 2)
        })
    return objects

# Text generation function
def generate_text(prompt, object_info, text_pipeline):
    object_list = ", ".join([f"{obj['name']} ({obj['confidence']})" for obj in object_info])
    full_prompt = f"{prompt}\nBased on the image, I detected: {object_list}.\n"

    try:
        output = text_pipeline(full_prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
        return output
    except Exception as e:
        return f"Text generation failed: {e}"

# Streamlit UI
def main():
    st.title("🔍 Vision + LLM Integration App")
    st.write("Upload an image and provide a prompt to generate intelligent text using object detection and a language model.")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    prompt = st.text_input("Enter your prompt")

    if uploaded_image is not None and prompt:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_image.read())
            image_path = temp_file.name

        # Show the image
        st.image(Image.open(image_path), caption="Uploaded Image", use_column_width=True)

        # Load models
        with st.spinner("Loading models..."):
            detection_model = load_detection_model()
            text_pipeline = load_text_model()

        # Detect objects
        with st.spinner("Detecting objects..."):
            objects = detect_objects(image_path, detection_model)

        # Display results
        if objects:
            st.success("Objects Detected:")
            for obj in objects:
                st.write(f"- {obj['name']} (Confidence: {obj['confidence']})")
        else:
            st.warning("No objects detected in the image.")

        # Generate text
        with st.spinner("Generating text..."):
            result = generate_text(prompt, objects, text_pipeline)
            st.subheader("🧠 Generated Response")
            st.write(result)

        # Clean up temp file
        os.unlink(image_path)
    else:
        st.info("Please upload an image and enter a prompt to proceed.")

if __name__ == "__main__":
    main()
