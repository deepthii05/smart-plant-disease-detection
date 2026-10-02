import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Smart Plant Disease Detection",
    page_icon="🌿"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌿 Smart Plant Disease Detection")

st.write(
    "Upload a plant leaf image to detect the possible disease "
    "and get basic treatment and prevention information."
)

st.info(
    "For best results, upload a clear image showing the plant leaf."
)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "best_plant_disease_model.keras"

MODEL_FILE_ID = "1BL3en0vR5VZ0RfI8OSBeBGFuGcDYzvRJ"

if not os.path.exists(MODEL_PATH):
    import gdown
    gdown.download(
        f"https://drive.google.com/uc?id={MODEL_FILE_ID}",
        MODEL_PATH,
        quiet=False
    )


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# ============================================================
# CLASS NAMES
# ============================================================

class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# ============================================================
# TREATMENT INFORMATION
# ============================================================

treatment_info = {

    "Apple___Apple_scab": {
        "disease": "Apple Scab",
        "description": "A fungal disease that causes dark olive or brown spots on apple leaves and fruits.",
        "treatment": "Remove and destroy infected leaves and fruits. Improve air circulation and avoid overhead watering. Use an appropriate fungicide according to local agricultural recommendations.",
        "prevention": "Keep the area clean, remove fallen infected leaves, prune overcrowded branches and use disease-resistant varieties when available."
    },

    "Apple___Black_rot": {
        "disease": "Apple Black Rot",
        "description": "A fungal disease that can cause dark lesions on leaves and fruit rot.",
        "treatment": "Remove infected plant material and dead wood. Maintain good airflow and apply an appropriate fungicide when recommended.",
        "prevention": "Remove mummified fruits and dead branches and keep the orchard clean."
    },

    "Apple___Cedar_apple_rust": {
        "disease": "Apple Cedar Apple Rust",
        "description": "A fungal disease that produces yellow-orange spots on apple leaves.",
        "treatment": "Remove heavily infected material and use an appropriate fungicide according to local recommendations.",
        "prevention": "Maintain good airflow and manage nearby alternate hosts when practical."
    },

    "Apple___healthy": {
        "disease": "Apple Healthy",
        "description": "The leaf appears healthy with no major disease detected.",
        "treatment": "No disease treatment is required.",
        "prevention": "Continue proper watering, nutrition, pruning and regular monitoring."
    },

    "Blueberry___healthy": {
        "disease": "Blueberry Healthy",
        "description": "The blueberry leaf appears healthy.",
        "treatment": "No disease treatment is required.",
        "prevention": "Maintain proper soil moisture, nutrition and regular monitoring."
    },

    "Cherry_(including_sour)___Powdery_mildew": {
        "disease": "Cherry Powdery Mildew",
        "description": "A fungal disease that produces a white powder-like growth on leaves and shoots.",
        "treatment": "Remove severely affected parts and improve air circulation. Use an appropriate fungicide according to local recommendations.",
        "prevention": "Avoid excessive nitrogen, maintain airflow and monitor new growth regularly."
    },

    "Cherry_(including_sour)___healthy": {
        "disease": "Cherry Healthy",
        "description": "The cherry leaf appears healthy.",
        "treatment": "No disease treatment is required.",
        "prevention": "Continue regular watering, nutrition and disease monitoring."
    },

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "disease": "Corn Cercospora Leaf Spot / Gray Leaf Spot",
        "description": "A fungal disease that causes gray or brown lesions on maize leaves.",
        "treatment": "Remove heavily infected plant debris where practical and use appropriate disease management practices or fungicides according to local recommendations.",
        "prevention": "Use crop rotation, resistant varieties and good field sanitation."
    },

    "Corn_(maize)___Common_rust_": {
        "disease": "Corn Common Rust",
        "description": "A fungal disease that produces rust-colored spots on maize leaves.",
        "treatment": "Monitor disease development and use resistant varieties or an appropriate fungicide when necessary.",
        "prevention": "Use resistant varieties and maintain proper crop management."
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
        "disease": "Corn Northern Leaf Blight",
        "description": "A fungal disease that produces long gray-green or brown lesions on maize leaves.",
        "treatment": "Use resistant varieties and appropriate fungicide management when recommended.",
        "prevention": "Practice crop rotation, remove infected residue where practical and use resistant varieties."
    },

    "Corn_(maize)___healthy": {
        "disease": "Corn Healthy",
        "description": "The maize leaf appears healthy.",
        "treatment": "No disease treatment is required.",
        "prevention": "Continue proper irrigation, nutrition and regular monitoring."
    },

    "Grape___Black_rot": {
        "disease": "Grape Black Rot",
        "description": "A fungal disease that causes dark lesions and can damage grape berries.",
        "treatment": "Remove infected leaves and fruit. Improve canopy airflow and use an appropriate fungicide according to local recommendations.",
        "prevention": "Remove infected plant material, maintain good canopy management and avoid prolonged leaf wetness."
    },

    "Grape___Esca_(Black_Measles)": {
        "disease": "Grape Esca / Black Measles",
        "description": "A complex grape disease that can cause leaf symptoms and fruit damage.",
        "treatment": "Remove severely affected plant material and manage infected vines according to local viticulture recommendations.",
        "prevention": "Use healthy planting material, maintain vine health and prevent wounds where possible."
    },

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "disease": "Grape Leaf Blight (Isariopsis Leaf Spot)",
        "description": "A fungal leaf disease that causes brown or dark lesions and can lead to leaf damage.",
        "treatment": "Remove severely infected leaves and improve air circulation around the vines. Avoid prolonged leaf wetness and use an appropriate fungicide according to local agricultural recommendations.",
        "prevention": "Maintain good canopy ventilation, remove infected plant debris and monitor leaves regularly."
    },

    "Grape___healthy": {
        "disease": "Grape Healthy",
        "description": "The grape leaf appears healthy.",
        "treatment": "No disease treatment is required.",
        "prevention": "Maintain proper irrigation, nutrition, pruning and regular disease monitoring."
    },

    "Orange___Haunglongbing_(Citrus_greening)": {
        "disease": "Citrus Huanglongbing (Citrus Greening)",
        "description": "A serious citrus disease that can cause yellowing and reduced plant productivity.",
        "treatment": "There is no simple curative treatment for infected trees. Follow local agricultural guidance for managing infected trees and controlling the insect vector.",
        "prevention": "Use certified healthy planting material and manage the insect vector according to local recommendations."
    },

    "Peach___Bacterial_spot": {
        "disease": "Peach Bacterial Spot",
        "description": "A bacterial disease that causes spots and lesions on peach leaves and fruit.",
        "treatment": "Remove severely affected material and use appropriate bacterial disease management practices according to local recommendations.",
        "prevention": "Use resistant varieties where available and maintain good orchard sanitation."
    },

    "Peach___healthy": {
        "disease": "Peach Healthy",
        "description": "The peach leaf appears healthy.",
        "treatment": "No disease treatment is required.",
        "prevention": "Maintain proper watering, nutrition and regular monitoring."
    },

    "Pepper,_bell___Bacterial_spot": {
        "disease": "Pepper Bell Bacterial Spot",
        "description": "A bacterial disease that causes spots on pepper leaves and fruit.",
        "treatment": "Remove severely infected material and avoid overhead irrigation. Follow local recommendations for bacterial disease control.",
        "prevention": "Use clean planting material, maintain sanitation and avoid working with wet plants."
    },

    "Pepper,_bell___healthy": {
        "disease": "Pepper Bell Healthy",
        "description": "The pepper leaf appears healthy.",
        "treatment": "No disease treatment is required.",
        "prevention": "Maintain good irrigation, nutrition and regular monitoring."
    },

    "Potato___Early_blight": {
        "disease": "Potato Early Blight",
        "description": "A fungal disease that causes dark circular lesions on potato leaves.",
        "treatment": "Remove infected plant debris and improve plant airflow. Use an appropriate fungicide according to local agricultural recommendations.",
        "prevention": "Practice crop rotation, maintain plant nutrition and avoid prolonged leaf wetness."
    },

    "Potato___Late_blight": {
        "disease": "Potato Late Blight",
        "description": "A destructive disease that causes dark water-soaked lesions on leaves.",
        "treatment": "Remove severely infected plant material and seek timely local agricultural guidance. Appropriate fungicide management may be required.",
        "prevention": "Use disease-free planting material, resistant varieties where available and avoid prolonged leaf wetness."
    },

    "Potato___healthy": {
        "disease": "Potato Healthy",
        "description": "The potato leaf appears healthy.",
        "treatment": "No disease treatment is required.",
        "prevention": "Maintain proper irrigation, nutrition and regular monitoring."
    },

    "Raspberry___healthy": {
        "disease": "Raspberry Healthy",
        "description": "The raspberry leaf appears healthy.",
        "treatment": "No disease treatment is required.",
        "prevention": "Maintain proper watering, nutrition, pruning and monitoring."
    },

    "Soybean___healthy": {
        "disease": "Soybean Healthy",
        "description": "The soybean leaf appears healthy.",
        "treatment": "No disease treatment is required.",
        "prevention": "Maintain proper irrigation, nutrition and regular crop monitoring."
    },

    "Squash___Powdery_mildew": {
        "disease": "Squash Powdery Mildew",
        "description": "A fungal disease that produces white powder-like growth on leaves.",
        "treatment": "Remove severely affected leaves and improve air circulation. Use an appropriate fungicide according to local recommendations.",
        "prevention": "Maintain good spacing and airflow and avoid excessive humidity around leaves."
    },

    "Strawberry___Leaf_scorch": {
        "disease": "Strawberry Leaf Scorch",
        "description": "A disease that produces reddish or brown lesions and scorched areas on strawberry leaves.",
        "treatment": "Remove severely affected leaves and maintain good plant sanitation. Follow local agricultural recommendations for disease management.",
        "prevention": "Remove infected debris, maintain airflow and avoid prolonged leaf wetness."
    },

    "Strawberry___healthy": {
        "disease": "Strawberry Healthy",
        "description": "The strawberry leaf appears healthy.",
        "treatment": "No disease treatment is required.",
        "prevention": "Maintain proper watering, nutrition and regular monitoring."
    },

    "Tomato___Bacterial_spot": {
        "disease": "Tomato Bacterial Spot",
        "description": "A bacterial disease that causes dark spots on tomato leaves and fruit.",
        "treatment": "Remove severely infected material and avoid overhead watering. Follow local recommendations for bacterial disease management.",
        "prevention": "Use clean planting material, maintain sanitation and avoid handling plants when they are wet."
    },

    "Tomato___Early_blight": {
        "disease": "Tomato Early Blight",
        "description": "A fungal disease that produces dark concentric lesions on tomato leaves.",
        "treatment": "Remove infected leaves and plant debris. Improve airflow and use an appropriate fungicide according to local recommendations.",
        "prevention": "Practice crop rotation, maintain good spacing and avoid prolonged leaf wetness."
    },

    "Tomato___Late_blight": {
        "disease": "Tomato Late Blight",
        "description": "A serious disease that produces dark water-soaked lesions on tomato leaves.",
        "treatment": "Remove severely infected plant material and seek timely local agricultural guidance. Appropriate fungicide management may be required.",
        "prevention": "Use disease-free plants, improve airflow and avoid prolonged leaf wetness."
    },

    "Tomato___Leaf_Mold": {
        "disease": "Tomato Leaf Mold",
        "description": "A fungal disease that commonly develops under humid conditions.",
        "treatment": "Remove affected leaves and improve ventilation. Reduce humidity around foliage and use an appropriate fungicide when recommended.",
        "prevention": "Improve greenhouse ventilation and avoid excessive leaf moisture."
    },

    "Tomato___Septoria_leaf_spot": {
        "disease": "Tomato Septoria Leaf Spot",
        "description": "A fungal disease that causes many small spots on tomato leaves.",
        "treatment": "Remove affected leaves and plant debris. Improve airflow and use an appropriate fungicide according to local recommendations.",
        "prevention": "Practice crop rotation, maintain sanitation and avoid overhead watering."
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "disease": "Tomato Two-Spotted Spider Mite",
        "description": "A pest infestation that can cause stippling, yellowing and leaf damage.",
        "treatment": "Inspect the underside of leaves and use appropriate integrated pest management practices. Follow local recommendations for mite control.",
        "prevention": "Monitor plants regularly, reduce plant stress and encourage beneficial organisms where appropriate."
    },

    "Tomato___Target_Spot": {
        "disease": "Tomato Target Spot",
        "description": "A fungal disease that causes circular target-like lesions on leaves and fruit.",
        "treatment": "Remove infected leaves and improve airflow. Use an appropriate fungicide according to local recommendations.",
        "prevention": "Maintain plant spacing, sanitation and avoid prolonged leaf wetness."
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "disease": "Tomato Yellow Leaf Curl Virus",
        "description": "A viral disease that can cause leaf curling, yellowing and reduced plant growth.",
        "treatment": "There is no direct cure for an infected plant. Remove severely infected plants where appropriate and manage the insect vector according to local recommendations.",
        "prevention": "Use healthy seedlings and control the insect vector using integrated pest management."
    },

    "Tomato___Tomato_mosaic_virus": {
        "disease": "Tomato Mosaic Virus",
        "description": "A viral disease that can cause mosaic patterns and reduced plant growth.",
        "treatment": "Remove severely infected plants and maintain strict sanitation. There is no direct cure for an infected plant.",
        "prevention": "Use clean planting material, disinfect tools and avoid spreading plant sap between plants."
    },

    "Tomato___healthy": {
        "disease": "Tomato Healthy",
        "description": "The tomato leaf appears healthy.",
        "treatment": "No disease treatment is required.",
        "prevention": "Maintain proper irrigation, nutrition, airflow and regular monitoring."
    }
}


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("Upload a Plant Leaf")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf",
        width=400
    )

    if st.button("Detect Disease"):

        with st.spinner("Analyzing image..."):

            # Same preprocessing used during training
            img_array = np.array(
                image,
                dtype=np.float32
            )

            img_array = tf.image.resize(
                img_array,
                [224, 224]
            ).numpy()

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            predictions = model.predict(
                img_array,
                verbose=0
            )

            predicted_index = np.argmax(
                predictions[0]
            )

            confidence = (
                float(predictions[0][predicted_index]) * 100
            )

        # ----------------------------------------------------
        # CONFIDENCE CHECK
        # ----------------------------------------------------

        if confidence < 70:

            st.warning(
                "The model is not confident about this image."
            )

            st.write(
                f"Confidence: {confidence:.2f}%"
            )

            st.info(
                "Please upload a clear image of a plant leaf."
            )

        else:

            disease = class_names[predicted_index]

            readable_name = (
                disease
                .replace("___", " - ")
                .replace("_", " ")
            )

            info = treatment_info.get(disease)

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.success("Prediction completed!")

            st.subheader("Prediction Result")

            st.write(f"**Disease:** {readable_name}")
            st.write(f"**Confidence:** {confidence:.2f}%")

            if info:

                st.subheader("Disease Information")

                st.write("**Description**")
                st.write(info["description"])

                st.write("**Recommended Treatment**")
                st.write(info["treatment"])

                st.write("**Prevention**")
                st.write(info["prevention"])


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "This system is developed for educational purposes."
)

st.caption(
    "Model trained using the PlantVillage dataset."
)