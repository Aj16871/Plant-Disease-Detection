import streamlit as st
import tensorflow as tf
import numpy as np  # prints the current working directory

# Load the model (cache to avoid reloading on every prediction)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("training_model.keras")
# Preprocess the uploaded image for prediction
def preprocess_image(image):
    img = tf.keras.preprocessing.image.load_img(image, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    return np.expand_dims(img_array, axis=0)  # Convert to batch format

# Get prediction from the model
def predict_image(model, image_array):
    predictions = model.predict(image_array)
    return np.argmax(predictions), predictions

# Filter class names based on the plant prefix
def filter_class_names(prefix, class_names):
    filtered_class_names = [name for name in class_names if name.lower().startswith(prefix.lower())]
    return filtered_class_names

# Map prediction index to class name
def get_class_name_from_predictions(predictions, filtered_class_names, class_names):
    # Re-map the predictions to the filtered list
    filtered_indexes = [class_names.index(name) for name in filtered_class_names]
    
    # Find the highest predicted class in the filtered list
    max_prediction_index = np.argmax(predictions[0][filtered_indexes])
    return filtered_class_names[max_prediction_index]

class Precaution:
    def __init__(self):
        self.disease_details = {
           'Apple___Apple_scab': {
                'precaution': "Apply fungicides, remove infected leaves, and maintain good orchard hygiene to prevent further spread.\n",
                'cause': "Caused by the fungus *Venturia inaequalis*, which thrives in wet, cool conditions.\n",
                'symptoms': "Olive-green to black velvety spots on leaves, fruits, and young twigs.\n",
                'management': "Use resistant varieties, apply fungicides during the growing season, and remove fallen leaves.\n"
            },
            'Apple___Black_rot': {
                'precaution': "Remove infected leaves, improve air circulation, and apply bactericides to prevent further spread.\n",
                'cause': "Caused by the fungus *Botryosphaeria obtusa*, often entering through wounds or cracks in the bark.\n",
                'symptoms': "Dark, sunken lesions on fruit, leaves, and bark, with fruit eventually rotting.\n",
                'management': "Prune out infected branches, remove mummified fruits, and apply appropriate fungicides.\n"
            },
            'Apple___Cedar_apple_rust': {
                'precaution': "Apply fungicides, remove infected leaves, and maintain good orchard hygiene to prevent further spread.\n",
                'cause': "Caused by the fungus *Gymnosporangium juniperi-virginianae*, which requires both apple and cedar trees to complete its lifecycle.\n",
                'symptoms': "Yellow-orange spots on leaves, which later develop black, cup-shaped structures.\n",
                'management': "Remove nearby cedar trees or galls, apply fungicides, and plant resistant apple varieties.\n"
            },
            'Apple___healthy': {
                'precaution': "No precautions needed, the plant is healthy.\n",
                'cause': "N/A",
                'symptoms': "N/A\n",
                'management': "Continue regular care to maintain plant health.\n"
            },
            'Blueberry___healthy': {
                'precaution': "No precautions needed, the plant is healthy.\n",
                'cause': "N/A\n",
                'symptoms': "N/A\n",
                'management': "Continue regular care to maintain plant health.\n"
            },
            'Cherry_(including_sour)___Powdery_mildew': {
                'precaution': "Apply fungicides, remove infected leaves, and maintain good orchard hygiene to prevent further spread.\n",
                'cause': "Caused by the fungus *Podosphaera clandestina*, which thrives in warm, dry conditions.\n",
                'symptoms': "White, powdery fungal growth on leaves, shoots, and fruits.\n",
                'management': "Prune for better air circulation, apply fungicides, and remove and destroy infected plant parts.\n"
            },
            'Cherry_(including_sour)___healthy': {
                'precaution': "No precautions needed, the plant is healthy.\n",
                'cause': "N/A\n",
                'symptoms': "N/A\n",
                'management': "Continue regular care to maintain plant health.\n"
            },
            'Corn_(maize)___Cercospora_leaf_spot (Gray_leaf_spot)': {
                'precaution': "Apply fungicides, remove infected leaves, and maintain good orchard hygiene to prevent further spread.\n",
                'cause': "Caused by the fungus *Cercospora zeae-maydis*, which thrives in warm, humid environments.\n",
                'symptoms': "Small, rectangular lesions that eventually turn gray, leading to reduced photosynthesis.\n",
                'management': "Rotate crops, till soil to bury crop residues, and apply fungicides if necessary.\n"
            },
            'Corn_(maize)___Common_rust': {
                'precaution': "Apply fungicides, remove infected leaves, and maintain good orchard hygiene to prevent further spread.\n",
                'cause': "Caused by the fungus *Puccinia sorghi*, spread by wind-blown spores.\n",
                'symptoms': "Reddish-brown pustules on both leaf surfaces, leading to leaf blighting.\n",
                'management': "Use resistant corn varieties, apply fungicides if necessary, and practice crop rotation.\n"
            },
            'Corn_(maize)___Northern_Leaf_Blight': {
                'precaution': "Apply fungicides, remove infected leaves, and maintain good orchard hygiene to prevent further spread.\n",
                'cause': "Caused by the fungus *Exserohilum turcicum*, which thrives in humid conditions.\n",
                'symptoms': "Large, cigar-shaped lesions on leaves, leading to premature leaf death.\n",
                'management': "Use resistant hybrids, rotate crops, and apply fungicides during severe outbreaks.\n"
            },
            'Corn_(maize)___healthy': {
                'precaution': "No precautions needed, the plant is healthy.\n",
                'cause': "N/A\n",
                'symptoms': "N/A\n",
                'management': "Continue regular care to maintain plant health.\n"
            },
            'Grape___Black_rot': {
                'precaution': "Apply fungicides, remove infected leaves, and maintain good orchard hygiene to prevent further spread.\n",
                'cause': "Caused by the fungus *Guignardia bidwellii*, which overwinters in mummified berries and infected canes.\n",
                'symptoms': "Brownish spots on leaves, black lesions on berries, and shriveling of fruit.\n",
                'management': "Remove and destroy infected plant material, prune for good air circulation, and apply fungicides regularly.\n"
            },
            'Grape___Esca_(Black_Measles)': {
                'precaution': "Apply fungicides, remove infected leaves, and maintain good orchard hygiene to prevent further spread.\n",
                'cause': "Caused by a complex of fungi, including *Phaeomoniella chlamydospora* and *Phaeoacremonium aleophilum*.\n",
                'symptoms': "Dark streaks in wood, leaf discoloration, and black spots on berries.\n",
                'management': "Prune out infected wood, avoid excessive irrigation, and apply fungicides to reduce infection.\n"
            },
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
                'precaution': "Apply fungicides, remove infected leaves, and maintain good orchard hygiene to prevent further spread.\n",
                'cause': "Caused by the fungus *Isariopsis clavispora*, which thrives in warm, wet conditions.\n",
                'symptoms': "Irregular, dark brown spots on leaves that can lead to defoliation.\n",
                'management': "Apply fungicides, remove and destroy affected leaves, and maintain good air circulation around vines.\n"
            },
            'Grape___healthy': {
                'precaution': "No precautions needed, the plant is healthy.\n",
                'cause': "N/A\n",
                'symptoms': "N/A\n",
                'management': "Continue regular care to maintain plant health.\n"
            },
            'Orange___Haunglongbing_(Citrus_greening)': {
                'precaution': "Apply insecticides, remove infected leaves, and maintain good orchard hygiene to prevent further spread.\n",
                'cause': "Caused by the bacterium *Candidatus Liberibacter spp.*, spread by the Asian citrus psyllid.\n",
                'symptoms': "Yellowing of leaves, misshapen fruit, and overall decline in tree health.\n",
                'management': "Control psyllid populations with insecticides, remove infected trees, and use certified disease-free planting material.\n"
            },
            'Peach___Bacterial_spot': {
                'precaution': "Remove infected leaves, improve air circulation, and apply bactericides to prevent further spread.\n",
                'cause': "Caused by the bacterium *Xanthomonas campestris pv. pruni*, spread by rain and wind.\n",
                'symptoms': "Small, water-soaked spots on leaves and fruit, leading to defoliation and fruit blemishes\n.",
                'management': "Apply bactericides, prune trees to improve air circulation, and select resistant varieties.\n"
            },
            'Peach___healthy': {
                'precaution': "No precautions needed, the plant is healthy.\n",
                'cause': "N/A\n",
                'symptoms': "N/A\n",
                'management': "Continue regular care to maintain plant health.\n"
            },
            'Pepper,_bell___Bacterial_spot': {
                'precaution': "Remove infected leaves, improve air circulation, and apply bactericides to prevent further spread.\n",
                'cause': "Caused by several species of the bacterium *Xanthomonas*, spread by splashing water and contaminated tools.\n",
                'symptoms': "Small, dark, water-soaked spots on leaves, stems, and fruit, often leading to defoliation.\n",
                'management': "Practice crop rotation, avoid overhead irrigation, and use copper-based bactericides.\n"
            },
            'Pepper,_bell___healthy': {
                'precaution': "No precautions needed, the plant is healthy.\n",
                'cause': "N/A\n",
                'symptoms': "N/A\n",
                'management': "Continue regular care to maintain plant health.\n"
            },
            'Potato___Early_blight': {
                'precaution': "Apply fungicides, remove infected leaves, and maintain good orchard hygiene to prevent further spread.\n",
                'cause': "Caused by the fungus *Alternaria solani*, which thrives in warm, wet conditions.\n",
                'symptoms': "Dark brown spots with concentric rings on leaves, leading to defoliation.\n",
                'management': "Use certified seed potatoes, rotate crops, and apply fungicides as needed.\n"
            },
            'Potato___Late_blight': {
                'precaution': "Apply fungicides, remove infected leaves, and maintain good orchard hygiene to prevent further spread.\n",
                'cause': "Caused by the oomycete *Phytophthora infestans*, which thrives in wet, cool conditions.\n",
                'symptoms': "Dark, water-soaked spots on leaves and stems, often with a pale green border, followed by rapid plant collapse.\n",
                'management': "Use resistant varieties, avoid overhead watering, and destroy infected plants promptly.\n"
            },
            'Potato___healthy': {
                'precaution': "No precautions needed, the plant is healthy.\n",
                'cause': "N/A\n",
                'symptoms': "N/A\n",
                'management': "Continue regular care to maintain plant health.\n"
            },
            'Raspberry___healthy': {
                'precaution': "No precautions needed, the plant is healthy.\n",
                'cause': "N/A\n",
                'symptoms': "N/A\n",
                'management': "Continue regular care to maintain plant health."
            },
            'Soybean___healthy': {
                'precaution': "No precautions needed, the plant is healthy.\n",
                'cause': "N/A\n",
                'symptoms': "N/A\n",
                'management': "Continue regular care to maintain plant health."
            },
            'Squash___Powdery_mildew': {
                'precaution': "Apply fungicides, remove infected leaves, and maintain good orchard hygiene to prevent further spread.\n",
                'cause': "Caused by several species of fungi, including *Podosphaera xanthii* and *Erysiphe cichoracearum*.\n",
                'symptoms': "White, powdery fungal growth on leaves, stems, and fruit.\n",
                'management': "Apply fungicides, improve air circulation by spacing plants properly, and water plants at the base to keep leaves dry."
            },
            'Strawberry___Leaf_scorch': {
                'precaution': "Remove infected leaves, improve air circulation, and apply bactericides to prevent further spread.\n",
                'cause': "Caused by the fungus *Diplocarpon earlianum*, which thrives in wet, warm conditions.\n",
                'symptoms': "Irregular, dark purple spots on leaves, leading to leaf browning and drying.\n",
                'management': "Remove and destroy infected leaves, ensure good air circulation, and apply fungicides as needed."
            },
            'Strawberry___healthy': {
                'precaution': "No precautions needed, the plant is healthy.\n",
                'cause': "N/A\n",
                'symptoms': "N/A\n",
                'management': "Continue regular care to maintain plant health."
            },
            'Tomato___Bacterial_spot': {
                'precaution': "Remove infected leaves, improve air circulation, and apply bactericides to prevent further spread.\n",
                'cause': "Caused by several species of the bacterium *Xanthomonas*, spread by splashing water and contaminated tools.\n",
                'symptoms': "Small, dark, water-soaked spots on leaves, stems, and fruit, often leading to defoliation.\n",
                'management': "Practice crop rotation, avoid overhead irrigation, and use copper-based bactericides.\n"
            },
            'Tomato___Early_blight': {
                'precaution': "Apply fungicides, remove infected leaves, and maintain good orchard hygiene to prevent further spread.\n",
                'cause': "Caused by the fungus *Alternaria solani*, which thrives in warm, wet conditions.\n",
                'symptoms': "Dark brown spots with concentric rings on leaves, leading to defoliation.\n",
                'management': "Use certified seeds, rotate crops, and apply fungicides during wet weather.\n"
            },
            'Tomato___Late_blight': {
                'precaution': "Apply fungicides, remove infected leaves, and maintain good orchard hygiene to prevent further spread.\n",
                'cause': "Caused by the oomycete *Phytophthora infestans*, which thrives in wet, cool conditions.\n",
                'symptoms': "Dark, water-soaked spots on leaves and stems, often with a pale green border, followed by rapid plant collapse.\n",
                'management': "Use resistant varieties, avoid overhead watering, and destroy infected plants promptly.\n"
            },
            'Tomato___Leaf_Mold': {
                'precaution': "Improve air circulation, remove infected leaves, and apply fungicides to prevent further spread.\n",
                'cause': "Caused by the fungus *Passalora fulva*, which thrives in humid conditions.\n",
                'symptoms': "Yellow spots on the upper leaf surface, with olive-green to gray mold on the underside.\n",
                'management': "Reduce humidity, improve air circulation, and apply fungicides if necessary.\n"
            },
            'Tomato___Septoria_leaf_spot': {
                'precaution': "Apply fungicides, remove infected leaves, and maintain good orchard hygiene to prevent further spread.\n",
                'cause': "Caused by the fungus *Septoria lycopersici*, which thrives in warm, wet conditions.\n",
                'symptoms': "Small, water-soaked spots on leaves that turn into circular lesions with dark borders.\n",
                'management': "Remove and destroy infected leaves, apply fungicides, and practice crop rotation.\n"
            },
            'Tomato___Spider_mites Two-spotted_spider_mite': {
                'precaution': "Apply miticides and improve air circulation to prevent further spread.\n",
                'cause': "Caused by the spider mite *Tetranychus urticae*, which feeds on plant cells.\n",
                'symptoms': "Yellowing of leaves, with fine webbing on the undersides.\n",
                'management': "Use miticides, increase humidity, and release natural predators like ladybugs or predatory mites.\n"
            },
            'Tomato___Target_Spot': {
                'precaution': "Apply fungicides, remove infected leaves, and maintain good orchard hygiene to prevent further spread.\n",
                'cause': "Caused by the fungus *Corynespora cassiicola*, which thrives in warm, humid conditions.\n",
                'symptoms': "Small, dark lesions on leaves and fruit, often leading to defoliation and fruit rot.\n",
                'management': "Apply fungicides, remove infected leaves, and improve air circulation around plants.\n"
            },
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
                'precaution': "Control whitefly population and remove infected plants to prevent further spread.\n",
                'cause': "Caused by the *Tomato yellow leaf curl virus (TYLCV)*, transmitted by the whitefly *Bemisia tabaci*.\n",
                'symptoms': "Yellowing and upward curling of leaves, stunted growth, and reduced fruit set.\n",
                'management': "Control whitefly populations with insecticides, remove infected plants, and use resistant varieties.\n"
            },
            'Tomato___Tomato_mosaic_virus': {
                'precaution': "Remove infected plants and sanitize tools to prevent further spread.\n",
                'cause': "Caused by the *Tomato mosaic virus (ToMV)*, spread by contact with infected plants or contaminated tools.\n",
                'symptoms': "Mottling and mosaic-like patterns on leaves, reduced fruit quality, and stunted growth.\n",
                'management': "Remove and destroy infected plants, disinfect tools, and avoid planting susceptible varieties.\n"
            },
            'Tomato___healthy': {
                'precaution': "No precautions needed, the plant is healthy.\n",
                'cause': "N/A\n",
                'symptoms': "N/A\n",
                'management': "Continue regular care to maintain plant health.\n"
            }
        }

    def get_precaution(self, class_name):
        disease_info = self.disease_details.get(class_name)
        if disease_info:
            return (
                f"**Precaution:** {disease_info['precaution']}\n"
                f"**Cause:** {disease_info['cause']}\n"
                f"**Symptoms:** {disease_info['symptoms']}\n"
                f"**Management:** {disease_info['management']}\n"
            )
        else:
            return "No specific precautions available for this disease."

# Display the prediction and precautions
def display_prediction(class_name):
    precaution = Precaution()
    st.success(f"🌿 Model predicts: **{class_name}**")
    precautions = precaution.get_precaution(class_name)
    st.warning(f"{precautions}")

# Sidebar for navigation
st.sidebar.title("🌱 Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Webcam Live Feed", "Disease Library"])

# Apply custom CSS for enhanced styling
st.markdown("""
    <style>
               /* Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
            color: #333;
        }

        /* Sidebar styling */
        .css-18e3th9 {
            background-color: #2e7d32 !important;
            color: white;
        }

        /* Header and text styles */
        h1, h3, h4, p, li, .stButton > button {
            font-family: 'Poppins', sans-serif;
        }

        /* Header custom styling */
        h1 {
            font-size: 3rem;
            color: #ffffff;
            text-align: center;
            background: linear-gradient(to right, #0072ff, #00c6ff);
            padding: 10px 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }

        /* Card layout */
        .card {
            background: rgba(255, 255, 255, 0.85);
            color: #333;
            border-radius: 10px;
            box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
            transition: 0.3s;
            padding: 20px;
            margin-bottom: 20px;
        }
        .card:hover {
            box-shadow: 0 8px 16px 0 rgba(0, 0, 0, 0.2);
        }

        /* Button hover effect */
        .stButton > button {
            background-color: #0072ff;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            transition: background-color 0.3s, transform 0.3s;
        }

        .stButton > button:hover {
            background-color: #0056b3;
            transform: scale(1.05);
        }

        /* Info and warning boxes styling */
        .st-info-box {
            background-color: #e1f5fe !important;
            color: #333;
        }

        .st-warning-box {
            background-color: #ffebee !important;  /* Lighter background */
            color: #b71c1c !important;  /* Darker text color */
        }

        /* Center align text in the main section */
        .css-1d391kg {
            text-align: center;
        }

        /* List styles */
        ul {
            list-style-type: none;
            padding-left: 0;
        }
        ul li {
            margin-bottom: 10px;
        }
        ul li:before {
            content: '✔️';
            margin-right: 10px;
            color: #00c6ff;
        }

        /* Tooltip styling */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
        }

        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #333;
            color: #fff;
            text-align: center;
            border-radius: 5px;
            padding: 5px 0;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }

        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
    </style>
""", unsafe_allow_html=True)

# Main content based on selection
if app_mode == "Home":
    st.markdown("<h1>PLANT DISEASE RECOGNITION SYSTEM</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div class="card">
            <h3>Welcome to the Plant Disease Recognition System! 🌿🔍</h3>
            <p>Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases.</p>
            <h4>How It Works</h4>
            <ul>
                <li><strong>Upload Image:</strong> Go to the <strong>Disease Recognition</strong> page and upload an image of a plant with suspected diseases.</li>
                <li><strong>Analysis:</strong> Our system will process the image using advanced algorithms to identify potential diseases.</li>
                <li><strong>Results:</strong> View the results and recommendations for further action.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

elif app_mode == "About":
    st.markdown("<h1>About</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div class="card">
            <h4>About Dataset</h4>
            <p>This dataset consists of approximately 87K RGB images of healthy and diseased crop leaves, categorized into 38 different classes. The dataset is divided into training, validation, and test sets, preserving the directory structure.</p>
            <ul>
                <li><strong>Training Set:</strong> 70,295 images</li>
                <li><strong>Validation Set:</strong> 17,572 images</li>
                <li><strong>Test Set:</strong> 33 images</li>
            </ul>
            <h4>Graph Representation</h4>
            <p>Below is a representation of the training and validation accuracy over time.</p>
        </div>
    """, unsafe_allow_html=True)
    st.image("download (1).png", use_column_width=True)

elif app_mode == "Disease Recognition":
    st.markdown("<h1>Disease Recognition</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-style: italic;'>Upload an image below to analyze plant disease.</p>", unsafe_allow_html=True)
    
    # User inputs the plant name
    plant_prefix = st.text_input("Enter the first few letters of the plant name:")
    
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image:
        st.image(test_image, caption="Uploaded Image", use_column_width=True, clamp=True)

        if st.button("Predict"):
            with st.spinner('Analyzing the image...'):
                model = load_model()
                image_array = preprocess_image(test_image)
                result_index, predictions = predict_image(model, image_array)

                # Full list of class names
                class_names = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot (Gray_leaf_spot)',
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy'
                ]
                
                filtered_class_names = filter_class_names(plant_prefix, class_names)
                
                if filtered_class_names:
                    class_name = get_class_name_from_predictions(predictions, filtered_class_names, class_names)
                    display_prediction(class_name)
                else:
                    st.warning("No classes match the entered plant prefix.")
    else:
        st.warning("⚠️ Please upload an image to analyze.")

elif app_mode == "Webcam Live Feed":
    st.markdown("<h1>Webcam Live Feed</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-style: italic;'>Take a photo using your webcam to analyze plant disease.</p>", unsafe_allow_html=True)
    
    # User inputs the plant name
    plant_prefix = st.text_input("Enter the first few letters of the plant name:")
    
    run = st.checkbox("Run")
    
    if run:
        img_file_buffer = st.camera_input("Take a photo")
        
        if img_file_buffer:
            st.image(img_file_buffer)
            
            if st.button("Predict"):
                with st.spinner('Analyzing the image...'):
                    model = load_model()
                    image_array = preprocess_image(img_file_buffer)
                    result_index, predictions = predict_image(model, image_array)

                    # Full list of class names
                    class_names = [
                        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot (Gray_leaf_spot)',
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy'
                    ]
                    
                    filtered_class_names = filter_class_names(plant_prefix, class_names)
                    
                    if filtered_class_names:
                        class_name = get_class_name_from_predictions(predictions, filtered_class_names, class_names)
                        display_prediction(class_name)
                    else:
                        st.warning("No classes match the entered plant prefix.")

elif app_mode == "Disease Library":
    st.markdown("<h1>Disease Library</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div class="card">
            <h4>Browse through a comprehensive library of plant diseases</h4>
            <p>Select a disease from the list to learn more about its symptoms, causes, and management practices.</p>
        </div>
    """, unsafe_allow_html=True)

    disease_list = list(Precaution().disease_details.keys())
    disease_selection = st.selectbox("Choose a disease", disease_list)
    
    precaution = Precaution()
    st.info(precaution.get_precaution(disease_selection))
