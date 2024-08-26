import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests

# Define translations
translations = {
    "English": {
        "title": "CropPro Assist",
        "welcome": "Welcome to the Plant Disease Recognition System! 🌿🔍",
        "mission": "Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases.",
        "how_it_works": "How It Works",
        "upload_image": "Upload Image:",
        "disease_recognition": "Go to the Disease Recognition page and upload an image of a plant with suspected diseases.",
        "analysis": "Analysis:",
        "results": "Results:",
        "about": "About",
        "about_dataset": "About Dataset",
        "dataset_description": "This dataset consists of approximately 87K RGB images of healthy and diseased crop leaves, categorized into 38 different classes.",
        "training_set": "Training Set:",
        "validation_set": "Validation Set:",
        "test_set": "Test Set:",
        "graph_representation": "Graph Representation",
        "disease_recognition": "Disease Recognition",
        "upload_prompt": "Upload an image below to analyze plant disease.",
        "enter_plant_name": "Enter the first few letters of the plant name:",
        "choose_image": "Choose an Image:",
        "predict": "Predict",
        "analyzing_image": "Analyzing the image...",
        "no_classes_match": "No classes match the entered plant prefix.",
        "please_upload_image": "⚠️ Please upload an image to analyze.",
        "webcam_live_feed": "Webcam Live Feed",
        "take_photo": "Take a photo using your webcam to analyze plant disease.",
        "run": "Run",
        "disease_library": "Disease Library",
        "browse_library": "Browse through a comprehensive library of plant diseases",
        "select_disease": "Choose a disease",
        "precaution": "Precaution",
        "cause": "Cause",
        "symptoms": "Symptoms",
        "management": "Management",
        "contact_bot": "Contact Bot",
        "fertilizer": "Fertilizer",
        "chat_with_bot": "Chat with our assistant",
        "Agriculture news": "Agriculture news",
        "Weather": "Weather"
    },
    "Hindi": {
        "title": "CropPro Assist",
        "welcome": "प्लांट रोग पहचान प्रणाली में आपका स्वागत है! 🌿🔍",
        "mission": "हमारा मिशन पौधों के रोगों की पहचान में मदद करना है। एक पौधे की छवि अपलोड करें, और हमारी प्रणाली किसी भी रोग के लक्षणों का पता लगाने के लिए इसका विश्लेषण करेगी।",
        "how_it_works": "यह कैसे काम करता है",
        "upload_image": "छवि अपलोड करें:",
        "disease_recognition": "रोग पहचान पृष्ठ पर जाएं और संदिग्ध रोगों वाले पौधे की छवि अपलोड करें।",
        "analysis": "विश्लेषण:",
        "results": "परिणाम:",
        "about": "के बारे में",
        "about_dataset": "डेटासेट के बारे में",
        "dataset_description": "इस डेटासेट में स्वस्थ और रोगग्रस्त फसलों की पत्तियों की लगभग 87K RGB छवियाँ शामिल हैं, जिन्हें 38 विभिन्न वर्गों में वर्गीकृत किया गया है।",
        "training_set": "प्रशिक्षण सेट:",
        "validation_set": "मान्यता सेट:",
        "test_set": "परीक्षण सेट:",
        "graph_representation": "ग्राफ प्रतिनिधित्व",
        "disease_recognition": "रोग पहचान",
        "upload_prompt": "पौधों के रोग का विश्लेषण करने के लिए नीचे एक छवि अपलोड करें।",
        "enter_plant_name": "पौधे के नाम के पहले कुछ अक्षर दर्ज करें:",
        "choose_image": "एक छवि चुनें:",
        "predict": "भविष्यवाणी करें",
        "analyzing_image": "छवि का विश्लेषण कर रहा है...",
        "no_classes_match": "दर्ज किए गए पौधे के उपसर्ग से कोई वर्ग मेल नहीं खाता।",
        "please_upload_image": "⚠️ कृपया विश्लेषण के लिए एक छवि अपलोड करें।",
        "webcam_live_feed": "वेबकैम लाइव फ़ीड",
        "take_photo": "पौधों के रोग का विश्लेषण करने के लिए अपने वेबकैम का उपयोग करके एक फोटो लें।",
        "run": "चलाएं",
        "disease_library": "रोग पुस्तकालय",
        "browse_library": "पौधों के रोगों के व्यापक पुस्तकालय के माध्यम से ब्राउज़ करें",
        "select_disease": "एक रोग चुनें",
        "precaution": "एहतियात",
        "cause": "कारण",
        "symptoms": "लक्षण",
        "management": "प्रबंधन",
        "fertilizer": "उर्वरक",
        "contact_bot": "बॉट से संपर्क करें",
        "chat_with_bot": "हमारे AI सहायक के साथ चैट करें",
        "Agriculture news": "Agriculture news",
        "Weather": "Weather"
    }
}

# Function to get user location
def fetch_weather(api_key, lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url)
    return response.json()

def fetch_air_quality(api_key, lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(url)
    return response.json()

def display_weather(weather_data, air_quality_data):
    if weather_data.get("cod") != 200:
        st.error("City not found or API limit reached.")
        return
    
    city = weather_data["name"]
    country = weather_data["sys"]["country"]
    temp = weather_data["main"]["temp"]
    feels_like = weather_data["main"]["feels_like"]
    humidity = weather_data["main"]["humidity"]
    weather_description = weather_data["weather"][0]["description"].capitalize()
    icon = weather_data["weather"][0]["icon"]

    air_quality_index = air_quality_data["list"][0]["main"]["aqi"]
    air_quality = ["Good", "Fair", "Moderate", "Poor", "Very Poor"][air_quality_index - 1]

    st.markdown(f"""
        <div style="text-align: center; background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%); padding: 20px; border-radius: 10px; color: #ffffff;">
            <h2 style="font-size: 36px;">Weather in {city}, {country}</h2>
            <img src="http://openweathermap.org/img/wn/{icon}.png" style="width: 100px;"/>
            <h3 style="font-size: 28px;">{weather_description}</h3>
            <p style="font-size: 24px;"><strong>Temperature:</strong> {temp}°C</p>
            <p style="font-size: 24px;"><strong>Feels Like:</strong> {feels_like}°C</p>
            <p style="font-size: 24px;"><strong>Humidity:</strong> {humidity}%</p>
            <p style="font-size: 24px;"><strong>Air Quality:</strong> {air_quality}</p>
        </div>
    """, unsafe_allow_html=True)

def weather_page():
    st.markdown("""
        <div style="text-align: center; background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%); padding: 20px; border-radius: 10px;">
            <h1 style="color: #ffffff; font-family: 'Poppins', sans-serif; font-size: 48px;">Weather Information</h1>
        </div>
    """, unsafe_allow_html=True)

    api_key = "e3adae5cd3177f317493c05f71b7062c"  # Your OpenWeather API key

    try:
        # Try to get the user's location using GPS
        lat, lon = st.geolocation
        weather_data = fetch_weather(api_key, lat, lon)
        air_quality_data = fetch_air_quality(api_key, lat, lon)
        display_weather(weather_data, air_quality_data)
    except:
        # If GPS is not available, provide an option to enter the city name manually
        st.write("GPS location not available. Please enter your city name:")
        city_name = st.text_input("Enter city name", "")
        if st.button("Get Weather", key="weather_button"):
            if city_name:
                # Get the latitude and longitude of the city name
                geocoding_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&appid={api_key}"
                geocoding_response = requests.get(geocoding_url)
                geocoding_data = geocoding_response.json()
                if geocoding_data:
                    lat = geocoding_data[0]["lat"]
                    lon = geocoding_data[0]["lon"]
                    weather_data = fetch_weather(api_key, lat, lon)
                    air_quality_data = fetch_air_quality(api_key, lat, lon)
                    display_weather(weather_data, air_quality_data)
                else:
                    st.error("City not found.")
            else:
                st.error("Please enter a city name.")

def fetch_agriculture_news(api_key):
    url = f"https://newsapi.org/v2/everything?q=agriculture+India&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get('articles')
    return articles

# Function to display the fetched news
def display_news(articles):
    for article in articles:
        st.subheader(article['title'])
        st.write(article['description'])
        st.markdown(f"[Read more]({article['url']})")

        if article['urlToImage']:
            st.image(article['urlToImage'], use_column_width=True)
        else:
            st.write("No image available")
        
        st.write("---")

# Function for the news page focused on India
def news_page():
    st.markdown("""
        <div style="text-align: center; background: linear-gradient(135deg, #FF8008 0%, #FFC837 100%); padding: 20px; border-radius: 10px;">
            <h1 style="color: #ffffff; font-family: 'Poppins', sans-serif; font-size: 48px;">Agriculture News</h1>
        </div>
    """, unsafe_allow_html=True)

    # API key for NewsAPI
    api_key = "9d657e5a32c5441d85c4bace7fad400b"

    # Fetch and display agriculture news articles for India
    articles = fetch_agriculture_news(api_key)
    if articles:
        display_news(articles)
    else:
        st.write("Failed to fetch news articles. Please check your API key.")
        

def model_prediction(test_image):
    model = tf.keras.models.load_model("training_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    
    predictions = model.predict(input_arr)
    return predictions

def filter_class_names(prefix, class_names):
    filtered_class_names = [name for name in class_names if name.lower().startswith(prefix.lower())]
    return filtered_class_names

def get_class_name_from_predictions(predictions, filtered_class_names, class_names):
    filtered_indexes = [class_names.index(name) for name in filtered_class_names]
    max_prediction_index = np.argmax(predictions[0][filtered_indexes])
    return filtered_class_names[max_prediction_index]

class Precaution:
    def __init__(self, language="English"):
        self.language = language
        self.disease_details = {
           
    'Apple___Cedar_apple_rust': {
        'precaution': {
            "English": "Apply fungicides, remove infected leaves, and maintain good orchard hygiene to prevent further spread.\n",
            "Hindi": "फफूंदनाशकों का उपयोग करें, संक्रमित पत्तियों को हटा दें और आगे फैलने से रोकने के लिए अच्छी बागवानी स्वच्छता बनाए रखें।\n"
        },
        'cause': {
            "English": "Caused by the fungus *Gymnosporangium juniperi-virginianae*, which requires both apple and cedar trees to complete its lifecycle.\n",
            "Hindi": "फफूंद *जिम्नोस्पोरेंजियम जुनीपेरी-वर्जिनियाना* के कारण होता है, जिसे अपना जीवन चक्र पूरा करने के लिए सेब और देवदार दोनों पेड़ों की आवश्यकता होती है।\n"
        },
        'symptoms': {
            "English": "Yellow-orange spots on leaves, which later develop black, cup-shaped structures.\n",
            "Hindi": "पत्तियों पर पीले-नारंगी धब्बे, जो बाद में काले, कप के आकार की संरचनाएं विकसित करते हैं।\n"
        },
        'management': {
            "English": "Remove nearby cedar trees or galls, apply fungicides, and plant resistant apple varieties.\n",
            "Hindi": "पास के देवदार के पेड़ या गॉल्स हटा दें, फफूंदनाशकों का उपयोग करें और प्रतिरोधी सेब की किस्में लगाएं।\n"
        },
        'fertilizer': {
            "English": "Use a balanced fertilizer like 10-10-10 (NPK) in early spring. Apply 2-4 pounds per tree, depending on the age and size of the tree. Ensure that the fertilizer is spread evenly around the root zone.\n",
            "Hindi": "शुरुआत में 10-10-10 (NPK) जैसे संतुलित उर्वरक का उपयोग करें। पेड़ की उम्र और आकार के अनुसार 2-4 पाउंड प्रति पेड़ लगाएं। सुनिश्चित करें कि उर्वरक को जड़ क्षेत्र के चारों ओर समान रूप से फैलाया जाए।\n"
        }
    },
    'Apple___healthy': {
        'precaution': {
            "English": "No precautions needed, the plant is healthy.\n",
            "Hindi": "कोई सावधानी की आवश्यकता नहीं है, पौधा स्वस्थ है।\n"
        },
        'cause': {
            "English": "N/A\n",
            "Hindi": "एन/ए\n"
        },
        'symptoms': {
            "English": "N/A\n",
            "Hindi": "एन/ए\n"
        },
        'management': {
            "English": "Continue regular care to maintain plant health.\n",
            "Hindi": "पौधे की सेहत बनाए रखने के लिए नियमित देखभाल जारी रखें।\n"
        },
        'fertilizer': {
            "English": "Apply a balanced fertilizer such as 10-10-10 (NPK) at the start of the growing season. For young trees, use about 1 pound per year of tree age, up to a maximum of 10 pounds. Spread evenly in the root zone.\n",
            "Hindi": "उगने के मौसम की शुरुआत में 10-10-10 (NPK) जैसे संतुलित उर्वरक का उपयोग करें। युवा पेड़ों के लिए, पेड़ की उम्र के प्रति वर्ष लगभग 1 पाउंड का उपयोग करें, अधिकतम 10 पाउंड तक। इसे जड़ क्षेत्र में समान रूप से फैलाएं।\n"
        }
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'precaution': {
            "English": "Apply fungicides, remove infected leaves, and maintain good orchard hygiene to prevent further spread.\n",
            "Hindi": "फफूंदनाशकों का उपयोग करें, संक्रमित पत्तियों को हटा दें और आगे फैलने से रोकने के लिए अच्छी बागवानी स्वच्छता बनाए रखें।\n"
        },
        'cause': {
            "English": "Caused by the fungus *Podosphaera clandestina*, which thrives in warm, dry conditions.\n",
            "Hindi": "फफूंद *पोडोस्फेरा क्लैन्डेस्टिना* के कारण होता है, जो गर्म, सूखे वातावरण में पनपता है।\n"
        },
        'symptoms': {
            "English": "White, powdery fungal growth on leaves, shoots, and fruits.\n",
            "Hindi": "पत्तियों, शूट्स और फलों पर सफेद, फफूंदी वृद्धि।\n"
        },
        'management': {
            "English": "Prune for better air circulation, apply fungicides, and remove and destroy infected plant parts.\n",
            "Hindi": "हवा के संचार के लिए प्रूनिंग करें, फफूंदनाशकों का उपयोग करें और संक्रमित पौधे के हिस्सों को हटा और नष्ट करें।\n"
        },
        'fertilizer': {
            "English": "Apply a balanced fertilizer like 10-10-10 (NPK) in spring. For mature trees, apply 1-2 pounds per tree. Ensure even distribution and avoid direct contact with the trunk.\n",
            "Hindi": "वसंत में 10-10-10 (NPK) जैसे संतुलित उर्वरक का उपयोग करें। परिपक्व पेड़ों के लिए, प्रति पेड़ 1-2 पाउंड लगाएं। समान वितरण सुनिश्चित करें और तने के सीधे संपर्क से बचें।\n"
        }
    },
    'Cherry_(including_sour)___healthy': {
        'precaution': {
            "English": "No precautions needed, the plant is healthy.\n",
            "Hindi": "कोई सावधानी की आवश्यकता नहीं है, पौधा स्वस्थ है।\n"
        },
        'cause': {
            "English": "N/A\n",
            "Hindi": "एन/ए\n"
        },
        'symptoms': {
            "English": "N/A\n",
            "Hindi": "एन/ए\n"
        },
        'management': {
            "English": "Continue regular care to maintain plant health.\n",
            "Hindi": "पौधे की सेहत बनाए रखने के लिए नियमित देखभाल जारी रखें।\n"
        },
        'fertilizer': {
            "English": "Use a balanced fertilizer like 10-10-10 (NPK) in early spring. For young trees, apply 0.5 to 1 pound per year of tree age, up to 5 pounds.\n",
            "Hindi": "शुरुआत में 10-10-10 (NPK) जैसे संतुलित उर्वरक का उपयोग करें। युवा पेड़ों के लिए, पेड़ की उम्र के अनुसार 0.5 से 1 पाउंड तक का उपयोग करें, अधिकतम 5 पाउंड तक।\n"
        }
    },
    'Corn_(maize)___Common_rust': {
        'precaution': {
            "English": "Apply fungicides, remove infected leaves, and maintain good orchard hygiene to prevent further spread.\n",
            "Hindi": "फफूंदनाशकों का उपयोग करें, संक्रमित पत्तियों को हटा दें और आगे फैलने से रोकने के लिए अच्छी बागवानी स्वच्छता बनाए रखें।\n"
        },
        'cause': {
            "English": "Caused by the fungus *Puccinia sorghi*, spread by wind-blown spores.\n",
            "Hindi": "फफूंद *पुक्सिनिया सोर्घी* के कारण होता है, जिसका प्रसार हवा से उड़ने वाले स्पोर्स द्वारा होता है।\n"
        },
        'symptoms': {
            "English": "Reddish-brown pustules on both leaf surfaces, leading to leaf blighting.\n",
            "Hindi": "दोनों पत्ती सतहों पर लाल-भूरे रंग के पुस्टुल्स, जिसके कारण पत्ती का मुरझाना होता है।\n"
        },
        'management': {
            "English": "Use resistant corn varieties, apply fungicides if necessary, and practice crop rotation.\n",
            "Hindi": "प्रतिरोधी मक्का की किस्में उपयोग करें, आवश्यकता होने पर फफूंदनाशकों का उपयोग करें और फसल चक्रण का अभ्यास करें।\n"
        },
        'fertilizer': {
            "English": "Use a nitrogen-rich fertilizer, such as 46-0-0 (Urea). Apply 1-2 pounds per 100 square feet at the early growth stage.\n",
            "Hindi": "नाइट्रोजन से भरपूर उर्वरक का उपयोग करें, जैसे 46-0-0 (यूरिया)। शुरुआती विकास चरण में प्रति 100 वर्ग फीट पर 1-2 पाउंड लगाएं।\n"
        }
    },
    'Corn_(maize)___healthy': {
        'precaution': {
            "English": "No precautions needed, the plant is healthy.\n",
            "Hindi": "कोई सावधानी की आवश्यकता नहीं है, पौधा स्वस्थ है।\n"
        },
        'cause': {
            "English": "N/A\n",
            "Hindi": "एन/ए\n"
        },
        'symptoms': {
            "English": "N/A\n",
            "Hindi": "एन/ए\n"
        },
        'management': {
            "English": "Continue regular care to maintain plant health.\n",
            "Hindi": "पौधे की सेहत बनाए रखने के लिए नियमित देखभाल जारी रखें।\n"
        },
        'fertilizer': {
            "English": "Apply a balanced NPK fertilizer (20-20-20) at planting, followed by a side-dressing of nitrogen at the knee-high stage.\n",
            "Hindi": "रोपण के समय संतुलित NPK उर्वरक (20-20-20) लगाएं, इसके बाद घुटने-ऊँचाई के चरण में नाइट्रोजन की साइड-ड्रेसिंग करें।\n"
        }
    },
    'Grape___Esca_(Black_Measles)': {
        'precaution': {
            "English": "Apply fungicides, remove infected leaves, and maintain good orchard hygiene to prevent further spread.\n",
            "Hindi": "फफूंदनाशकों का उपयोग करें, संक्रमित पत्तियों को हटा दें और आगे फैलने से रोकने के लिए अच्छी बागवानी स्वच्छता बनाए रखें।\n"
        },
        'cause': {
            "English": "Caused by a complex of fungi, including *Phaeomoniella chlamydospora* and *Phaeoacremonium aleophilum*.\n",
            "Hindi": "फफूंदों के जटिल के कारण होता है, जिसमें *फियोमोनिएला क्लेमिडोस्पोरा* और *फियोएक्रेमोनियम एलियोफिलम* शामिल हैं।\n"
        },
        'symptoms': {
            "English": "Dark streaks in wood, leaf discoloration, and black spots on berries.\n",
            "Hindi": "लकड़ी में काले धब्बे, पत्ती का रंग बदलना और बेरी पर काले धब्बे।\n"
        },
        'management': {
            "English": "Prune out infected wood, avoid excessive irrigation, and apply fungicides to reduce infection.\n",
            "Hindi": "संक्रमित लकड़ी को हटा दें, अत्यधिक सिंचाई से बचें और संक्रमण को कम करने के लिए फफूंदनाशकों का उपयोग करें।\n"
        },
        'fertilizer': {
            "English": "Use a balanced fertilizer like 10-10-10 (NPK) in spring, and apply 1 pound per year of vine age, up to 6 pounds per vine. Mulch to retain moisture and suppress weeds.\n",
            "Hindi": "वसंत में 10-10-10 (NPK) जैसे संतुलित उर्वरक का उपयोग करें, और बेल की उम्र के अनुसार प्रति वर्ष 1 पाउंड, अधिकतम 6 पाउंड प्रति बेल लगाएं। नमी बनाए रखने और खरपतवार को दबाने के लिए मल्च का उपयोग करें।\n"
        }
    },
    'Grape___healthy': {
        'precaution': {
            "English": "No precautions needed, the plant is healthy.\n",
            "Hindi": "कोई सावधानी की आवश्यकता नहीं है, पौधा स्वस्थ है।\n"
        },
        'cause': {
            "English": "N/A\n",
            "Hindi": "एन/ए\n"
        },
        'symptoms': {
            "English": "N/A\n",
            "Hindi": "एन/ए\n"
        },
        'management': {
            "English": "Continue regular care to maintain plant health.\n",
            "Hindi": "पौधे की सेहत बनाए रखने के लिए नियमित देखभाल जारी रखें।\n"
        },
        'fertilizer': {
            "English": "Apply 1 pound of 10-10-10 (NPK) fertilizer per year of vine age, up to 4 pounds per vine. Fertilize in early spring before new growth begins.\n",
            "Hindi": "बेल की उम्र के अनुसार प्रति वर्ष 1 पाउंड 10-10-10 (NPK) उर्वरक लगाएं, अधिकतम 4 पाउंड प्रति बेल। नई वृद्धि शुरू होने से पहले प्रारंभिक वसंत में उर्वरक डालें।\n"
        }
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'precaution': {
            "English": "Apply insecticides, remove infected leaves, and maintain good orchard hygiene to prevent further spread.\n",
            "Hindi": "कीटनाशकों का उपयोग करें, संक्रमित पत्तियों को हटा दें और आगे फैलने से रोकने के लिए अच्छी बागवानी स्वच्छता बनाए रखें।\n"
        },
        'cause': {
            "English": "Caused by the bacterium *Candidatus Liberibacter spp.*, spread by the Asian citrus psyllid.\n",
            "Hindi": "बैक्टीरिया *कैंडिडेटस लिबेरिबैक्टर स्पीपी.* के कारण होता है, जिसका प्रसार एशियन सिट्रस साइलिड द्वारा होता है।\n"
        },
        'symptoms': {
            "English": "Yellowing of leaves, misshapen fruit, and overall decline in tree health.\n",
            "Hindi": "पत्तियों का पीला होना, फल का विकृत होना और पेड़ की सेहत में समग्र गिरावट।\n"
        },
        'management': {
            "English": "Control psyllid populations with insecticides, remove infected trees, and use certified disease-free planting material.\n",
            "Hindi": "कीटनाशकों के साथ साइलिड आबादी को नियंत्रित करें, संक्रमित पेड़ों को हटा दें और प्रमाणित रोग-मुक्त पौध सामग्री का उपयोग करें।\n"
        },
        'fertilizer': {
            "English": "Apply a citrus-specific fertilizer with micronutrients, such as 6-4-6 or 8-3-9, during the growing season. Apply 1-2 pounds per tree in three equal doses throughout the year.\n",
            "Hindi": "माइक्रोन्यूट्रिएंट्स के साथ साइट्रस-विशिष्ट उर्वरक जैसे 6-4-6 या 8-3-9 का उपयोग करें। बढ़ते मौसम के दौरान प्रति वर्ष तीन बराबर खुराक में 1-2 पाउंड प्रति पेड़ लगाएं।\n"
        }
    },
    'Peach___Bacterial_spot': {
        'precaution': {
            "English": "Remove infected leaves, improve air circulation, and apply bactericides to prevent further spread.\n",
            "Hindi": "संक्रमित पत्तियों को हटा दें, हवा के संचार में सुधार करें और आगे फैलने से रोकने के लिए बैक्टीरियासाइड्स का उपयोग करें।\n"
        },
        'cause': {
            "English": "Caused by the bacterium *Xanthomonas campestris pv. pruni*, spread by rain and wind.\n",
            "Hindi": "बैक्टीरिया *जैन्थोमोनास कैम्पेस्ट्रिस पीवी प्रूनी* के कारण होता है, जिसका प्रसार वर्षा और हवा द्वारा होता है।\n"
        },
        'symptoms': {
            "English": "Small, water-soaked spots on leaves and fruit, leading to defoliation and fruit blemishes.\n",
            "Hindi": "पत्तियों और फल पर छोटे, पानी से भीगे हुए धब्बे, जिसके कारण पत्तियों का झड़ना और फल के दाग पड़ना होता है।\n"
        },
        'management': {
            "English": "Apply bactericides, prune trees to improve air circulation, and select resistant varieties.\n",
            "Hindi": "बैक्टीरियासाइड्स का उपयोग करें, पेड़ों की कटाई करें हवा के संचार में सुधार करें और प्रतिरोधी किस्में चुनें।\n"
        },
        'fertilizer': {
            "English": "Use a balanced fertilizer like 10-10-10 (NPK) in early spring. For mature trees, apply 1-2 pounds per tree. Ensure even distribution and avoid direct contact with the trunk.\n",
            "Hindi": "वसंत में 10-10-10 (NPK) जैसे संतुलित उर्वरक का उपयोग करें। परिपक्व पेड़ों के लिए, प्रति पेड़ 1-2 पाउंड लगाएं। समान वितरण सुनिश्चित करें और तने के सीधे संपर्क से बचें।\n"
        }
    },
    'Peach___healthy': {
        'precaution': {
            "English": "No precautions needed, the plant is healthy.\n",
            "Hindi": "कोई सावधानी की आवश्यकता नहीं है, पौधा स्वस्थ है।\n"
        },
        'cause': {
            "English": "N/A\n",
            "Hindi": "एन/ए\n"
        },
        'symptoms': {
            "English": "N/A\n",
            "Hindi": "एन/ए\n"
        },
        'management': {
            "English": "Continue regular care to maintain plant health.\n",
            "Hindi": "पौधे की सेहत बनाए रखने के लिए नियमित देखभाल जारी रखें।\n"
        },
        'fertilizer': {
            "English": "Apply a balanced fertilizer like 10-10-10 (NPK) in early spring. For young trees, apply 1 pound per year of tree age, up to 10 pounds per tree.\n",
            "Hindi": "वसंत में 10-10-10 (NPK) जैसे संतुलित उर्वरक का उपयोग करें। युवा पेड़ों के लिए, पेड़ की उम्र के अनुसार प्रति वर्ष 1 पाउंड, अधिकतम 10 पाउंड प्रति पेड़ लगाएं।\n"
        }
    },
    'Pepper,_bell___Bacterial_spot': {
        'precaution': {
            "English": "Remove infected leaves, improve air circulation, and apply bactericides to prevent further spread.\n",
            "Hindi": "संक्रमित पत्तियों को हटा दें, हवा के संचार में सुधार करें और आगे फैलने से रोकने के लिए बैक्टीरियासाइड्स का उपयोग करें।\n"
        },
        'cause': {
            "English": "Caused by several species of the bacterium *Xanthomonas*, spread by splashing water and contaminated tools.\n",
            "Hindi": "बैक्टीरिया *जैन्थोमोनास* की कई प्रजातियों के कारण होता है, जिसका प्रसार पानी के छींटे और दूषित औजारों द्वारा होता है।\n"
        },
        'symptoms': {
            "English": "Small, dark, water-soaked spots on leaves, stems, and fruit, often leading to defoliation.\n",
            "Hindi": "पत्तियों, तनों और फल पर छोटे, काले, पानी से भीगे हुए धब्बे, जिसके कारण अक्सर पत्तियों का झड़ना होता है।\n"
        },
        'management': {
            "English": "Practice crop rotation, avoid overhead irrigation, and use copper-based bactericides.\n",
            "Hindi": "फसल चक्रण का अभ्यास करें, ओवरहेड सिंचाई से बचें और ताम्बा-आधारित बैक्टीरियासाइड्स का उपयोग करें।\n"
        },
        'fertilizer': {
            "English": "Apply a balanced fertilizer like 10-10-10 (NPK) at planting, and side-dress with calcium nitrate (15.5-0-0) when fruiting begins. Use 2-3 pounds per 100 square feet.\n",
            "Hindi": "रोपण के समय 10-10-10 (NPK) जैसे संतुलित उर्वरक का उपयोग करें, और फल लगने पर कैल्शियम नाइट्रेट (15.5-0-0) से साइड-ड्रेसिंग करें। प्रति 100 वर्ग फीट 2-3 पाउंड का उपयोग करें।\n"
        }
    },
    'Potato___Early_blight': {
        'precaution': {
            "English": "Apply fungicides, remove infected leaves, and maintain good orchard hygiene to prevent further spread.\n",
            "Hindi": "फफूंदनाशकों का उपयोग करें, संक्रमित पत्तियों को हटा दें और अच्छी बागवानी स्वच्छता बनाए रखें आगे फैलने से रोकने के लिए।\n"
        },
        'cause': {
            "English": "Caused by the fungus *Alternaria solani*, which thrives in warm, wet conditions.\n",
            "Hindi": "फफूंद *अल्टरनारिया सोलानी* के कारण होता है, जिसका प्रसार गर्म, नम स्थितियों में होता है।\n"
        },
        'symptoms': {
            "English": "Dark brown spots with concentric rings on leaves, leading to defoliation.\n",
            "Hindi": "पत्तियों पर गहरे भूरे रंग के धब्बे, जिसके कारण पत्तियों का झड़ना होता है।\n"
        },
        'management': {
            "English": "Use certified seed potatoes, rotate crops, and apply fungicides as needed.\n",
            "Hindi": "प्रमाणित आलू बीज का उपयोग करें, फसल चक्रण करें और आवश्यकतानुसार फफूंदनाशकों का उपयोग करें।\n"
        },
        'fertilizer': {
            "English": "Apply a balanced fertilizer like 10-20-20 (NPK) at planting, and side-dress with nitrogen (34-0-0) after the plants reach 6 inches in height. Use 1.5 pounds per 100 feet of row.\n",
            "Hindi": "रोपण के समय 10-20-20 (NPK) जैसे संतुलित उर्वरक का उपयोग करें, और पौधों के 6 इंच की ऊँचाई तक पहुँचने के बाद नाइट्रोजन (34-0-0) से साइड-ड्रेसिंग करें। पंक्ति के 100 फीट प्रति 1.5 पाउंड का उपयोग करें।\n"
        }
    },
    'Potato___healthy': {
        'precaution': {
            "English": "No precautions needed, the plant is healthy.\n",
            "Hindi": "कोई सावधानी की आवश्यकता नहीं है, पौधा स्वस्थ है।\n"
        },
        'cause': {
            "English": "N/A\n",
            "Hindi": "एन/ए\n"
        },
        'symptoms': {
            "English": "N/A\n",
            "Hindi": "एन/ए\n"
        },
        'management': {
            "English": "Continue regular care to maintain plant health.\n",
            "Hindi": "पौधे की सेहत बनाए रखने के लिए नियमित देखभाल जारी रखें।\n"
        },
        'fertilizer': {
            "English": "Use a balanced fertilizer like 10-10-10 (NPK) at planting, followed by a side-dressing of nitrogen (34-0-0) after plants reach 6 inches in height. Use 2-3 pounds per 100 feet of row.\n",
            "Hindi": "रोपण के समय 10-10-10 (NPK) जैसे संतुलित उर्वरक का उपयोग करें, और पौधों के 6 इंच की ऊँचाई तक पहुँचने के बाद नाइट्रोजन (34-0-0) से साइड-ड्रेसिंग करें। पंक्ति के 100 फीट प्रति 2-3 पाउंड का उपयोग करें।\n"
        }
    },
    'Squash___Powdery_mildew': {
        'precaution': {
            "English": "Apply fungicides, remove infected leaves, and maintain good orchard hygiene to prevent further spread.\n",
            "Hindi": "फफूंदनाशकों का उपयोग करें, संक्रमित पत्तियों को हटा दें और अच्छी बागवानी स्वच्छता बनाए रखें आगे फैलने से रोकने के लिए।\n"
        },
        'cause': {
            "English": "Caused by several species of fungi, including *Podosphaera xanthii* and *Erysiphe cichoracearum*.\n",
            "Hindi": "फफूंद की कई प्रजातियों के कारण होता है, जिसमें *पोडोस्फेरा जैन्थी* और *एरीसिफे सिकोरेसियरम* शामिल हैं।\n"
        },
        'symptoms': {
            "English": "White, powdery fungal growth on leaves, stems, and fruit.\n",
            "Hindi": "पत्तियों, तनों और फल पर सफेद, पाउडरी फफूंदी वृद्धि।\n"
        },
        'management': {
            "English": "Apply fungicides, improve air circulation by spacing plants properly, and water plants at the base to keep leaves dry.\n",
            "Hindi": "फफूंदनाशकों का उपयोग करें, पौधों के बीच उचित दूरी रखकर हवा के संचार में सुधार करें और पौधों के आधार पर पानी दें ताकि पत्तियां सूखी रहें।\n"
        },
        'fertilizer': {
            "English": "Apply a balanced fertilizer like 10-10-10 (NPK) at planting, and side-dress with calcium nitrate (15.5-0-0) when flowering begins. Use 2-3 pounds per 100 square feet.\n",
            "Hindi": "रोपण के समय 10-10-10 (NPK) जैसे संतुलित उर्वरक का उपयोग करें, और फूल लगने पर कैल्शियम नाइट्रेट (15.5-0-0) से साइड-ड्रेसिंग करें। प्रति 100 वर्ग फीट 2-3 पाउंड का उपयोग करें।\n"
        }
    },
    'Strawberry___Leaf_scorch': {
        'precaution': {
            "English": "Remove infected leaves, improve air circulation, and apply bactericides to prevent further spread.\n",
            "Hindi": "संक्रमित पत्तियों को हटा दें, हवा के संचार में सुधार करें और बैक्टीरियासाइड्स का उपयोग करें आगे फैलने से रोकने के लिए।\n"
        },
        'cause': {
            "English": "Caused by the fungus *Diplocarpon earlianum*, which thrives in wet, warm conditions.\n",
            "Hindi": "फफूंद *डिप्लोकार्पोन एरलियनम* के कारण होता है, जिसका प्रसार नम, गर्म स्थितियों में होता है।\n"
        },
        'symptoms': {
            "English": "Irregular, dark purple spots on leaves, leading to leaf browning and drying.\n",
            "Hindi": "पत्तियों पर असामान्य, गहरे बैंगनी रंग के धब्बे, जिसके कारण पत्तियों का भूरा होना और सूखना होता है।\n"
        },
        'management': {
            "English": "Remove and destroy infected leaves, ensure good air circulation, and apply fungicides as needed.\n",
            "Hindi": "संक्रमित पत्तियों को हटा दें और नष्ट करें, अच्छा हवा के संचार सुनिश्चित करें, और आवश्यकतानुसार फफूंदनाशकों का उपयोग करें।\n"
        },
        'fertilizer': {
            "English": "Use a balanced fertilizer like 10-10-10 (NPK) in spring, and side-dress with ammonium nitrate (33-0-0) at mid-season. Use 2-3 pounds per 100 feet of row.\n",
            "Hindi": "वसंत में 10-10-10 (NPK) जैसे संतुलित उर्वरक का उपयोग करें, और मध्य-सीजन में अमोनियम नाइट्रेट (33-0-0) से साइड-ड्रेसिंग करें। पंक्ति के 100 फीट प्रति 2-3 पाउंड का उपयोग करें।\n"
        }
    },
    'Tomato___Early_blight': {
        'precaution': {
            "English": "Apply fungicides, remove infected leaves, and maintain good orchard hygiene to prevent further spread.\n",
            "Hindi": "फफूंदनाशकों का उपयोग करें, संक्रमित पत्तियों को हटा दें, और अच्छी बागवानी स्वच्छता बनाए रखें आगे फैलने से रोकने के लिए।\n"
        },
        'cause': {
            "English": "Caused by the fungus *Alternaria solani*, which thrives in warm, wet conditions.\n",
            "Hindi": "फफूंद *अल्टरनारिया सोलानी* के कारण होता है, जिसका प्रसार गर्म, नम स्थितियों में होता है।\n"
        },
        'symptoms': {
            "English": "Dark brown spots with concentric rings on leaves, leading to defoliation.\n",
            "Hindi": "पत्तियों पर गहरे भूरे रंग के धब्बे, जिसके कारण पत्तियों का झड़ना होता है।\n"
        },
        'management': {
            "English": "Use certified seeds, rotate crops, and apply fungicides during wet weather.\n",
            "Hindi": "प्रमाणित बीज का उपयोग करें, फसल चक्रण करें, और नम मौसम में फफूंदनाशकों का उपयोग करें।\n"
        },
        'fertilizer': {
            "English": "Use a balanced fertilizer like 10-10-10 (NPK) at planting, and side-dress with calcium nitrate (15.5-0-0) when fruiting begins. Apply 2-3 pounds per 100 feet of row.\n",
            "Hindi": "रोपण के समय 10-10-10 (NPK) जैसे संतुलित उर्वरक का उपयोग करें, और फल लगने पर कैल्शियम नाइट्रेट (15.5-0-0) से साइड-ड्रेसिंग करें। पंक्ति के 100 फीट पर 2-3 पाउंड का उपयोग करें।\n"
        }
    },
    'Tomato___healthy': {
        'precaution': {
            "English": "No precautions needed, the plant is healthy.\n",
            "Hindi": "कोई सावधानी की आवश्यकता नहीं है, पौधा स्वस्थ है।\n"
        },
        'cause': {
            "English": "N/A\n",
            "Hindi": "एन/ए\n"
        },
        'symptoms': {
            "English": "N/A\n",
            "Hindi": "एन/ए\n"
        },
        'management': {
            "English": "Continue regular care to maintain plant health.\n",
            "Hindi": "पौधे की सेहत बनाए रखने के लिए नियमित देखभाल जारी रखें।\n"
        },
        'fertilizer': {
            "English": "Use a balanced fertilizer like 10-10-10 (NPK) at planting, and side-dress with calcium nitrate (15.5-0-0) during the fruiting stage. Apply 2-3 pounds per 100 feet of row.\n",
            "Hindi": "रोपण के समय 10-10-10 (NPK) जैसे संतुलित उर्वरक का उपयोग करें, और फल लगने के दौरान कैल्शियम नाइट्रेट (15.5-0-0) से साइड-ड्रेसिंग करें। पंक्ति के 100 फीट पर 2-3 पाउंड का उपयोग करें।\n"
        }
    }
}


    def get_precaution(self, class_name):
        disease_info = self.disease_details.get(class_name)
        if disease_info:
            return (
                f"**{translations[self.language]['precaution']}:** {disease_info['precaution'][self.language]}\n"
                f"**{translations[self.language]['cause']}:** {disease_info['cause'][self.language]}\n"
                f"**{translations[self.language]['symptoms']}:** {disease_info['symptoms'][self.language]}\n"
                f"**{translations[self.language]['management']}:** {disease_info['management'][self.language]}\n"
                f"**{translations[self.language]['fertilizer']}:** {disease_info['fertilizer'][self.language]}\n"
            )
        else:
            return "No specific precautions available for this disease."

def display_prediction(class_name, language):
    precaution = Precaution(language=language)
    st.success(f"🌿 Model predicts: **{class_name}**")
    precautions = precaution.get_precaution(class_name)
    st.warning(f"{precautions}")



def soil_classification_page():
    st.markdown(""" <div style="text-align: center; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 10px;"> <h1 style="color: #ffffff; font-family: 'Poppins', sans-serif; font-size: 48px;">Soil Classification and Crop Recommendation</h1> </div> """, unsafe_allow_html=True)

    st.write("Upload an image of soil, and the system will classify it and recommend the best crops to grow.")

    soil_image = st.file_uploader("Upload Soil Image", type=["jpg", "jpeg", "png"])

    if soil_image:
        st.image(soil_image, caption="Uploaded Soil Image", use_column_width=True, clamp=True)

        # Add a "Predict" button
        predict_button = st.button("Predict")

        if predict_button:
            model = tf.keras.models.load_model("keras_model.h5")

            # Process the uploaded image for prediction
            image = tf.keras.preprocessing.image.load_img(soil_image, target_size=(224, 224))
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            input_arr = np.array([input_arr]) / 255.0  # Normalize
            prediction = model.predict(input_arr)

            # Define the soil classes and associated crop recommendations
            soil_classes = {
                0: "Yellow Soil",
                1: "Peat Soil",
                2: "Laterite Soil",
                3: "Cinder Soil",
                4: "Black Soil"
            }

            crop_recommendations = {
                "Yellow Soil": {
        "Best Crops": ["Cotton", "Wheat"],
        "Fertilizer": "Apply NPK 12-32-16 fertilizer at 100 kg/ha.",
        "Cost Estimator (Fertilizer)": "₹5,000 - ₹7,000 per ha",
        "Cost Estimator (Crop)": "₹15,000 - ₹20,000 per ha",
        "Suitable City": "Ahmedabad, Gujarat",
        "Government Scheme (MSP)": "₹4,500 per quintal (Cotton), ₹2,500 per quintal (Wheat)",
        "Supported Tractors": "Mahindra, John Deere"
    },
    "Peat Soil": {
        "Best Crops": ["Rice", "Sugarcane"],
        "Fertilizer": "Use organic compost and NPK 10-26-26 fertilizer.",
        "Cost Estimator (Fertilizer)": "₹3,000 - ₹5,000 per ha",
        "Cost Estimator (Crop)": "₹25,000 - ₹35,000 per ha",
        "Suitable City": "Kolkata, West Bengal",
        "Government Scheme (MSP)": "₹2,500 per quintal (Rice), ₹3,500 per quintal (Sugarcane)",
        "Supported Tractors": "Sonalika, New Holland"
    },
    "Laterite Soil": {
        "Best Crops": ["Coconut", "Tea"],
        "Fertilizer": "Apply potassium-rich fertilizers like K2O.",
        "Cost Estimator (Fertilizer)": "₹4,000 - ₹6,000 per ha",
        "Cost Estimator (Crop)": "₹30,000 - ₹40,000 per ha",
        "Suitable City": "Kochi, Kerala",
        "Government Scheme (MSP)": "₹1,500 per quintal (Coconut), ₹2,000 per quintal (Tea)",
        "Supported Tractors": "Escorts, Swaraj"
    },
    "Cinder Soil": {
        "Best Crops": ["Millet", "Maize"],
        "Fertilizer": "Use balanced NPK 15-15-15 fertilizer.",
        "Cost Estimator (Fertilizer)": "₹2,500 - ₹4,000 per ha",
        "Cost Estimator (Crop)": "₹10,000 - ₹15,000 per ha",
        "Suitable City": "Hyderabad, Telangana",
        "Government Scheme (MSP)": "₹2,000 per quintal (Millet), ₹1,800 per quintal (Maize)",
        "Supported Tractors": "Mahindra, John Deere"
    },
    "Black Soil": {
        "Best Crops": ["Cotton", "Soybean"],
        "Fertilizer": "Apply NPK 20-10-10 fertilizer at 50 kg/ha.",
        "Cost Estimator (Fertilizer)": "₹4,500 - ₹6,500 per ha",
        "Cost Estimator (Crop)": "₹18,000 - ₹25,000 per ha",
        "Suitable City": "Nagpur, Maharashtra",
        "Government Scheme (MSP)": "₹4,500 per quintal (Cotton), ₹3,500 per quintal (Soybean)",
        "Supported Tractors": "Sonalika, New Holland"
    }
          }

            predicted_class = np.argmax(prediction)
            soil_type = soil_classes[predicted_class]

            # Display results in a table format
            st.write("<div style='background-color: #000001; padding: 5px; border-radius: 10px;'>## Soil Classification and Crop Recommendation</div>", unsafe_allow_html=True)
            st.write("<div style='background-color: #000001; padding: 5px; border-radius: 10px;'>### Soil Type: **{}**</div>".format(soil_type), unsafe_allow_html=True)
            st.write("<div style='background-color: #000001; padding: 5px; border-radius: 10px;'>### Recommendations:</div>", unsafe_allow_html=True)
            st.markdown("""
<style>
table {{
    border-collapse: collapse;
    width: 100%;
    background-color: #f7f7f7;
}}
th, td {{
    border: 1px solid #ddd;
    padding: 10px;
    text-align: left;
    color: #333;
    background-color: #f7f7f7;
}}
th {{
    background-color: #f0f0f0;
}}
</style>
<table>
<tr>
    <th>Category</th>
    <th>Recommendation</th>
</tr>
<tr>
    <td><b>Best Crops</b></td>
    <td>{0}</td>
</tr>
<tr>
    <td><b>Fertilizer</b></td>
    <td>{1}</td>
</tr>
<tr>
    <td><b>Cost Estimator (Fertilizer)</b></td>
    <td>{2}</td>
</tr>
<tr>
    <td><b>Cost Estimator (Crop)</b></td>
    <td>{3}</td>
</tr>
<tr>
    <td><b>Suitable City</b></td>
    <td>{4}</td>
</tr>
<tr>
    <td><b>Government Scheme (MSP)</b></td>
    <td>{5}</td>
</tr>
<tr>
    <td><b>Supported Tractors</b></td>
    <td>{6}</td>
</tr>
</table>
""".format(
    ', '.join(crop_recommendations[soil_type]['Best Crops']),
    crop_recommendations[soil_type]['Fertilizer'],
    crop_recommendations[soil_type]['Cost Estimator (Fertilizer)'],
    crop_recommendations[soil_type]['Cost Estimator (Crop)'],
    crop_recommendations[soil_type]['Suitable City'],
    crop_recommendations[soil_type]['Government Scheme (MSP)'],
    crop_recommendations[soil_type]['Supported Tractors']
), unsafe_allow_html=True)
            
if "language" not in st.session_state:
    st.session_state.language = "English"

# Sidebar for language selection
st.sidebar.title("🌱 CropPro Assist")
language_selection = st.sidebar.selectbox("Language", options=["English", "Hindi"], index=0 if st.session_state.language == "English" else 1)
st.session_state.language = language_selection


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

# Main content based on user selection
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a page", [
    "Home",
    "Disease Recognition",
    "Webcam live feed",
    "Soil Classification",
    "Weather",
    "Agriculture News",
    "Disease Library",
    "Contact Bot"
])

if app_mode == "Home":
    st.markdown(f"<h1>{translations[st.session_state.language]['title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<div class='card'><h3>{translations[st.session_state.language]['welcome']}</h3>"
                f"<p>{translations[st.session_state.language]['mission']}</p>"
                f"<h4>{translations[st.session_state.language]['how_it_works']}</h4>"
                f"<ul><li><strong>{translations[st.session_state.language]['upload_image']}</strong> {translations[st.session_state.language]['disease_recognition']}</li>"
                f"<li><strong>{translations[st.session_state.language]['analysis']}</strong> Our system will process the image using advanced algorithms to identify potential diseases.</li>"
                f"<li><strong>{translations[st.session_state.language]['results']}</strong> View the results and recommendations for further action.</li></ul></div>",
                unsafe_allow_html=True)

elif app_mode == "Disease Recognition":
    st.markdown(f"<h1>{translations[st.session_state.language]['title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p>{translations[st.session_state.language]['upload_prompt']}</p>", unsafe_allow_html=True)
    
    plant_prefix = st.text_input(translations[st.session_state.language]["enter_plant_name"])
    test_image = st.file_uploader(translations[st.session_state.language]["choose_image"])
    
    if test_image:
        st.image(test_image, caption="Uploaded Image", use_column_width=True, clamp=True)
        if st.button(translations[st.session_state.language]["predict"]):
            with st.spinner(translations[st.session_state.language]["analyzing_image"]):
                predictions = model_prediction(test_image)

                class_names = [
                    'Apple___Cedar_apple_rust', 'Apple___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy',
                    'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
                    'Peach___Bacterial_spot',  'Pepper,_bell___Bacterial_spot',
                    'Potato___Early_blight', 
                    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
                    'Tomato___Early_blight'
                ]
                
                filtered_class_names = filter_class_names(plant_prefix, class_names)
                if filtered_class_names:
                    class_name = get_class_name_from_predictions(predictions, filtered_class_names, class_names)
                    display_prediction(class_name, language=st.session_state.language)
                else:
                    st.warning(translations[st.session_state.language]["no_classes_match"])
    else:
        st.warning(translations[st.session_state.language]["please_upload_image"])

elif app_mode == "Soil Classification":
    soil_classification_page()

elif app_mode == "Weather":
    weather_page()

elif app_mode == "Agriculture News":
    news_page()

elif app_mode == "Disease Library":
    st.markdown(f"<h1>{translations[st.session_state.language]['disease_library']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<div class='card'><h4>{translations[st.session_state.language]['browse_library']}</h4>"
                f"<p>{translations[st.session_state.language]['select_disease']}</p></div>", unsafe_allow_html=True)
    
    disease_list = list(Precaution(language=st.session_state.language).disease_details.keys())
    disease_selection = st.selectbox(translations[st.session_state.language]["select_disease"], disease_list)
    
    precaution = Precaution(language=st.session_state.language)
    st.info(precaution.get_precaution(disease_selection))

elif app_mode == "Contact Bot":
    st.markdown(f"<h1>{translations[st.session_state.language]['contact_bot']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p>{translations[st.session_state.language]['chat_with_bot']}</p>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; height: 70vh;">
            <iframe 
                src="https://cdn.botpress.cloud/webchat/v2.1/shareable.html?botId=f30e8203-66c5-4e87-a73d-d2e4799b1235"
                style="border: none; width: 400px; height: 500px;">
            </iframe>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif app_mode =="Webcam live feed":
    st.markdown(f"<h1>{translations[st.session_state.language]['webcam_live_feed']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p>{translations[st.session_state.language]['take_photo']}</p>", unsafe_allow_html=True)
    
    plant_prefix = st.text_input(translations[st.session_state.language]["enter_plant_name"])
    run = st.checkbox(translations[st.session_state.language]["run"])
    
    if run:
        img_file_buffer = st.camera_input(translations[st.session_state.language]["take_photo"])
        
        if img_file_buffer:
            st.image(img_file_buffer)
            if st.button(translations[st.session_state.language]["predict"]):
                with st.spinner(translations[st.session_state.language]["analyzing_image"]):
                    predictions = model_prediction(img_file_buffer)

                    class_names = [
                        'Apple___Cedar_apple_rust', 'Apple___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy',
                        'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
                        'Peach___Bacterial_spot',  'Pepper,_bell___Bacterial_spot',
                        'Potato___Early_blight', 
                        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
                        'Tomato___Early_blight'
                    ]
                    
                    filtered_class_names = filter_class_names(plant_prefix, class_names)
                    if filtered_class_names:
                        class_name = get_class_name_from_predictions(predictions, filtered_class_names, class_names)
                        display_prediction(class_name, language=st.session_state.language)
                    else:
                        st.warning(translations[st.session_state.language]["no_classes_match"])
