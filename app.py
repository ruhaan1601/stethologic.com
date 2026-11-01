import os
import gc
import json
import urllib.request
import requests
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
import librosa
import gradio as gr

# Force PyTorch to use minimal memory
torch.set_num_threads(1)
torch.set_grad_enabled(False)

# -------------------------------------------------------------
# 1. DOWNLOAD WEIGHTS & LABELS
# -------------------------------------------------------------
WEIGHTS_FILE = "model_weights.pt"
LABELS_FILE = "labels.json"

WEIGHTS_URL = "https://github.com/ruhaan1601/stethologic.com/releases/download/v1.0/model_weights.pt"
LABELS_URL = "https://github.com/ruhaan1601/stethologic.com/releases/download/v1.0/labels.json"

def download_if_missing():
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    
    if not os.path.exists(WEIGHTS_FILE):
        print("Downloading PyTorch model weights...")
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_FILE)
        
    if not os.path.exists(LABELS_FILE):
        print("Downloading class labels...")
        urllib.request.urlretrieve(LABELS_URL, LABELS_FILE)

download_if_missing()

with open(LABELS_FILE, "r") as f:
    class_names = json.load(f)

# Initialize ResNet18 without downloading unnecessary default weights
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
state = torch.load(WEIGHTS_FILE, map_location="cpu")
model.load_state_dict(state)
model.eval()
del state
gc.collect()

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------------------------------------------------------------
# 2. DR. INFO (GEMINI API CHATBOT)
# -------------------------------------------------------------
gemini_api_key = os.environ.get("GEMINI_API_KEY")

SYSTEM_INSTRUCTION = """
You are 'Dr. Info', clinical triage assistant for StethoLogic (stethologic.com).
- Discuss cough sounds, lung sounds (wheezing, coarse crackles, voluntary coughs), and respiratory tips.
- Explain conditions (Bronchitis, Pneumonia, Asthma, Whooping Cough, Voluntary Cough, Normal) clearly.
- Reply in whatever language the user asks (English, Hindi, Tamil, Telugu, etc.).
- Maintain clear boundaries: give triage guidance, not an authoritative medical prescription.
- Urge immediate emergency care if danger signs exist (shortness of breath at rest, blood in sputum, chest pain, high fever).
- Mention partner services (ordering medicine or consulting an online doctor) when helpful.
- Keep answers warm, encouraging, concise, and structured with bullet points.
"""

def dynamic_chat(message, history):
    if not gemini_api_key:
        return "⚠️ Chatbot API key not detected. Please verify GEMINI_API_KEY is configured in your Render Environment settings."
    
    try:
        contents = []
        for turn in history:
            user_msg = turn[0]
            bot_msg = turn[1]
            if user_msg:
                contents.append({"role": "user", "parts": [{"text": str(user_msg)}]})
            if bot_msg:
                contents.append({"role": "model", "parts": [{"text": str(bot_msg)}]})
        
        contents.append({"role": "user", "parts": [{"text": str(message)}]})

        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_api_key}"
        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.3}
        }
        
        resp = requests.post(api_url, json=payload, timeout=20)
        data = resp.json()
        
        if "candidates" in data and len(data["candidates"]) > 0:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        elif "error" in data:
            return f"API Error: {data['error'].get('message', 'Unknown issue')}"
        else:
            return "Unable to process the query right now. Please try again."

    except Exception as e:
        return f"Service temporarily busy. Please try again shortly. (Error: {str(e)})"

# -------------------------------------------------------------
# 3. ACOUSTIC REASONINGS & TRANSLATIONS
# -------------------------------------------------------------
SYMPTOM_REASONS = {
    "English": {
        "Normal": "Sound profile demonstrates standard laminar airflow without mucosal rattle, abnormal wheezing, or explosive airway obstruction.",
        "Bronchitis": "Audio displays low-pitched mucosal rattling and deep resonant bronchial vibrations typical of tracheobronchial inflammation.",
        "Pneumonia": "Acoustic signature exhibits discontinuous fine crackles and diminished resonance indicating fluid or exudate accumulation in alveolar sacs.",
        "Asthma_Wheeze": "High-pitched, musical harmonic frequencies detected during expiration, characteristic of narrowed bronchiolar airways.",
        "Whooping_Cough": "Paroxysmal cough bursts followed by a distinctive high-pitched acoustic inspiratory 'whoop' intake.",
        "Voluntary_Cough": "Acoustic signature lacks involuntary deep thoracic pressure build-up and shows a sharp, mechanical onset typical of simulated or voluntary throat clearing."
    },
    "हिन्दी (Hindi)": {
        "Normal": "श्वसन ध्वनि सामान्य है। इसमें वायुमार्ग में रुकावट, घरघराहट या सूजन के कोई लक्षण नहीं पाए गए।",
        "Bronchitis": "गहरी गूंज और बलगम की कंपन ध्वनि पाई गई, जो श्वासनली में सूजन (ब्रोंकाइटिस) का संकेत देती है।",
        "Pneumonia": "फेफड़ों में तरल जमाव और बारीक चटकने जैसी ध्वनियां (क्रॅकल्स) पाई गईं, जो निमोनिया का संकेत हैं।",
        "Asthma_Wheeze": "सांस छोड़ते समय सीटी जैसी तीखी आवाज (घरघराहट) पाई गई, जो वायुमार्ग सिकुड़ने का लक्षण है।",
        "Whooping_Cough": "लगातार तेज खांसी के झटके और अंत में सांस खींचते समय तेज आवाज (हूप) की पहचान हुई है।",
        "Voluntary_Cough": "आवाज में फेफड़ों का स्वाभाविक आंतरिक दबाव नहीं है। यह जानबूझकर या बनावटी रूप से की गई खांसी का संकेत देती है।"
    },
    "தமிழ் (Tamil)": {
        "Normal": "வழக்கமான சுவாச ஒலி. மூச்சுப்பாதையில் அடைப்போ அல்லது சளி சத்தமோ காணப்படவில்லை.",
        "Bronchitis": "மூச்சுக்குழாயில் சளியின் அதிர்வு மற்றும் ஆழ்ந்த இருமல் ஒலி காணப்படுகிறது (மூச்சுக்குழாய் அழற்சி).",
        "Pneumonia": "நுரையீரலில் நீர் அல்லது சளி தேங்குவதால் ஏற்படும் நுட்பமான வெடிப்பு ஒலிகள் பதிவாகியுள்ளன (நிமோனியா).",
        "Asthma_Wheeze": "மூச்சுப்பாதை சுருங்குவதால் வெளிப்படும் விசில் போன்ற வீசிங் ஒலி கண்டறியப்பட்டது (ஆஸ்துமா).",
        "Whooping_Cough": "தொடர் இருமல் தாக்குதல்களும், மூச்சை உள்ளிழுக்கும்போது எழும் கூர்மையான 'ஹூப்' ஒலியும் உள்ளன.",
        "Voluntary_Cough": "இயற்கையான ஆழமான சுவாச அழுத்தமின்றி, செயற்கையாக அல்லது தாமாகச் செய்யப்பட்ட தொண்டை சரிசெய்தல் இருமல் போல் தெரிகிறது."
    },
    "తెలుగు (Telugu)": {
        "Normal": "సాధారణ శ్వాస ధ్వని. శ్వాసనాళాలలో ఎటువంటి అడ్డంకులు లేదా పిల్లికూతలు లేవు.",
        "Bronchitis": "శ్వాసనాళాల వాపు మరియు శ్లేష్మం కంపనం ధ్వని గుర్తించబడింది (బ్రోన్కైటిస్ లక్షణం).",
        "Pneumonia": "ఊపిరితిత్తులలో ద్రవం చేరడం వల్ల వచ్చే చిన్న పగుళ్ల వంటి శబ్దాలు గుర్తించబడ్డాయి (న్యుమోనియా).",
        "Asthma_Wheeze": "శ్వాస వదిలేటప్పుడు పిల్లికూతలు మరియు ఈల శబ్దం స్పష్టంగా ఉంది (ఆస్తమా సూచన).",
        "Whooping_Cough": "వరుస దగ్గు దాడులు మరియు శ్వాస తీసుకునేటప్పుడు వచ్చే తీక్షణమైన 'హూప్' శబ్దం కనిపించింది.",
        "Voluntary_Cough": "ఊపిరితిత్తుల సహజ పీడనం లోపించినందున ఇది ఉద్దేశపూర్వకంగా చేసిన కృత్రిమ లేదా సాధారణ దగ్గుగా కనిపిస్తోంది."
    }
}

I18N = {
    "English": {
        "title": "# 🩺 **StethoLogic**",
        "subtitle": "### *Advanced Acoustic Disease Screening & Clinical Triage*",
        "banner": "📢 SPONSORED: Get 20% Off at 1mg & Apollo Pharmacy with Code: HEALTH20",
        "tab_screener": "🎙️ Acoustic Screener",
        "tab_chat": "💬 Talk to Dr. Info",
        "audio_label": "Record Cough or Respiration Sound",
        "btn_analyze": "⚡ Analyze Acoustic Signature",
        "partner_header": "### 🏥 Connected Healthcare Services",
        "res_label": "Diagnostic Assessment",
        "why_label": "Acoustic Symptom Breakdown (Why this condition is indicated):",
        "info_bot_cta": "ℹ️ **For more information talk to our Info-Bot (Dr. Info).**",
        "spec_label": "Mel-Frequency Spectrogram",
        "ad_box": '<small style="color:gray;">ADVERTISEMENT</small><p><strong>N95 Protective Masks & Home Nebulizers</strong> — Same Day Delivery Available.</p>',
        "quiet_err": "⚠️ Audio too quiet. Hold microphone closer and retry.",
        "input_err": "⚠️ Please provide an audio recording first."
    },
    "हिन्दी (Hindi)": {
        "title": "# 🩺 **StethoLogic**",
        "subtitle": "### *उन्नत ध्वनिक रोग जांच एवं नैदानिक सहायता*",
        "banner": "📢 प्रायोजित: 1mg और अपोलो फार्मेसी पर 20% छूट पाएं | कोड: HEALTH20",
        "tab_screener": "🎙️ ऑडियो जांच",
        "tab_chat": "💬 डॉ. इन्फो से बात करें",
        "audio_label": "खांसी या सांस की आवाज रिकॉर्ड करें",
        "btn_analyze": "⚡ ध्वनिक विश्लेषण शुरू करें",
        "partner_header": "### 🏥 पार्टनर स्वास्थ्य सेवाएं",
        "res_label": "जांच परिणाम",
        "why_label": "ध्वनिक लक्षण विश्लेषण (यह स्थिति क्यों पाई गई):",
        "info_bot_cta": "ℹ️ **For more information talk to our Info-Bot (डॉ. इन्फो से पूछें).**",
        "spec_label": "मेल-फ़्रीक्वेंसी स्पेक्ट्रोग्राम",
        "ad_box": '<small style="color:gray;">विज्ञापन</small><p><strong>N95 मास्क और नेबुलाइजर</strong> — उसी दिन डिलीवरी उपलब्ध।</p>',
        "quiet_err": "⚠️ आवाज़ बहुत धीमी है। माइक पास लाकर दोबारा प्रयास करें।",
        "input_err": "⚠️ कृपया पहले ऑडियो रिकॉर्ड या अपलोड करें।"
    },
    "தமிழ் (Tamil)": {
        "title": "# 🩺 **StethoLogic**",
        "subtitle": "### *மேம்பட்ட ஒலி அடிப்படையிலான சுவாச நோய் பரிசோதனை*",
        "banner": "📢 விளம்பரம்: 1mg மற்றும் அப்பல்லோ பார்மசியில் 20% தள்ளுபடி பெற குறியீடு: HEALTH20",
        "tab_screener": "🎙️ ஆடியோ பரிசோதனை",
        "tab_chat": "💬 டாக்டர் இன்போவிடம் பேசுக",
        "audio_label": "இருமல் அல்லது மூச்சு ஒலியைப் பதிவு செய்க",
        "btn_analyze": "⚡ ஒலி பகுப்பாய்வைத் தொடங்கு",
        "partner_header": "### 🏥 கூட்டாளர் மருத்துவச் சேவைகள்",
        "res_label": "பரிசோதனை முடிவு",
        "why_label": "ஒலி அறிகுறிகள் (இந்த நிலை ஏன் கண்டறியப்பட்டது):",
        "info_bot_cta": "ℹ️ **For more information talk to our Info-Bot (டாக்டர் இன்போவிடம் ஆலோசிக்கவும்).**",
        "spec_label": "மெல்-அதிர்வெண் ஸ்பெக்ட்ரோகிராம்",
        "ad_box": '<small style="color:gray;">விளம்பரம்</small><p><strong>N95 முகக்கவசங்கள் மற்றும் நெபுலைசர்கள்</strong> — ஒரே நாளில் டெலிவரி.</p>',
        "quiet_err": "⚠️ ஒலி மிகவும் மெதுவாக உள்ளது. மைக்கை அருகில் வைத்து மீண்டும் முயற்சிக்கவும்.",
        "input_err": "⚠️ தயவுசெய்து முதலில் ஆடியோவை பதிவு செய்யவும் அல்லது பதிவேற்றவும்."
    },
    "తెలుగు (Telugu)": {
        "title": "# 🩺 **StethoLogic**",
        "subtitle": "### *ఆధునిక శబ్ద శ్వాసకోశ వ్యాధి పరీక్ష మరియు క్లినికల్ సహాయం*",
        "banner": "📢 ప్రాయోజితం: 1mg మరియు అపోలో ఫార్మసీలో 20% తగ్గింపు పొందండి | కోడ్: HEALTH20",
        "tab_screener": "🎙️ ఆడియో స్క్రీనర్",
        "tab_chat": "💬 డాక్టర్ ఇన్ఫోతో మాట్లాడండి",
        "audio_label": "దగ్గు లేదా శ్వాస శబ్దాన్ని రికార్డ్ చేయండి",
        "btn_analyze": "⚡ శబ్ద విశ్లేషణను ప్రారంభించండి",
        "partner_header": "### 🏥 భాగస్వామ్య ఆరోగ్య సేవలు",
        "res_label": "పరీక్ష ఫలితం",
        "why_label": "ధ్వని లక్షణాల వివరణ (ఈ పరిస్థితి ఎందుకు గుర్తించబడింది):",
        "info_bot_cta": "ℹ️ **For more information talk to our Info-Bot (డాక్టర్ ఇన్ఫోని అడగండి).**",
        "spec_label": "మెల్-ఫ్రీక్వెన్సీ స్పెక్ట్రోగ్రామ్",
        "ad_box": '<small style="color:gray;">ప్రకటన</small><p><strong>N95 మాస్క్‌లు & నెబ్యులైజర్లు</strong> — అదే రోజు డెలివరీ లభ్యం.</p>',
        "quiet_err": "⚠️ శబ్దం చాలా తక్కువగా ఉంది. మైక్రోఫోన్‌ను దగ్గరగా ఉంచి మళ్లీ ప్రయత్నించండి.",
        "input_err": "⚠️ దయచేసి ముందుగా ఆడియోను రికార్డ్ చేయండి లేదా అప్‌లోడ్ చేయండి."
    }
}

custom_css = """
footer {visibility: hidden !important;}
.gradio-container {max-width: 980px !important; margin: auto !important;}
.sponsor-banner {
    background: linear-gradient(90deg, #0f172a, #2563eb);
    color: white;
    padding: 12px;
    border-radius: 8px;
    text-align: center;
    font-weight: bold;
    margin: 10px 0;
}
.ad-box {
    border: 2px dashed #9ca3af;
    padding: 10px;
    text-align: center;
    border-radius: 6px;
    background-color: rgba(156, 163, 175, 0.08);
}
"""

# -------------------------------------------------------------
# 4. LIGHTWEIGHT INFERENCE PIPELINE (NO MATPLOTLIB)
# -------------------------------------------------------------
def analyze_audio(audio_path, language):
    lang_dict = I18N.get(language, I18N["English"])
    reasons_dict = SYMPTOM_REASONS.get(language, SYMPTOM_REASONS["English"])
    
    if audio_path is None:
        return lang_dict["input_err"], "", "", None, {}
    
    spec_path = "eval_spec.png"
    try:
        signal, sr = librosa.load(audio_path, sr=22050, duration=3.0)
        if np.max(np.abs(signal)) < 0.02:
            return lang_dict["quiet_err"], "", "", None, {}

        # Mel Spectrogram converted to 8-bit image without matplotlib
        mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = ((mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6) * 255).astype(np.uint8)
        
        img = Image.fromarray(mel_norm).convert('RGB')
        img.save(spec_path)

        input_tensor = img_transforms(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)

        prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
        top_disease = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[top_disease] * 100

        result_text = f"🚨 {top_disease} ({confidence:.1f}% Confidence Match)"
        why_text = reasons_dict.get(
            top_disease, 
            "The acoustic frequency distribution closely aligns with the sound markers of this condition."
        )
        cta_text = lang_dict["info_bot_cta"]

        del signal, mel, mel_db, mel_norm, img, input_tensor, outputs, probs
        gc.collect()

        return result_text, why_text, cta_text, spec_path, prob_dict

    except Exception as e:
        gc.collect()
        return f"Error: {str(e)}", "", "", None, {}

def change_language(selected_lang):
    d = I18N.get(selected_lang, I18N["English"])
    banner_html = f'<div class="sponsor-banner">{d["banner"]}</div>'
    ad_html = f'<div class="ad-box" style="width:100%; margin-top:15px;">{d["ad_box"]}</div>'
    header_md = f"{d['title']}\n{d['subtitle']}"
    return [
        gr.update(value=header_md),
        gr.update(value=banner_html),
        gr.update(label=d["audio_label"]),
        gr.update(value=d["btn_analyze"]),
        gr.update(value=d["partner_header"]),
        gr.update(label=d["res_label"]),
        gr.update(label=d["why_label"]),
        gr.update(label=d["spec_label"]),
        gr.update(value=ad_html)
    ]

# -------------------------------------------------------------
# 5. UI LAYOUT & PARTNER LINKS
# -------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft(primary_hue="sky", neutral_hue="slate"), css=custom_css, title="StethoLogic") as demo:
    with gr.Row():
        with gr.Column(scale=8):
            header = gr.Markdown(f"{I18N['English']['title']}\n{I18N['English']['subtitle']}")
        with gr.Column(scale=4):
            lang_selector = gr.Dropdown(
                ["English", "हिन्दी (Hindi)", "தமிழ் (Tamil)", "తెలుగు (Telugu)"],
                value="English",
                label="🌐 Select Language / భాష / மொழி"
            )
    
    banner = gr.HTML(f'<div class="sponsor-banner">{I18N["English"]["banner"]}</div>')

    with gr.Tabs():
        with gr.TabItem("🎙️ StethoLogic Screener"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label=I18N["English"]["audio_label"])
                    analyze_btn = gr.Button(I18N["English"]["btn_analyze"], variant="primary")
                    gr.Markdown("---")
                    partner_title = gr.Markdown(I18N["English"]["partner_header"])
                    with gr.Row():
                        gr.HTML('<a href="https://www.1mg.com" target="_blank" style="text-decoration:none;"><button style="width:100%; padding:10px; border-radius:6px; background:#10b981; color:white; border:none; cursor:pointer; font-weight:bold;">💊 Order Prescriptions</button></a>')
                        gr.HTML('<a href="https://www.practo.com" target="_blank" style="text-decoration:none;"><button style="width:100%; padding:10px; border-radius:6px; background:#6366f1; color:white; border:none; cursor:pointer; font-weight:bold;">👨‍⚕️ Consult Pulmonologist</button></a>')

                with gr.Column():
                    result_box = gr.Textbox(label=I18N["English"]["res_label"], interactive=False)
                    why_box = gr.Textbox(label=I18N["English"]["why_label"], interactive=False, lines=3)
                    cta_box = gr.Markdown("")
                    confidence_labels = gr.Label(num_top_classes=4, label="Differential Probability Breakdown")
                    spec_output = gr.Image(label=I18N["English"]["spec_label"], type="filepath")

            with gr.Row():
                ad_box = gr.HTML(f'<div class="ad-box" style="width:100%; margin-top:15px;">{I18N["English"]["ad_box"]}</div>')

        with gr.TabItem("💬 Dr. Info"):
            gr.ChatInterface(
                fn=dynamic_chat,
                description="I am Dr. Info, your clinical triage companion. Ask about symptoms, lung sound characteristics, or recovery steps in English, Hindi, Tamil, Telugu, or Hinglish."
            )

    lang_selector.change(
        fn=change_language,
        inputs=[lang_selector],
        outputs=[header, banner, audio_input, analyze_btn, partner_title, result_box, why_box, spec_output, ad_box]
    )

    analyze_btn.click(
        fn=analyze_audio,
        inputs=[audio_input, lang_selector],
        outputs=[result_box, why_box, cta_box, spec_output, confidence_labels]
    )

port = int(os.environ.get("PORT", 7860))
demo.queue().launch(
    server_name="0.0.0.0",
    server_port=port,
    prevent_thread_lock=False,
    show_error=True
)
