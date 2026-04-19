import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import urllib.request

# --- 1. UI CONFIG ---
st.set_page_config(page_title="NeuralNet Protocol", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# --- 2. DARK CYBERPUNK CSS ---
st.markdown("""
    <style>
    /* Dark Mode Background */
    .stApp { background-color: #0e1117; color: #e6edf3; }
    
    /* Cyber Header */
    .cyber-box {
        background-color: #161b22;
        padding: 25px;
        border: 1px solid #30363d;
        border-top: 3px solid #00ffcc; /* Neon Cyan */
        border-radius: 8px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .cyber-box h1 { color: #00ffcc; font-family: 'Courier New', monospace; margin: 0; font-weight: 900;}
    .cyber-box p { color: #8b949e; font-size: 1rem; margin-top: 5px; }

    /* Glowing Result Panels */
    .panel-dog { background: #161b22; border: 1px solid #39d353; padding: 25px; border-radius: 8px; text-align: center; box-shadow: 0 0 15px rgba(57, 211, 83, 0.2);}
    .panel-cat { background: #161b22; border: 1px solid #d2a8ff; padding: 25px; border-radius: 8px; text-align: center; box-shadow: 0 0 15px rgba(210, 168, 255, 0.2);}
    .panel-none { background: #161b22; border: 1px solid #ff7b72; padding: 25px; border-radius: 8px; text-align: center; box-shadow: 0 0 15px rgba(255, 123, 114, 0.2);}
    
    .lbl-title { font-size: 2rem; color: #ffffff; font-weight: bold; font-family: 'Courier New', monospace; letter-spacing: 2px;}
    .lbl-text { font-size: 1.2rem; color: #c9d1d9; margin-top: 10px;}
    .highlight { color: #00ffcc; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- 3. CORE LOGIC (UNCHANGED) ---
@st.cache_resource
def load_ai_engine():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        urllib.request.urlretrieve(url, "imagenet_classes.txt")
        with open("imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]
    except Exception:
        categories = [f"Object {i}" for i in range(1000)]

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, categories, transform

# --- 4. HENRY'S DASHBOARD ---
st.markdown("""
<div class="cyber-box">
    <h1>[:: Image Classifier ::]</h1>
    <p>SYSTEM ARCHITECT: OTUONYE TOBECHUKWU HENRY | REG: 20231371822 | MODULE: CSC 309</p>
</div>
""", unsafe_allow_html=True)

model, categories, transform = load_ai_engine()

# --- NEW SIDEBAR LAYOUT ---
with st.sidebar:
    st.markdown("### 🎛️ SYSTEM CONTROLS")
    st.caption("Upload image data for processing.")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    st.divider()
    if uploaded_file is not None:
        run_scan = st.button(">> EXECUTE SCAN", use_container_width=True)
    else:
        run_scan = False

# Main Display Area
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    
    # Put the image in the center, not taking up the whole screen
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        st.image(img, caption="TARGET ACQUIRED", use_container_width=True)
        
    if run_scan:
        with st.spinner("Compiling Neural Tensors..."):
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                output = model(tensor)

            probs = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_id = torch.topk(probs, 1)
            cid = top_id.item()
            conf = top_prob.item() * 100
            label = categories[cid].replace('_', ' ').title().upper()

            st.markdown("<br>", unsafe_allow_html=True)
            
            if 151 <= cid <= 268: # Dog
                st.markdown(f"""
                <div class="panel-dog">
                    <div class="lbl-title">TARGET MATCH: DOG</div>
                    <div class="lbl-text">SUB-SPECIES: <span class="highlight">{label}</span> | PROBABILITY: <span class="highlight">{conf:.1f}%</span></div>
                </div>
                """, unsafe_allow_html=True)
            elif 281 <= cid <= 285: # Cat
                st.markdown(f"""
                <div class="panel-cat">
                    <div class="lbl-title">TARGET MATCH: CAT</div>
                    <div class="lbl-text">SUB-SPECIES: <span class="highlight">{label}</span> | PROBABILITY: <span class="highlight">{conf:.1f}%</span></div>
                </div>
                """, unsafe_allow_html=True)
            else: # Unknown
                st.markdown(f"""
                <div class="panel-none">
                    <div class="lbl-title">ANOMALY DETECTED</div>
                    <div class="lbl-text">PREDICTION: <span class="highlight">{label}</span> | PROBABILITY: <span class="highlight">{conf:.1f}%</span></div>
                </div>
                """, unsafe_allow_html=True)
else:
    st.info("Awaiting input data. Please upload a file via the side panel.")