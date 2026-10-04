import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from moviepy import VideoFileClip

# Import modul internal
from tahap2_preprocessing import VisualPreprocessor, AudioPreprocessor
from tahap3_architecture import EduInclusiveFusionModel

# =============================================================================
# CONFIG & PAGE SETUP
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "assets", "CogniSense_logo_transparent.png")
ICON_PATH = os.path.join(BASE_DIR, "assets", "CogniSense_icon_transparent.png")

st.set_page_config(
    page_title="CogniSense AI | Skrining Risiko Multimodal",
    page_icon=Image.open(ICON_PATH),
    layout="wide",
    initial_sidebar_state="expanded"
)

# Logo di sidebar (versi ikon saat sidebar dilipat)
try:
    st.logo(LOGO_PATH, icon_image=ICON_PATH, size="large")
except TypeError:
    st.logo(LOGO_PATH, icon_image=ICON_PATH)

# Custom Styling CSS untuk Tampilan Clean & Professional
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .recommendation-box {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 1rem;
        border-radius: 4px;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "./checkpoints/best_model.pth"
HISTORY_PATH = "./checkpoints/history.json"
LEVEL_NAMES = ["Tidak Berisiko", "Risiko Ringan", "Risiko Sedang", "Risiko Berat"]

RECOMMENDATIONS = {
    0: "Tidak ada indikasi risiko dominan pada sampel ini. Tetap pantau perkembangan belajar dan komunikasi anak secara berkala.",
    1: "Terdapat indikasi risiko awal pada sampel ini. Bila ada kekhawatiran terhadap perkembangan anak, diskusikan dengan orang tua, guru, atau tenaga profesional yang sesuai.",
    2: "Terdapat indikasi risiko yang perlu ditinjau lebih lanjut. Pertimbangkan konsultasi dengan tenaga profesional yang kompeten untuk asesmen menyeluruh.",
    3: "Terdapat indikasi risiko tinggi pada sampel ini. Hasil ini perlu dikonfirmasi melalui asesmen langsung oleh tenaga profesional yang kompeten."
}

# =============================================================================
# HELPER PROCESSOR UNTUK INPUT VIDEO / AUDIO
# =============================================================================
def process_audio_or_video_input(uploaded_file, temp_dir="./temp"):
    os.makedirs(temp_dir, exist_ok=True)
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    saved_input_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(saved_input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if file_ext in [".mp4", ".mov", ".avi"]:
        output_wav_path = os.path.join(temp_dir, "extracted_audio.wav")
        video_clip = VideoFileClip(saved_input_path)
        if video_clip.audio is None:
            raise ValueError("Berkas video tidak memiliki trek audio/suara.")
        video_clip.audio.write_audiofile(output_wav_path, logger=None)
        video_clip.close()
        return output_wav_path, True
    else:
        return saved_input_path, False

# =============================================================================
# GRAD-CAM CLASS FOR EXPLAINABLE AI (XAI)
# =============================================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, img_tensor: torch.Tensor, audio_tensor: torch.Tensor, target_class: int = None):
        self.model.eval()
        self.model.zero_grad()

        img_tensor.requires_grad_(True)
        outputs = self.model(img_tensor, audio_tensor)
        logits = outputs["dys_logits"]

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        score = logits[0, target_class]
        score.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(img_tensor.shape[2], img_tensor.shape[3]), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam, target_class, outputs


def overlay_heatmap(orig_img_pil: Image.Image, heatmap_np: np.ndarray, alpha: float = 0.5):
    orig_np = np.array(orig_img_pil.convert("RGB"))
    h, w, _ = orig_np.shape

    heatmap_resized = cv2.resize(heatmap_np, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(orig_np, 1.0 - alpha, heatmap_colored, alpha, 0)
    return overlay

# =============================================================================
# HELPER LOADER MODEL
# =============================================================================
@st.cache_resource
def load_trained_model():
    if not os.path.exists(CKPT_PATH):
        return None
    model = EduInclusiveFusionModel(pretrained_visual=False).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

# =============================================================================
# HEADER & DASHBOARD UI
# =============================================================================
st.image(LOGO_PATH, width=320)
st.markdown('<div class="sub-header">Sistem Skrining Multimodal Terpadu Disleksia (Visual) dan Speech Delay (Audio) Berbasis Explainable AI</div>', unsafe_allow_html=True)
st.warning(
    "**Batasan penggunaan:** hasil ini adalah indikasi risiko dari model penelitian, "
    "bukan diagnosis medis atau psikologis. Model dilatih menggunakan label proxy "
    "dan pasangan data sintetis; hasil perlu dikonfirmasi melalui asesmen profesional."
)

st.markdown("---")

# Sidebar Uploaders
st.sidebar.title("Panel Kontrol Input")
st.sidebar.subheader("1. Citra Tulisan Tangan")
uploaded_image = st.sidebar.file_uploader("Unggah berkas gambar", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

st.sidebar.subheader("2. Rekaman Suara / Video (.wav, .mp4, .mov)")
uploaded_audio = st.sidebar.file_uploader("Unggah berkas audio atau video", type=["wav", "mp4", "mov", "avi"], label_visibility="collapsed")

model = load_trained_model()

if model is None:
    st.error(f"Berkas checkpoint model tidak ditemukan pada jalur '{CKPT_PATH}'. Pastikan file 'best_model.pth' sudah diletakkan pada folder 'checkpoints/'.")
    st.stop()

# Tabs Interface
tab1, tab2 = st.tabs(["Skrining Risiko & Visualisasi XAI", "Performa & Evaluasi Model"])

with tab1:
    if uploaded_image is not None and uploaded_audio is not None:
        os.makedirs("./temp", exist_ok=True)
        temp_img_path = os.path.join("./temp", uploaded_image.name)

        with open(temp_img_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        st.info("Berkas gambar dan audio/video berhasil dimuat. Klik tombol di bawah untuk memulai proses inferensi.")
        
        if st.button("Jalankan Skrining Risiko", use_container_width=True, type="primary"):
            with st.spinner("Model sedang memproses analisis ekstraksi fitur multimodal..."):
                try:
                    temp_aud_path, is_video = process_audio_or_video_input(uploaded_audio)
                except Exception as e:
                    st.error(f"Gagal memproses berkas audio/video: {str(e)}")
                    st.stop()

                # Preprocessing
                v_prep = VisualPreprocessor(train=False)
                a_prep = AudioPreprocessor()

                raw_img = Image.open(temp_img_path)
                img_tensor = v_prep(raw_img).unsqueeze(0).to(DEVICE)
                audio_tensor = a_prep(temp_aud_path).unsqueeze(0).to(DEVICE)

                # Target Layer Auto-Detect
                conv_layers = [m for m in model.visual_branch.modules() if isinstance(m, nn.Conv2d)]
                target_layer = conv_layers[-1]
                grad_cam = GradCAM(model, target_layer)

                # Inference & Grad-CAM Heatmap
                heatmap, dys_pred, outputs = grad_cam.generate_heatmap(img_tensor, audio_tensor)
                
                dys_probs = F.softmax(outputs["dys_logits"], dim=1)[0].detach().cpu().numpy()
                speech_probs = F.softmax(outputs["speech_logits"], dim=1)[0].detach().cpu().numpy()
                speech_pred = int(speech_probs.argmax())

                overlay_img = overlay_heatmap(raw_img, heatmap, alpha=0.5)

            if is_video:
                st.info("Sistem mendeteksi input berkas video dan berhasil mengekstrak trek audio secara otomatis untuk analisis Speech Delay.")

            st.subheader("Ringkasan Indikasi Risiko")
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                st.metric(
                    label="Tingkat Risiko Disleksia (Visual)", 
                    value=f"Level {dys_pred} - {LEVEL_NAMES[dys_pred]}", 
                    delta=f"Tingkat Keyakinan: {dys_probs[dys_pred]*100:.2f}%"
                )
            with col_m2:
                st.metric(
                    label="Tingkat Risiko Speech Delay (Audio)", 
                    value=f"Level {speech_pred} - {LEVEL_NAMES[speech_pred]}", 
                    delta=f"Tingkat Keyakinan: {speech_probs[speech_pred]*100:.2f}%"
                )

            # Box Rekomendasi
            max_risk_level = max(dys_pred, speech_pred)
            st.markdown(f"""
                <div class="recommendation-box">
                    <strong>Tindak lanjut yang disarankan:</strong><br>
                    {RECOMMENDATIONS[max_risk_level]}
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Visualisasi Explainable AI (XAI)")

            # Visualisation Plot Grid
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))

            # Panel 1: Original Image
            axes[0, 0].imshow(raw_img)
            axes[0, 0].set_title("1. Citra Input Tulisan Tangan", fontsize=11, fontweight="bold")
            axes[0, 0].axis("off")

            # Panel 2: Grad-CAM Overlay
            axes[0, 1].imshow(overlay_img)
            axes[0, 1].set_title(f"2. Grad-CAM Visual (Level Risiko {dys_pred})", fontsize=11, fontweight="bold")
            axes[0, 1].axis("off")

            # Panel 3: Mel-Spectrogram
            mel_spec = audio_tensor.squeeze().cpu().numpy()
            im = axes[1, 0].imshow(mel_spec, aspect="auto", origin="lower", cmap="magma")
            axes[1, 0].set_title(f"3. Feature Audio Mel-Spectrogram (Level {speech_pred})", fontsize=11, fontweight="bold")
            axes[1, 0].set_xlabel("Time Frames")
            axes[1, 0].set_ylabel("Mel Frequency Bins")

            # Panel 4: Confidence Chart
            x = np.arange(4)
            width = 0.35
            axes[1, 1].bar(x - width/2, dys_probs * 100, width, label='Disleksia (Visual)', color='#1E3A8A')
            axes[1, 1].bar(x + width/2, speech_probs * 100, width, label='Speech Delay (Audio)', color='#D97706')
            axes[1, 1].set_title("4. Distribusi Probabilitas Prediksi (%)", fontsize=11, fontweight="bold")
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(["L0\nTidak", "L1\nRingan", "L2\nSedang", "L3\nBerat"])
            axes[1, 1].set_ylim(0, 100)
            axes[1, 1].legend()
            axes[1, 1].grid(axis="y", linestyle="--", alpha=0.5)

            plt.tight_layout()
            st.pyplot(fig)

    else:
        st.info("Silakan unggah berkas gambar tulisan tangan dan berkas rekaman suara (.wav) atau video (.mp4/.mov) melalui panel kontrol di sebelah kiri.")

with tab2:
    st.subheader("Grafik Evaluasi Pembelajaran Model")
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            history = json.load(f)

        if isinstance(history, list):
            epochs = [item.get("epoch", i + 1) for i, item in enumerate(history)]
            train_loss = [item.get("train", {}).get("loss_total", item.get("train_loss", 0)) for item in history]
            val_loss = [item.get("val", {}).get("loss_total", item.get("val_loss", 0)) for item in history]
            val_f1 = [item.get("val", {}).get("f1_macro_avg", item.get("val_f1_avg", item.get("val_f1", 0))) for item in history]
        elif isinstance(history, dict):
            train_loss = history.get("train_loss", [])
            val_loss = history.get("val_loss", [])
            val_f1 = history.get("val_f1_avg", history.get("val_f1", []))
            epochs = list(range(1, len(train_loss) + 1))
        else:
            st.error("Format berkas history.json tidak dikenali.")
            st.stop()

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            fig_loss, ax_loss = plt.subplots(figsize=(6, 4))
            ax_loss.plot(epochs, train_loss, label="Train Loss", color="#1E3A8A", marker="o")
            ax_loss.plot(epochs, val_loss, label="Validation Loss", color="#DC2626", marker="o")
            ax_loss.set_title("Learning Curve: Loss per Epoch", fontweight="bold")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.legend()
            ax_loss.grid(True, linestyle="--", alpha=0.6)
            st.pyplot(fig_loss)

        with col_g2:
            fig_f1, ax_f1 = plt.subplots(figsize=(6, 4))
            ax_f1.plot(epochs, val_f1, label="Validation F1-Score Average", color="#16A34A", marker="s")
            ax_f1.set_title("Performance Curve: Validation F1-Score", fontweight="bold")
            ax_f1.set_xlabel("Epoch")
            ax_f1.set_ylabel("F1 Score")
            ax_f1.legend()
            ax_f1.grid(True, linestyle="--", alpha=0.6)
            st.pyplot(fig_f1)
            
        if val_f1:
            best_index = int(np.argmax(val_f1))
            st.success(
                f"Macro F1 validasi terbaik tercatat pada Epoch {epochs[best_index]} "
                f"dengan nilai {val_f1[best_index]:.4f} ({val_f1[best_index] * 100:.2f}%)."
            )
    else:
        st.warning(f"Berkas 'history.json' tidak ditemukan pada lokasi '{HISTORY_PATH}'.")