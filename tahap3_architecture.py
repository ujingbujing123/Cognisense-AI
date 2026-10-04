"""
=============================================================================
EduInclusive-Fusion — TAHAP 3
Dual-Branch Multi-Modal Architecture (PyTorch)

USB 2026 - Lomba Implementasi Data Mining
=============================================================================

Menerima keluaran Dataset Tahap 2:
    image_tensor : (B, 1, 224, 224)   grayscale, adaptive-thresholded
    audio_tensor : (B, 1, n_mels, T)  Mel-Spectrogram (dB, dinormalisasi)
    target       : (B, 2)             [level_disleksia, level_speech_delay], masing 0-3

Arsitektur:
    Visual Branch : ResNet50 pretrained (ImageNet) -> vektor fitur 512
        - Grayscale (1 channel) direplikasi jadi 3 channel di forward() supaya
          bisa pakai bobot pretrained ImageNet apa adanya (tanpa modif conv1).
    Audio Branch  : 2D-CNN pada Mel-Spectrogram -> vektor fitur 512
        - Dipilih 2D-CNN (bukan Wav2Vec2) karena selaras langsung dengan
          keluaran Tahap 2 (Mel-Spectrogram, bukan raw waveform) dan jauh
          lebih ringan untuk dilatih dari nol / fine-tune di laptop biasa.
          Wav2Vec2 tetap bisa dipasang sebagai alternatif (lihat catatan di
          bagian akhir file), tapi butuh raw waveform + download model besar.
    Fusion        : Concatenate (512+512=1024) -> Dense+ReLU+Dropout(0.3)
                     -> DUA kepala output terpisah, masing-masing Linear(*,4)
                        untuk 4 tingkat (Tidak Berisiko/Ringan/Sedang/Berat).

Loss: jumlah 2 CrossEntropyLoss (satu per kepala). Opsional: beri bobot kelas
(class_weight) karena kelas 0 (Tidak Berisiko) selalu 50% dari data (lihat
distribusi Tahap 1) -> gunakan `compute_class_weights()` di bagian bawah.

Dependencies tambahan: pip install torch torchvision --break-system-packages
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# =============================================================================
# 1. VISUAL BRANCH — ResNet50 pretrained -> vektor 512
# =============================================================================

class VisualBranch(nn.Module):
    def __init__(self, feature_dim: int = 512, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        # Buang fully-connected asli (1000 kelas ImageNet), pakai sebagai ekstraktor fitur
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # output (B, 2048, 1, 1)
        self.project = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Normalisasi ImageNet (dipakai setelah replikasi ke 3 channel)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W) hasil adaptive thresholding, nilai [0,1]
        x = x.repeat(1, 3, 1, 1)                # replikasi grayscale -> 3 channel
        x = (x - self.mean) / self.std          # normalisasi gaya ImageNet
        feat = self.backbone(x)                 # (B, 2048, 1, 1)
        feat = torch.flatten(feat, 1)            # (B, 2048)
        return self.project(feat)                # (B, feature_dim)


# =============================================================================
# 2. AUDIO BRANCH — 2D-CNN pada Mel-Spectrogram -> vektor 512
# =============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AudioBranch(nn.Module):
    """2D-CNN ringan pada Mel-Spectrogram. Memakai AdaptiveAvgPool2d di akhir
    supaya tidak bergantung pada panjang waktu (T) yang presisi — aman kalau
    durasi audio sedikit berbeda dari 5 detik saat inferensi."""

    def __init__(self, feature_dim: int = 512, in_mels: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # (B, 256, 4, 4) apapun ukuran input
        self.project = nn.Sequential(
            nn.Linear(256 * 4 * 4, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T), nilai [0,1] (sudah dinormalisasi di Tahap 2)
        feat = self.conv(x)
        feat = self.pool(feat)
        feat = torch.flatten(feat, 1)
        return self.project(feat)


# =============================================================================
# 3. FUSION LAYER + DUA KEPALA OUTPUT ORDINAL
# =============================================================================

class EduInclusiveFusionModel(nn.Module):
    """
    Dual-branch late fusion. Mengembalikan dict:
        {"dys_logits": (B,4), "speech_logits": (B,4),
         "visual_feat": (B,512), "audio_feat": (B,512), "fused_feat": (B,fusion_hidden)}

    visual_feat/audio_feat/fused_feat disertakan supaya mudah dipakai di Tahap 5
    (Grad-CAM butuh feature map sebelum pooling -> lihat get_visual_target_layer()).
    """

    def __init__(self,
                 feature_dim: int = 512,
                 fusion_hidden: int = 256,
                 n_levels: int = 4,
                 pretrained_visual: bool = True,
                 freeze_visual_backbone: bool = False):
        super().__init__()
        self.visual_branch = VisualBranch(feature_dim, pretrained_visual, freeze_visual_backbone)
        self.audio_branch = AudioBranch(feature_dim)

        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.dys_head = nn.Linear(fusion_hidden, n_levels)
        self.speech_head = nn.Linear(fusion_hidden, n_levels)

    def forward(self, image: torch.Tensor, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        v_feat = self.visual_branch(image)
        a_feat = self.audio_branch(audio)
        fused_in = torch.cat([v_feat, a_feat], dim=1)   # (B, 1024)
        fused = self.fusion(fused_in)                    # (B, fusion_hidden)
        return {
            "dys_logits": self.dys_head(fused),
            "speech_logits": self.speech_head(fused),
            "visual_feat": v_feat,
            "audio_feat": a_feat,
            "fused_feat": fused,
        }

    def get_visual_target_layer(self) -> nn.Module:
        """Layer conv terakhir ResNet50 (layer4) — target Grad-CAM standar di Tahap 5."""
        return self.visual_branch.backbone[-2]  # urutan children: ...,layer3,layer4,avgpool -> [-2]=layer4


# =============================================================================
# 4. LOSS GABUNGAN (2 KEPALA) + CLASS WEIGHT OPSIONAL
# =============================================================================

def compute_class_weights(level_counts: Dict[int, int], n_levels: int = 4) -> torch.Tensor:
    """Hitung bobot kelas = total / (n_levels * count), untuk menangani kelas 0
    (Tidak Berisiko) yang selalu 50% dari data (lihat distribusi Tahap 1)."""
    total = sum(level_counts.values())
    weights = [total / (n_levels * level_counts.get(k, 1)) for k in range(n_levels)]
    return torch.tensor(weights, dtype=torch.float32)


class DualHeadLoss(nn.Module):
    def __init__(self, dys_weight: torch.Tensor = None, speech_weight: torch.Tensor = None,
                 dys_loss_weight: float = 1.0, speech_loss_weight: float = 1.0):
        super().__init__()
        self.dys_ce = nn.CrossEntropyLoss(weight=dys_weight)
        self.speech_ce = nn.CrossEntropyLoss(weight=speech_weight)
        self.dys_loss_weight = dys_loss_weight
        self.speech_loss_weight = speech_loss_weight

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # targets: (B, 2) -> kolom 0 = dys_level, kolom 1 = speech_level
        loss_dys = self.dys_ce(outputs["dys_logits"], targets[:, 0])
        loss_speech = self.speech_ce(outputs["speech_logits"], targets[:, 1])
        total = self.dys_loss_weight * loss_dys + self.speech_loss_weight * loss_speech
        logs = {"loss_total": total.item(), "loss_dys": loss_dys.item(), "loss_speech": loss_speech.item()}
        return total, logs


# =============================================================================
# 5. SANITY CHECK — forward pass dengan tensor dummy sesuai shape Tahap 2
# =============================================================================

def main():
    torch.manual_seed(42)
    batch_size = 4
    n_mels = 128
    # T dihitung dari Tahap 2: durasi 5s, sr=16000, hop_length=256
    sr, duration, hop = 16000, 5.0, 256
    n_frames = int(sr * duration) // hop + 1

    dummy_image = torch.rand(batch_size, 1, 224, 224)
    dummy_audio = torch.rand(batch_size, 1, n_mels, n_frames)
    dummy_targets = torch.randint(0, 4, (batch_size, 2))

    print("Membangun model (mengunduh bobot ResNet50 pretrained jika belum ada)...")
    model = EduInclusiveFusionModel(pretrained_visual=True)
    model.eval()

    with torch.no_grad():
        out = model(dummy_image, dummy_audio)

    print(f"dys_logits    shape: {tuple(out['dys_logits'].shape)}")
    print(f"speech_logits shape: {tuple(out['speech_logits'].shape)}")
    print(f"visual_feat   shape: {tuple(out['visual_feat'].shape)}")
    print(f"audio_feat    shape: {tuple(out['audio_feat'].shape)}")
    print(f"fused_feat    shape: {tuple(out['fused_feat'].shape)}")

    criterion = DualHeadLoss()
    loss, logs = criterion(out, dummy_targets)
    print(f"\nContoh loss (bobot acak, target acak): {logs}")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameter     : {n_params:,}")
    print(f"Parameter dilatih   : {n_trainable:,}")


if __name__ == "__main__":
    main()


# =============================================================================
# CATATAN: ALTERNATIF WAV2VEC2 (opsional, tidak dipakai secara default)
# =============================================================================
# Kalau kamu tetap ingin memakai Wav2Vec2 sesuai opsi di spesifikasi lomba,
# konsekuensinya:
#   1. Input audio harus RAW WAVEFORM (bukan Mel-Spectrogram) -> Tahap 2 perlu
#      cabang preprocessing audio kedua yang mengembalikan waveform 1D.
#   2. Perlu `transformers` (HuggingFace) + download bobot pretrained
#      (~360MB, wav2vec2-base) yang JAUH lebih berat untuk fine-tune penuh
#      dibanding 2D-CNN kecil ini.
#   3. Cocok disebut di proposal sebagai "opsi lanjutan/future work" bila juri
#      menanyakan kenapa tidak pakai Wav2Vec2 — alasannya efisiensi komputasi
#      untuk deployment offline di Streamlit lokal (sesuai poin 4 spesifikasi).