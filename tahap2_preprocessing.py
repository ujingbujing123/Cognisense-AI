"""
=============================================================================
TAHAP 2
Preprocessing & Augmentasi Detail (Visual + Audio)

USB 2026 - Lomba Implementasi Data Mining
=============================================================================

File ini MENGGANTIKAN placeholder transform di Tahap 1 dengan pipeline
preprocessing lengkap sesuai spesifikasi proyek:

VISUAL:
  - Resize 224x224
  - Grayscale (sudah dilakukan sejak load)
  - Augmentasi (Train saja): Random Rotation +-10 derajat, Random Shift/Translation
  - Adaptive Thresholding (cv2.adaptiveThreshold) — dilakukan SETELAH augmentasi
    geometris, supaya efek threshold tetap konsisten pada tulisan tangan yang
    sudah dirotasi/digeser.

AUDIO:
  - Noise reduction (spectral gating sederhana, noise profile dari ~0.3 detik
    pertama sinyal — pendekatan ringan tanpa dependency eksternal seperti
    `noisereduce`, supaya tetap 100% offline & ringan)
  - Trim/Padding ke durasi konsisten 5 detik
  - Konversi ke Mel-Spectrogram (dB scale, dinormalisasi ke [0,1])

Dependencies tambahan dari Tahap 1:
    pip install opencv-python librosa torchvision soundfile --break-system-packages

Cara pakai:
    from tahap2_preprocessing_dataset import EduInclusiveFusionDataset
    train_ds = EduInclusiveFusionDataset("data/pairing_output/pairing_train.csv", train=True)
    test_ds  = EduInclusiveFusionDataset("data/pairing_output/pairing_test.csv", train=False)
"""

import os
import csv
from typing import Tuple

import numpy as np

try:
    import cv2
except ImportError:
    raise ImportError("OpenCV belum terinstall. Jalankan: pip install opencv-python --break-system-packages")

try:
    import librosa
except ImportError:
    raise ImportError("Librosa belum terinstall. Jalankan: pip install librosa soundfile --break-system-packages")

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
except ImportError:
    raise ImportError(
        "PyTorch/torchvision belum terinstall. "
        "Jalankan: pip install torch torchvision --break-system-packages"
    )

from PIL import Image


# =============================================================================
# 1. CONFIG PREPROCESSING
# =============================================================================

IMG_CONFIG = {
    "target_size": (224, 224),   # (width, height)
    "aug_rotation_deg": 10,      # Random Rotation +-10 derajat
    "aug_translate_frac": 0.1,   # Random Shift/Translation max 10% lebar/tinggi
    "adaptive_block_size": 11,   # ukuran blok tetangga utk adaptiveThreshold (ganjil)
    "adaptive_C": 2,             # konstanta pengurang mean/gaussian weighted sum
}

AUDIO_CONFIG = {
    "sample_rate": 16000,        # Hz, umum untuk speech (Wav2Vec2 pretrained pakai 16kHz)
    "duration_sec": 5.0,         # durasi konsisten sesuai spesifikasi
    "n_fft": 1024,
    "hop_length": 256,
    "n_mels": 128,
    "noise_profile_sec": 0.3,    # asumsi 0.3 detik pertama = noise floor
}


# =============================================================================
# 2. VISUAL PREPROCESSOR
# =============================================================================

class VisualPreprocessor:
    """
    Pipeline: Resize -> (Train only) Random Rotation & Shift -> Adaptive
    Thresholding -> Normalisasi ke tensor [0,1], shape (1, H, W).
    """

    def __init__(self, train: bool = True):
        self.train = train
        aug_ops = []
        if train:
            aug_ops.append(transforms.RandomRotation(degrees=IMG_CONFIG["aug_rotation_deg"]))
            aug_ops.append(transforms.RandomAffine(
                degrees=0,
                translate=(IMG_CONFIG["aug_translate_frac"], IMG_CONFIG["aug_translate_frac"]),
            ))
        self.augment = transforms.Compose(aug_ops) if aug_ops else None

    def _adaptive_threshold(self, img_uint8: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(
            img_uint8,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            IMG_CONFIG["adaptive_block_size"],
            IMG_CONFIG["adaptive_C"],
        )

    def __call__(self, image: Image.Image) -> "torch.Tensor":
        # 1. Grayscale + Resize
        image = image.convert("L").resize(IMG_CONFIG["target_size"])

        # 2. Augmentasi geometris (hanya saat train)
        if self.augment is not None:
            image = self.augment(image)

        # 3. Adaptive Thresholding (butuh uint8 numpy array)
        arr_uint8 = np.array(image, dtype=np.uint8)
        arr_thresh = self._adaptive_threshold(arr_uint8)

        # 4. Normalisasi -> tensor (1, H, W)
        arr_norm = arr_thresh.astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr_norm).unsqueeze(0)
        return tensor


# =============================================================================
# 3. AUDIO PREPROCESSOR (Dengan RAM Caching & Safety Fallback)
# =============================================================================

class AudioPreprocessor:
    """
    Pipeline: Check Cache -> Load (16kHz) -> Noise Reduction (spectral gating)
    -> Trim/Pad ke 5s -> Mel-Spectrogram -> Normalisasi -> Store to RAM Cache.
    """

    def __init__(self):
        self.sr = AUDIO_CONFIG["sample_rate"]
        self.target_len = int(self.sr * AUDIO_CONFIG["duration_sec"])
        self.expected_frames = 1 + self.target_len // AUDIO_CONFIG["hop_length"]
        self.cache = {}  # ⚡ RAM Cache (Menyimpan spektrogram di System RAM Colab)

    def _reduce_noise(self, y: np.ndarray) -> np.ndarray:
        noise_len = int(AUDIO_CONFIG["noise_profile_sec"] * self.sr)
        if len(y) <= noise_len:
            return y

        stft_full = librosa.stft(y, n_fft=AUDIO_CONFIG["n_fft"], hop_length=AUDIO_CONFIG["hop_length"])
        stft_noise = librosa.stft(y[:noise_len], n_fft=AUDIO_CONFIG["n_fft"],
                                   hop_length=AUDIO_CONFIG["hop_length"])

        noise_mag_profile = np.mean(np.abs(stft_noise), axis=1, keepdims=True)
        magnitude, phase = librosa.magphase(stft_full)
        magnitude_denoised = np.maximum(magnitude - noise_mag_profile, 0.0)

        stft_denoised = magnitude_denoised * phase
        y_denoised = librosa.istft(stft_denoised, hop_length=AUDIO_CONFIG["hop_length"], length=len(y))
        return y_denoised

    def _trim_or_pad(self, y: np.ndarray) -> np.ndarray:
        if len(y) > self.target_len:
            return y[: self.target_len]
        if len(y) < self.target_len:
            return np.pad(y, (0, self.target_len - len(y)), mode="constant")
        return y

    def _to_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr,
            n_fft=AUDIO_CONFIG["n_fft"],
            hop_length=AUDIO_CONFIG["hop_length"],
            n_mels=AUDIO_CONFIG["n_mels"],
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        denom = (mel_db.max() - mel_db.min()) + 1e-8
        mel_norm = (mel_db - mel_db.min()) / denom
        return mel_norm.astype(np.float32)

    def __call__(self, audio_path: str) -> "torch.Tensor":
        # 1. CEK CACHE: Jika spektrogram sudah ada di RAM, langsung return!
        if audio_path in self.cache:
            return self.cache[audio_path]

        # 2. JIKA BELUM ADA: Process audio dari disk
        try:
            y, _ = librosa.load(audio_path, sr=self.sr, mono=True)
            y = self._reduce_noise(y)
            y = self._trim_or_pad(y)
            mel_norm = self._to_mel_spectrogram(y)
            tensor = torch.from_numpy(mel_norm).unsqueeze(0)  # (1, n_mels, time_steps)
        except Exception as e:
            if audio_path not in _CORRUPTED_AUDIO_LOG:
                _CORRUPTED_AUDIO_LOG.add(audio_path)
                print(f"[WARNING] Audio tidak terbaca, dipakai fallback silence: {audio_path} ({e})")
            silence = np.zeros((AUDIO_CONFIG["n_mels"], self.expected_frames), dtype=np.float32)
            tensor = torch.from_numpy(silence).unsqueeze(0)

        # 3. SIMPAN KE CACHE RAM
        self.cache[audio_path] = tensor
        return tensor


# =============================================================================
# 4. UPDATED DATASET (menggantikan versi placeholder Tahap 1)
# =============================================================================

class EduInclusiveFusionDataset(Dataset):
    """
    Dataset multi-modal final: membaca hasil synthetic pairing (Tahap 1) dan
    menerapkan preprocessing lengkap (Tahap 2).

    Parameter:
        pairing_csv (str): path ke pairing_train.csv / pairing_test.csv
        train (bool): True -> aktifkan augmentasi visual (rotasi/shift).
                      False -> tanpa augmentasi (dipakai untuk data Test/Val).
    """

    def __init__(self, pairing_csv: str, train: bool = True):
        self.records = self._load_csv(pairing_csv)
        self.train = train
        self.visual_transform = VisualPreprocessor(train=train)
        self.audio_transform = AudioPreprocessor()

    @staticmethod
    def _load_csv(pairing_csv: str):
        records = []
        with open(pairing_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["final_label"] = int(row["final_label"])
                row["dys_level"] = int(row["dys_level"])
                row["speech_level"] = int(row["speech_level"])
                records.append(row)
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        row = self.records[idx]

        try:
            image = Image.open(row["image_path"])
            image.load()  # paksa baca isi file sekarang, supaya error ketahuan di sini
            image_tensor = self.visual_transform(image)
        except Exception as e:
            if row["image_path"] not in _CORRUPTED_IMAGE_LOG:
                _CORRUPTED_IMAGE_LOG.add(row["image_path"])
                print(f"[WARNING] Gambar tidak terbaca, dipakai fallback putih polos: "
                      f"{row['image_path']} ({e})")
            image_tensor = torch.ones(1, *IMG_CONFIG["target_size"])

        audio_tensor = self.audio_transform(row["audio_path"])

        # Target: [level_disleksia, level_speech_delay], masing-masing 0..3
        # (0 Tidak Berisiko, 1 Ringan, 2 Sedang, 3 Berat)
        targets = torch.tensor([row["dys_level"], row["speech_level"]], dtype=torch.long)
        return image_tensor, audio_tensor, targets

# =============================================================================
# 5. SANITY CHECK — jalankan langsung untuk verifikasi shape & nilai tensor
# =============================================================================

def main():
    pairing_csv = os.path.join("data", "pairing_output", "pairing_train.csv")

    if not os.path.exists(pairing_csv):
        print(f"[INFO] {pairing_csv} belum ada. Jalankan tahap1_pairing_dataset.py "
              f"terlebih dahulu untuk generate pairing CSV.")
        return

    ds = EduInclusiveFusionDataset(pairing_csv, train=True)
    print(f"Jumlah sampel di dataset: {len(ds)}")

    img_tensor, audio_tensor, target = ds[0]
    print(f"Contoh sampel index 0:")
    print(f"  image_tensor shape : {tuple(img_tensor.shape)}  "
          f"(min={img_tensor.min():.3f}, max={img_tensor.max():.3f})")
    print(f"  audio_tensor shape : {tuple(audio_tensor.shape)}  "
          f"(min={audio_tensor.min():.3f}, max={audio_tensor.max():.3f})")
    names = ["Tidak Berisiko", "Ringan", "Sedang", "Berat"]
    print(f"  level disleksia    : {int(target[0])} ({names[int(target[0])]})")
    print(f"  level speech delay : {int(target[1])} ({names[int(target[1])]})")


if __name__ == "__main__":
    main()