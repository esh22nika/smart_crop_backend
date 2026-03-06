"""
Soil Classifier — wraps the local Pixsoil SavedModel (11 classes).
Falls back to a colour-heuristic when the model isn't available.

FIX: Added robust image validation that doesn't rely on MIME type headers,
     since Android/iOS devices often send 'application/octet-stream' for
     camera captures.

Model architecture (confirmed from keras_metadata.pb):
  Input  : 256×256×3  float32  [0, 1]   (key: 'conv2d_62_input')
  Layers : 4× Conv2D(29 filters) + MaxPool → Flatten → Dense(29,29,29,11) → Softmax
  Output : 11 classes (softmax probabilities)

Pixsoil label order — from the JS export (DO NOT REORDER):
  0: alike       ← ambiguous / mixed signal
  1: clay
  2: dry rocky
  3: grassy      ← vegetated surface
  4: gravel
  5: humus
  6: loam
  7: not         ← non-soil image
  8: sandy
  9: silty
 10: yellow      ← laterite / iron-rich (Deccan / Konkan context)
"""

import io
import os
import numpy as np
from PIL import Image, UnidentifiedImageError

# ── 11 Pixsoil class labels (JS export order — must not change) ───────
SOIL_LABELS: list[str] = [
    "alike",      # 0
    "clay",       # 1
    "dry rocky",  # 2
    "grassy",     # 3
    "gravel",     # 4
    "humus",      # 5
    "loam",       # 6
    "not",        # 7
    "sandy",      # 8
    "silty",      # 9
    "yellow",     # 10
]

_INVALID_LABELS: set[str] = {"alike", "grassy", "not"}

SOIL_PROFILES: dict[str, dict] = {
    "clay": dict(
        display_name="Clay Soil",
        N=(50, 80), P=(35, 65), K=(70, 110), pH=(5.5, 7.5),
        crops=["Rice", "Jute", "Sugarcane", "Banana", "Taro"],
        description=(
            "Heavy texture, slow drainage, high water-holding capacity. "
            "Swells when wet. Good for water-intensive Kharif crops."
        ),
    ),
    "dry rocky": dict(
        display_name="Dry Rocky / Skeletal Soil",
        N=(10, 25), P=(5, 18), K=(20, 45), pH=(6.5, 8.0),
        crops=["Millet", "Sorghum", "Groundnut", "Castor", "Pulses"],
        description=(
            "Shallow depth over rock, poor moisture retention. "
            "Suited only to drought-tolerant crops with minimal inputs."
        ),
    ),
    "gravel": dict(
        display_name="Gravelly / Coarse Soil",
        N=(12, 28), P=(6, 20), K=(22, 50), pH=(6.5, 8.0),
        crops=["Millet", "Sorghum", "Groundnut", "Castor"],
        description=(
            "Coarse texture, very fast drainage, low fertility. "
            "Requires organic amendment and mulching."
        ),
    ),
    "humus": dict(
        display_name="Humus / Organic-Rich Soil",
        N=(100, 140), P=(60, 90), K=(100, 150), pH=(6.0, 7.5),
        crops=["Vegetables", "Maize", "Rice", "Groundnut", "Banana"],
        description=(
            "Dark, organic-rich, high CEC and biological activity. "
            "Very fertile; found near forest fringes and wetlands in Konkan."
        ),
    ),
    "loam": dict(
        display_name="Loamy Soil",
        N=(70, 110), P=(45, 75), K=(85, 130), pH=(6.0, 7.5),
        crops=["Wheat", "Maize", "Soybean", "Cotton", "Vegetables"],
        description=(
            "Ideal sand-silt-clay balance. Best water retention and drainage. "
            "Most versatile soil for Maharashtra crops."
        ),
    ),
    "sandy": dict(
        display_name="Sandy Soil",
        N=(15, 30), P=(8, 20), K=(25, 55), pH=(6.0, 7.5),
        crops=["Millet", "Groundnut", "Watermelon", "Muskmelon", "Castor"],
        description=(
            "Low water and nutrient retention, fast drainage. "
            "Requires heavy organic amendments and frequent irrigation."
        ),
    ),
    "silty": dict(
        display_name="Silty / Alluvial Soil",
        N=(80, 120), P=(40, 70), K=(90, 140), pH=(6.5, 8.0),
        crops=["Rice", "Wheat", "Sugarcane", "Maize", "Banana"],
        description=(
            "Fine particle size, excellent fertility, found along river plains "
            "(Krishna, Godavari basins). High natural N and K."
        ),
    ),
    "yellow": dict(
        display_name="Laterite / Yellow-Red Soil",
        N=(20, 42), P=(10, 26), K=(32, 62), pH=(5.0, 6.5),
        crops=["Cashew", "Coconut", "Rice", "Groundnut", "Mango"],
        description=(
            "Iron and aluminium-rich, heavily leached, acidic. "
            "Common in Konkan belt (Ratnagiri, Sindhudurg, Raigad). "
            "Needs lime and organic matter for most crops."
        ),
    ),
    "_unknown": dict(
        display_name="Mixed / Unknown Soil",
        N=(40, 70), P=(25, 50), K=(55, 95), pH=(6.0, 7.5),
        crops=["Sorghum", "Millet", "Groundnut", "Soybean"],
        description="Soil type could not be determined — values are average estimates.",
    ),
}

SHORT_NAME: dict[str, str] = {
    "clay"       : "Clay",
    "dry rocky"  : "Dry Rocky",
    "gravel"     : "Gravelly",
    "humus"      : "Humus",
    "loam"       : "Loamy",
    "sandy"      : "Sandy",
    "silty"      : "Silty/Alluvial",
    "yellow"     : "Laterite",
    "alike"      : "Mixed",
    "grassy"     : "Vegetated Surface",
    "not"        : "Non-soil Image",
}

_VALID_INDICES: list[int] = [
    i for i, lbl in enumerate(SOIL_LABELS) if lbl not in _INVALID_LABELS
]


def _validate_and_open_image(image_bytes: bytes) -> Image.Image:
    """
    Opens and validates image bytes using Pillow.
    Raises ValueError with a user-friendly message if invalid.
    This is the single source of truth for image validation —
    we do NOT rely on MIME type headers from the HTTP request,
    since mobile camera images frequently arrive as
    'application/octet-stream' or with no content-type at all.
    """
    if len(image_bytes) < 100:
        raise ValueError("Image data is too small or empty")
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()  # Force decode — catches truncated files
        return img.convert("RGB")
    except UnidentifiedImageError:
        raise ValueError(
            "Could not identify image format. "
            "Please use JPEG or PNG taken in natural lighting."
        )
    except Exception as e:
        raise ValueError(f"Image could not be opened: {e}")


class SoilClassifier:
    MODEL_PATH = "models/soil_model"

    def __init__(self):
        self.model   = None
        self._infer  = None
        self.source  = "color_heuristic"
        self._load()

    def _load(self):
        try:
            import tensorflow as tf
            if os.path.exists(self.MODEL_PATH):
                m = tf.saved_model.load(self.MODEL_PATH)
                self._infer = m.signatures["serving_default"]
                self.model  = m
                self.source = "pixsoil_local"
                print("✅ Pixsoil model loaded (11-class, 256×256 input)")
            else:
                print(f"⚠️  Pixsoil model not found at '{self.MODEL_PATH}'. "
                      "Colour heuristic will be used.")
        except Exception as e:
            print(f"⚠️  Pixsoil load error: {e}. Using colour heuristic.")

    def classify(self, image_bytes: bytes) -> dict:
        """
        Classify soil from raw image bytes.
        FIX: Validates image using Pillow (not MIME type) before classification.
        """
        # Validate image first — raises ValueError with friendly message if bad
        try:
            img = _validate_and_open_image(image_bytes)
        except ValueError as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=422, detail=str(e))

        if self.model is not None:
            return self._run_model_from_img(img, image_bytes)
        return self._color_heuristic_from_img(img)

    def _run_model(self, image_bytes: bytes) -> dict:
        """Legacy entry point — redirects to classify()."""
        return self.classify(image_bytes)

    def _run_model_from_img(self, img: Image.Image, image_bytes: bytes) -> dict:
        """Run TF model inference on a pre-opened PIL Image."""
        import tensorflow as tf
        try:
            resized = img.resize((256, 256), Image.BILINEAR)
            arr = np.expand_dims(
                np.array(resized, dtype=np.float32) / 255.0, axis=0
            )

            out_dict = self._infer(**{"conv2d_62_input": tf.constant(arr)})
            probs    = list(out_dict.values())[0].numpy()[0]

            if abs(probs.sum() - 1.0) > 0.05:
                probs = _softmax(probs)

            top_idx = int(probs.argmax())
            label   = SOIL_LABELS[top_idx]
            conf    = float(probs[top_idx]) * 100.0

            if label in _INVALID_LABELS:
                best_valid = max(_VALID_INDICES, key=lambda i: probs[i])
                if probs[best_valid] >= 0.20:
                    label = SOIL_LABELS[best_valid]
                    conf  = float(probs[best_valid]) * 100.0
                    print(f"  Context label predicted — using runner-up: {label} ({conf:.1f}%)")
                else:
                    print("  Low confidence on all valid soil labels — falling back to colour heuristic.")
                    return self._color_heuristic_from_img(img)

            return self._build_result(label, conf, "pixsoil_local")

        except Exception as e:
            print(f"Pixsoil inference error: {e}")
            return self._color_heuristic_from_img(img)

    def _color_heuristic(self, image_bytes: bytes) -> dict:
        """Legacy entry point for colour heuristic."""
        img = _validate_and_open_image(image_bytes)
        return self._color_heuristic_from_img(img)

    def _color_heuristic_from_img(self, img: Image.Image) -> dict:
        """Colour heuristic on a pre-opened PIL Image."""
        small  = img.resize((64, 64))
        arr    = np.array(small, dtype=np.float32)
        r, g, b = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
        brightness = (r + g + b) / 3.0
        redness    = r / (g + 1)

        if brightness < 55:
            label, conf = "humus",      60.0
        elif brightness < 75:
            label, conf = "clay",       62.0
        elif redness > 1.4 and brightness < 130:
            label, conf = "yellow",     63.0
        elif redness > 1.2 and brightness > 130:
            label, conf = "dry rocky",  58.0
        elif brightness > 170 and r > 155 and g > 140:
            label, conf = "sandy",      60.0
        elif b > g and brightness < 130:
            label, conf = "clay",       60.0
        elif g > r * 0.95 and 110 < brightness < 165:
            label, conf = "loam",       59.0
        elif 100 < brightness < 160 and abs(r - g) < 25:
            label, conf = "silty",      59.0
        elif brightness < 100 and redness < 1.1:
            label, conf = "gravel",     55.0
        else:
            label, conf = "loam",       52.0

        return self._build_result(label, conf, "color_heuristic")

    def _build_result(self, label: str, conf: float, source: str) -> dict:
        profile  = SOIL_PROFILES.get(label, SOIL_PROFILES["_unknown"])
        n_mid    = (profile["N"][0] + profile["N"][1]) // 2
        p_mid    = (profile["P"][0] + profile["P"][1]) // 2
        k_mid    = (profile["K"][0] + profile["K"][1]) // 2
        ph_mid   = round((profile["pH"][0] + profile["pH"][1]) / 2, 1)
        return {
            "soil_type"       : profile["display_name"],
            "soil_label"      : label,
            "soil_type_short" : SHORT_NAME.get(label, profile["display_name"]),
            "confidence"      : round(conf, 1),
            "description"     : profile["description"],
            "estimated_N"     : n_mid,
            "estimated_P"     : p_mid,
            "estimated_K"     : k_mid,
            "estimated_pH"    : ph_mid,
            "n_range"         : list(profile["N"]),
            "p_range"         : list(profile["P"]),
            "k_range"         : list(profile["K"]),
            "ph_range"        : list(profile["pH"]),
            "best_crops"      : profile["crops"],
            "source"          : source,
        }


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()