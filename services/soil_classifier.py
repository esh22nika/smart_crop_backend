import io
import numpy as np
from PIL import Image
import os

SOIL_CLASSES = ["Alluvial", "Black", "Clay", "Red"]
SOIL_NPK = {
    "Alluvial": dict(N=(80,120), P=(40,70),  K=(90,140), pH=(6.5,8.0),
                     crops=["Rice","Wheat","Sugarcane","Maize","Mustard"]),
    "Black":    dict(N=(60,80),  P=(50,80),  K=(80,120), pH=(7.0,8.5),
                     crops=["Cotton","Soybean","Wheat","Sorghum","Sunflower"]),
    "Clay":     dict(N=(50,75),  P=(40,65),  K=(70,110), pH=(5.5,7.5),
                     crops=["Rice","Jute","Taro"]),
    "Red":      dict(N=(20,40),  P=(15,30),  K=(40,70),  pH=(5.5,7.0),
                     crops=["Groundnut","Millet","Tobacco","Pulses","Castor"]),
}

class SoilClassifier:
    MODEL_PATH = "models/soil_model"   # SavedModel folder after TFjs conversion

    def __init__(self):
        self.model  = None
        self.source = "color_heuristic"
        self._load()

    def _load(self):
        try:
            import tensorflow as tf
            if os.path.exists(self.MODEL_PATH):
                self.model  = tf.saved_model.load(self.MODEL_PATH)
                self.source = "pixsoil_local"
                print("✅ Pixsoil model loaded")
            else:
                print("⚠️  Pixsoil model not found at models/soil_model. Run conversion first.")
        except Exception as e:
            print(f"⚠️  Pixsoil load error: {e}")

    def classify(self, image_bytes: bytes) -> dict:
        return self._run_model(image_bytes) if self.model else self._color_heuristic(image_bytes)

    def _run_model(self, image_bytes: bytes) -> dict:
        import tensorflow as tf
        try:
            img   = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
            arr   = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, 0)
            infer = self.model.signatures["serving_default"]
            probs = list(infer(tf.constant(arr)).values())[0].numpy()[0]
            idx   = int(probs.argmax())
            return self._make_result(SOIL_CLASSES[idx], float(probs[idx]) * 100, "pixsoil_local")
        except Exception as e:
            print(f"Pixsoil inference error: {e}")
            return self._color_heuristic(image_bytes)

    def _color_heuristic(self, image_bytes: bytes) -> dict:
        img   = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((64, 64))
        arr   = np.array(img, dtype=np.float32)
        r, g, b = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
        if b < 60 and r < 80:      soil, conf = "Black",    65.0
        elif r > g * 1.3:          soil, conf = "Red",      62.0
        elif r > 140 and g > 120:  soil, conf = "Alluvial", 60.0
        else:                      soil, conf = "Clay",     58.0
        return self._make_result(soil, conf, "color_heuristic")

    def _make_result(self, soil: str, conf: float, source: str) -> dict:
        npk = SOIL_NPK.get(soil, SOIL_NPK["Black"])
        return {
            "soil_type"   : soil,
            "confidence"  : round(conf, 1),
            "estimated_N" : (npk["N"][0] + npk["N"][1]) // 2,
            "estimated_P" : (npk["P"][0] + npk["P"][1]) // 2,
            "estimated_K" : (npk["K"][0] + npk["K"][1]) // 2,
            "estimated_pH": round((npk["pH"][0] + npk["pH"][1]) / 2, 1),
            "n_range"     : list(npk["N"]),
            "p_range"     : list(npk["P"]),
            "k_range"     : list(npk["K"]),
            "ph_range"    : list(npk["pH"]),
            "best_crops"  : npk["crops"],
            "source"      : source,
        }


