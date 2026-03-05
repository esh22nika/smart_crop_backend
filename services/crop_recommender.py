import joblib, numpy as np
from huggingface_hub import hf_hub_download

CROP_ID_MAP = {
    1:"Rice", 2:"Maize", 3:"Jute", 4:"Cotton", 5:"Coconut",
    6:"Papaya", 7:"Orange", 8:"Apple", 9:"Muskmelon", 10:"Watermelon",
    11:"Grapes", 12:"Mango", 13:"Banana", 14:"Pomegranate", 15:"Lentil",
    16:"Blackgram", 17:"Mungbean", 18:"Mothbeans", 19:"Pigeonpeas",
    20:"Kidneybeans", 21:"Chickpea", 22:"Coffee",
}
CROP_EMOJI = {
    "Rice":"🌾","Maize":"🌽","Cotton":"🌿","Wheat":"🌾","Jute":"🌿",
    "Coconut":"🥥","Papaya":"🍈","Orange":"🍊","Apple":"🍎","Mango":"🥭",
    "Banana":"🍌","Grapes":"🍇","Watermelon":"🍉","Muskmelon":"🍈",
    "Pomegranate":"🍎","Lentil":"🫘","Blackgram":"🫘","Mungbean":"🫘",
    "Mothbeans":"🫘","Pigeonpeas":"🫘","Kidneybeans":"🫘",
    "Chickpea":"🫘","Coffee":"☕",
}

class CropRecommender:
    REPO = "Arko007/agromind-crop-recommendation"

    def __init__(self, hf_token: str = ""):
        self.model  = None
        self.sc     = None   # StandardScaler
        self.ms     = None   # MinMaxScaler
        self.source = "mock"
        self.token  = hf_token or None
        self._load()

    def _load(self):
        try:
            kw = {"repo_id": self.REPO, "token": self.token}
            self.model = joblib.load(hf_hub_download(**kw, filename="crop_predict_model.pkl"))
            self.sc    = joblib.load(hf_hub_download(**kw, filename="crop_predict_standscaler.pkl"))
            self.ms    = joblib.load(hf_hub_download(**kw, filename="crop_predict_minmaxscaler.pkl"))
            self.source = "Arko007/agromind-crop-recommendation"
            print(f"✅ Crop model loaded from HF Hub")
        except Exception as e:
            print(f"⚠️  Crop model load failed: {e}. Using mock.")

    def recommend(self, N, P, K, temperature, humidity, ph, rainfall) -> list[dict]:
        if self.model is None:
            return self._mock()
        # IMPORTANT: apply MinMax first, then Standard — exactly as trained
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        scaled   = self.sc.transform(self.ms.transform(features))
        probs    = self.model.predict_proba(scaled)[0]
        top3_idx = np.argsort(probs)[::-1][:3]
        results  = []
        for i in top3_idx:
            # Model outputs class labels 1-22 directly
            # predict_proba classes_ tells us the mapping
            label    = self.model.classes_[i]
            crop_id  = int(label)
            name     = CROP_ID_MAP.get(crop_id, f"Crop_{crop_id}")
            results.append({
                "crop_name" : name,
                "emoji"     : CROP_EMOJI.get(name, "🌾"),
                "confidence": round(float(probs[i]) * 100, 1),
            })
        return results

    def _mock(self) -> list[dict]:
        return [
            {"crop_name":"Soybean","emoji":"🫘","confidence":91.0},
            {"crop_name":"Cotton", "emoji":"🌿","confidence":76.0},
            {"crop_name":"Rice",   "emoji":"🌾","confidence":63.0},
        ]


