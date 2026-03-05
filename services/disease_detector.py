import io, json, torch
import numpy as np
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

# ── Crop name → label prefix mapping ─────────────────────────────────
# Class names use format: "Crop__disease" (double underscore)
# Keys are lowercase versions of what Flutter sends as crop_name.
# Value is a list of prefixes to match against class_names.
# None = crop not in dataset → skip filtering, use global argmax.
CROP_ALIASES: dict[str, list[str] | None] = {
    "tomato"     : ["Tomato"],
    "rice"       : ["Rice"],
    "wheat"      : ["Wheat"],
    "apple"      : ["Apple"],
    "grape"      : ["Grape"],
    "grapes"     : ["Grape"],
    "corn"       : ["Corn"],
    "maize"      : ["Corn"],
    "potato"     : ["Potato"],
    "soybean"    : ["Soybean"],
    "tea"        : ["Tea"],
    "cherry"     : ["Cherry"],
    "peach"      : ["Peach"],
    "pepper"     : ["Pepper"],
    "bell pepper": ["Pepper"],
    "capsicum"   : ["Pepper"],
    "strawberry" : ["Strawberry"],
    "blueberry"  : ["Blueberry"],
    "raspberry"  : ["Raspberry"],
    "squash"     : ["Squash"],
    "orange"     : ["Orange"],
    # Crops not in this dataset — no filtering
    "cotton"     : None,
    "sugarcane"  : None,
    "mango"      : None,
    "banana"     : None,
    "coconut"    : None,
    "jute"       : None,
    "groundnut"  : None,
    "mustard"    : None,
}

# ── Treatment database — keyed on lowercase label for flexibility ─────
# Format: label_fragment → (friendly_name, chemical_action, organic_action)
# We match by checking if key is a substring of raw_label.lower()
DISEASE_ACTIONS: dict[str, tuple[str, str, str]] = {
    # Tomato
    "tomato__late_blight"           : ("Tomato Late Blight",      "Apply mancozeb or metalaxyl immediately. Remove infected plants.",       "Copper sulfate spray"),
    "tomato__early_blight"          : ("Tomato Early Blight",     "Apply chlorothalonil. Mulch around base of plants.",                     "Baking soda spray (1 tbsp/L)"),
    "tomato__bacterial_spot"        : ("Tomato Bacterial Spot",   "Apply copper bactericide. Avoid wetting foliage.",                       "Neem oil + soap solution"),
    "tomato__leaf_mold"             : ("Tomato Leaf Mold",        "Improve ventilation. Apply mancozeb fungicide.",                         "Diluted milk spray (1:9 ratio)"),
    "tomato__septoria_leaf_spot"    : ("Tomato Septoria Leaf Spot","Apply chlorothalonil at first sign. Remove lower leaves.",              "Copper-based organic spray"),
    "tomato__spider_mites"          : ("Tomato Spider Mites",     "Apply abamectin miticide. Increase humidity.",                           "Neem oil spray, release predatory mites"),
    "tomato__target_spot"           : ("Tomato Target Spot",      "Apply azoxystrobin fungicide.",                                          "Remove affected leaves, improve airflow"),
    "tomato__yellow_leaf_curl"      : ("Tomato Yellow Leaf Curl Virus","Remove infected plants. Control whitefly vector with imidacloprid.","Yellow sticky traps, neem oil"),
    "tomato__mosaic_virus"          : ("Tomato Mosaic Virus",     "Remove infected plants immediately. Disinfect tools.",                   "Wash hands/tools with soap. No organic cure."),
    "tomato__healthy"               : ("Healthy Tomato",          "No action needed.",                                                      "Continue regular neem spray as preventive"),
    # Rice
    "rice__hispa"                   : ("Rice Hispa",              "Apply chlorpyrifos. Clip and destroy affected leaves.",                   "Light trapping, release egg parasitoids"),
    "rice__leaf_blast"              : ("Rice Leaf Blast",         "Apply tricyclazole fungicide at early stage.",                            "Silica-based soil amendment, balanced N"),
    "rice__neck_blast"              : ("Rice Neck Blast",         "Apply propiconazole. Drain and re-flood field.",                          "Avoid excess nitrogen, silica amendment"),
    "rice__brown_spot"              : ("Rice Brown Spot",         "Apply mancozeb. Ensure potassium sufficiency.",                           "Balanced fertilization, remove infected debris"),
    "rice__bacterial_leaf_blight"   : ("Rice Bacterial Leaf Blight","Apply copper oxychloride. Drain standing water.",                      "Remove infected stubble, balanced nutrition"),
    "rice__healthy"                 : ("Healthy Rice",            "No action needed.",                                                      "Maintain balanced nutrition"),
    # Wheat
    "wheat__yellow_rust"            : ("Wheat Yellow Rust",       "Apply propiconazole or tebuconazole urgently.",                           "Use resistant varieties, remove volunteers"),
    "wheat__brown_rust"             : ("Wheat Brown Rust",        "Apply triazole fungicide at flag-leaf stage.",                            "Crop rotation, use resistant varieties"),
    "wheat__loose_smut"             : ("Wheat Loose Smut",        "Apply carboxin seed treatment.",                                         "Use certified disease-free seed"),
    "wheat__healthy"                : ("Healthy Wheat",           "No action needed.",                                                      "Regular monitoring"),
    # Apple
    "apple__black_rot"              : ("Apple Black Rot",         "Prune infected branches. Apply copper fungicide.",                        "Bordeaux mixture, remove mummified fruit"),
    "apple__apple_scab"             : ("Apple Scab",              "Apply captan or myclobutanil fungicide.",                                 "Neem oil spray weekly"),
    "apple__cedar_apple_rust"       : ("Cedar Apple Rust",        "Apply myclobutanil fungicide in spring.",                                 "Remove nearby juniper trees"),
    "apple__healthy"                : ("Healthy Apple",           "No action needed.",                                                      "Regular pruning for airflow"),
    # Corn / Maize
    "corn__common_rust"             : ("Corn Common Rust",        "Apply propiconazole. Avoid overhead irrigation.",                         "Sulfur-based spray"),
    "corn__northern_leaf_blight"    : ("Corn Northern Leaf Blight","Apply azoxystrobin at early stage.",                                    "Remove crop debris after harvest"),
    "corn__gray_leaf_spot"          : ("Corn Gray Leaf Spot",     "Apply strobilurin fungicide.",                                           "Crop rotation, remove debris"),
    "corn__healthy"                 : ("Healthy Corn",            "No action needed.",                                                      "Maintain soil moisture"),
    # Potato
    "potato__early_blight"          : ("Potato Early Blight",     "Apply mancozeb. Rotate crops next season.",                              "Copper sulfate spray"),
    "potato__late_blight"           : ("Potato Late Blight",      "Apply metalaxyl urgently. Destroy infected plants.",                     "Garlic extract spray"),
    "potato__healthy"               : ("Healthy Potato",          "No action needed.",                                                      "Hilling soil around stems"),
    # Grape
    "grape__black_rot"              : ("Grape Black Rot",         "Apply myclobutanil at bud break.",                                       "Remove mummified berries, prune canopy"),
    "grape__esca"                   : ("Grape Esca",              "No chemical cure. Prune affected wood.",                                  "Protect pruning wounds with paste"),
    "grape__leaf_blight"            : ("Grape Leaf Blight",       "Apply copper-based fungicide.",                                          "Improve canopy ventilation"),
    "grape__healthy"                : ("Healthy Grape",           "No action needed.",                                                      "Regular canopy management"),
    # Tea
    "tea__algal_leaf"               : ("Tea Algal Leaf Spot",     "Apply copper oxychloride.",                                              "Improve drainage and airflow"),
    "tea__anthracnose"              : ("Tea Anthracnose",         "Apply carbendazim fungicide.",                                           "Remove infected leaves, avoid wounding"),
    "tea__bird_eye_spot"            : ("Tea Bird Eye Spot",       "Apply mancozeb fungicide.",                                             "Remove infected material"),
    "tea__brown_blight"             : ("Tea Brown Blight",        "Apply propiconazole.",                                                   "Improve drainage, balanced fertilization"),
    "tea__healthy"                  : ("Healthy Tea",             "No action needed.",                                                      "Regular pruning and fertilization"),
    # Generic healthy fallback
    "healthy"                       : ("Healthy Plant",           "No action needed. Continue regular care.",                               "Regular neem spray as preventive measure"),
}

def _get_treatment(raw_label: str) -> tuple[str, str, str]:
    """Match raw label to treatment using substring matching on lowercased label."""
    label_lower = raw_label.lower()
    # Try most specific match first (longer keys win)
    best_key, best_len = None, 0
    for key in DISEASE_ACTIONS:
        if key in label_lower and len(key) > best_len:
            best_key, best_len = key, len(key)
    if best_key:
        return DISEASE_ACTIONS[best_key]
    # Fallback: derive name from label itself
    parts   = raw_label.replace("__", "___").split("___")
    crop    = parts[0].replace("_", " ").strip()
    disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown condition"
    return (
        f"{crop} — {disease}",
        "Consult your local agricultural extension officer.",
        "Neem oil spray as general preventive measure",
    )

def _parse_label(raw_label: str) -> dict:
    friendly_name, action, organic = _get_treatment(raw_label)
    healthy = "healthy" in raw_label.lower()
    # Extract crop name from label  (handles both __ and ___ separators)
    parts = raw_label.replace("__", "|||").split("|||")
    crop  = parts[0].replace("_", " ").strip() if parts else "Unknown"
    return {
        "disease_name"    : friendly_name,
        "crop_identified" : crop,
        "is_healthy"      : healthy,
        "immediate_action": action,
        "organic_option"  : organic,
    }

def _get_allowed_indices(crop_name: str, class_names: list[str]) -> list[int] | None:
    """
    Return indices in class_names that match the given crop.
    Returns None if crop is unknown or not in dataset → use global argmax.
    Returns [] only if crop IS in dataset but zero classes matched (shouldn't happen
    with correct CROP_ALIASES, but we handle it gracefully).
    """
    key = crop_name.strip().lower()
    prefixes = CROP_ALIASES.get(key, "NOT_FOUND")

    if prefixes == "NOT_FOUND":
        # Crop name not in our alias table at all — try direct substring match
        # This handles typos or new crop names gracefully
        direct = [i for i, c in enumerate(class_names) if key in c.lower()]
        return direct if direct else None   # None → global argmax

    if prefixes is None:
        # Crop explicitly marked as not in dataset
        return None

    # Match any class whose name starts with one of our prefixes (case-insensitive)
    allowed = [
        i for i, c in enumerate(class_names)
        if any(c.lower().startswith(p.lower()) for p in prefixes)
    ]
    return allowed if allowed else None  # None → global argmax, not empty list


class DiseaseDetector:
    REPO = "Arko007/nfnet-f1-plant-disease"

    def __init__(self, hf_token: str = ""):
        self.model       = None
        self.class_names : list[str] = []
        self.transform   = None
        self.source      = "mock"
        self.token       = hf_token or None
        self._load()

    def _load(self):
        try:
            import timm
            from safetensors.torch import load_file

            kw = {"repo_id": self.REPO, "token": self.token}
            weights_path = hf_hub_download(**kw, filename="model.safetensors")
            config_path  = hf_hub_download(**kw, filename="config.json")

            with open(config_path) as f:
                cfg = json.load(f)

            # ── Read EVERYTHING from config — don't hardcode ──────────
            self.class_names = cfg["class_names"]          # list of 88 strings
            img_size         = cfg["input_size"]           # 512
            arch             = cfg["architecture"]         # "nfnet_f1"
            mean             = cfg["normalization"]["mean"]
            std              = cfg["normalization"]["std"]

            print(f"  Config loaded: {len(self.class_names)} classes, {img_size}px, arch={arch}")

            # ── Build transform exactly as the model card shows ───────
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

            # ── Build model ───────────────────────────────────────────
            self.model = timm.create_model(
                arch,
                pretrained  = False,
                num_classes = len(self.class_names),
            )
            state = load_file(weights_path)
            self.model.load_state_dict(state, strict=False)
            self.model.eval()
            torch.set_num_threads(4)

            self.source = self.REPO
            print(f"✅ Disease model loaded: {self.REPO}")

        except Exception as e:
            print(f"⚠️  Disease model load failed: {e}. Using mock.")

    async def diagnose(self, image_bytes: bytes, crop_name: str = "Unknown") -> dict:
        if self.model is None:
            return self._mock(crop_name)
        try:
            # ── Preprocess ────────────────────────────────────────────
            img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = self.transform(img).unsqueeze(0)  # (1, 3, 512, 512)

            # ── Inference ─────────────────────────────────────────────
            with torch.no_grad():
                logits = self.model(tensor)
                probs  = torch.softmax(logits, dim=1)[0].numpy()  # (88,)

            # ── Crop-filtered argmax ──────────────────────────────────
            if crop_name.strip().lower() not in ("unknown", ""):
                allowed = _get_allowed_indices(crop_name, self.class_names)
            else:
                allowed = None  # no crop hint → global argmax

            if allowed is not None and len(allowed) > 0:
                # Restrict to crop-specific subset
                subset_probs = probs[allowed]
                best_local   = int(np.argmax(subset_probs))
                top_idx      = allowed[best_local]
                confidence   = round(float(probs[top_idx]) * 100, 1)
                print(f"  Crop filter: '{crop_name}' → {len(allowed)} classes → '{self.class_names[top_idx]}' ({confidence}%)")
            else:
                # Global argmax (unknown crop or crop not in dataset)
                top_idx    = int(np.argmax(probs))
                confidence = round(float(probs[top_idx]) * 100, 1)
                print(f"  Global argmax → '{self.class_names[top_idx]}' ({confidence}%)")

            raw_label = self.class_names[top_idx]
            parsed    = _parse_label(raw_label)
            severity  = 1 if parsed["is_healthy"] else min(5, max(1, int((1 - probs[top_idx]) * 8 + 2)))

            return {
                **parsed,
                "severity"                    : severity,
                "confidence"                  : confidence,
                "affected_area_pct"           : round((severity / 5) * 40, 1),
                "days_until_critical"         : max(2, 7 - severity),
                "alert_neighbours_recommended": not parsed["is_healthy"] and severity >= 3,
                "raw_label"                   : raw_label,
            }

        except Exception as e:
            print(f"Inference error: {e}")
            return self._mock(crop_name)

    def _mock(self, crop_name: str) -> dict:
        return {
            "disease_name"                : "Bacterial Leaf Blight",
            "crop_identified"             : crop_name,
            "is_healthy"                  : False,
            "severity"                    : 3,
            "confidence"                  : 82.0,
            "affected_area_pct"           : 20.0,
            "days_until_critical"         : 5,
            "alert_neighbours_recommended": True,
            "immediate_action"            : "1. Remove infected leaves\n2. Apply copper bactericide\n3. Avoid overhead irrigation",
            "organic_option"              : "Spray 1% Bordeaux mixture weekly",
            "raw_label"                   : "mock",
        }