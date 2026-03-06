"""
Smart Crop DSS — FastAPI Backend  v5.0
Maharashtra-focused.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from contextlib import asynccontextmanager
import sqlite3, os, httpx, asyncio
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

from services.weather_service   import WeatherService
from services.soil_classifier   import SoilClassifier
from services.crop_recommender  import CropRecommender
from services.disease_detector  import DiseaseDetector
from services.risk_engine       import RiskEngine, CROP_PROFILES, INPUT_COSTS, DEFAULT_INPUT_COST, MARKET_DATA
from services.pest_engine       import PestEngine

# ── Globals ───────────────────────────────────────────────────────────
weather_svc : WeatherService  = None
soil_clf    : SoilClassifier  = None
crop_rec    : CropRecommender = None
disease_det : DiseaseDetector = None
risk_eng    : RiskEngine      = None
pest_eng    : PestEngine      = None

DB_PATH          = "sightings.db"
DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY", "")

# ── In-process cache ─────────────────────────────────────────────────
_price_cache: dict = {}
_msp_cache  : dict = {}
_yield_cache: dict = {}
_CACHE_TTL = 3600

# ── Harvest days ──────────────────────────────────────────────────────
HARVEST_DAYS: dict[str, int] = {
    "rice": 120,       "wheat": 120,        "cotton": 180,
    "soybean": 100,    "soyabean": 100,     "maize": 100,
    "sugarcane": 365,  "groundnut": 130,    "banana": 300,
    "mango": 120,      "coconut": 365,      "pomegranate": 180,
    "grapes": 240,     "onion": 120,        "tomato": 80,
    "chickpea": 110,   "gram": 110,         "pigeonpeas": 180,
    "arhar/tur": 180,  "lentil": 110,       "orange": 300,
    "coffee": 365,     "jute": 120,         "mungbean": 65,
    "blackgram": 75,   "urad": 75,          "watermelon": 90,
    "muskmelon": 85,   "papaya": 240,       "mothbeans": 90,
    "kidneybeans": 90, "apple": 150,        "bajra": 80,
    "jowar": 110,      "ragi": 120,         "sunflower": 100,
    "sesamum": 80,     "safflower": 130,    "linseed": 120,
}

# ── Nearest Maharashtra APMC mandis per district ─────────────────────
DISTRICT_MANDI: dict[str, list[str]] = {
    "Nagpur"                   : ["Nagpur", "Wardha"],
    "Wardha"                   : ["Wardha", "Nagpur"],
    "Amravati"                 : ["Amravati", "Akola"],
    "Akola"                    : ["Akola", "Washim"],
    "Washim"                   : ["Washim", "Akola"],
    "Buldhana"                 : ["Buldhana", "Akola"],
    "Yavatmal"                 : ["Yavatmal", "Wardha"],
    "Chandrapur"               : ["Chandrapur", "Nagpur"],
    "Gadchiroli"               : ["Gadchiroli", "Chandrapur"],
    "Gondia"                   : ["Gondia", "Bhandara"],
    "Bhandara"                 : ["Bhandara", "Nagpur"],
    "Chhatrapati Sambhajinagar": ["Aurangabad", "Jalna"],
    "Dharashiv"                : ["Osmanabad", "Latur"],
    "Beed"                     : ["Beed", "Aurangabad"],
    "Hingoli"                  : ["Hingoli", "Nanded"],
    "Jalna"                    : ["Jalna", "Aurangabad"],
    "Latur"                    : ["Latur", "Osmanabad"],
    "Nanded"                   : ["Nanded", "Latur"],
    "Parbhani"                 : ["Parbhani", "Hingoli"],
    "Pune"                     : ["Pune", "Satara"],
    "Nashik"                   : ["Nashik", "Yeola"],
    "Ahilyanagar"              : ["Ahmednagar", "Kopargaon"],
    "Solapur"                  : ["Solapur", "Pandharpur"],
    "Satara"                   : ["Satara", "Karad"],
    "Sangli"                   : ["Sangli", "Miraj"],
    "Kolhapur"                 : ["Kolhapur", "Ichalkaranji"],
    "Raigad"                   : ["Alibag", "Panvel"],
    "Ratnagiri"                : ["Ratnagiri", "Chiplun"],
    "Sindhudurg"               : ["Sindhudurg", "Sawantwadi"],
    "Thane"                    : ["Thane", "Kalyan"],
    "Palghar"                  : ["Palghar", "Vasai"],
    "Mumbai suburban"          : ["Mumbai", "Vashi"],
    "Dhule"                    : ["Dhule", "Shirpur"],
    "Nandurbar"                : ["Nandurbar", "Shahada"],
    "Jalgaon"                  : ["Jalgaon", "Bhusawal"],
}

# ── Agmarknet commodity name map ──────────────────────────────────────
_COMMODITY_MAP: dict[str, str] = {
    "rice": "Rice",             "wheat": "Wheat",
    "cotton": "Cotton(Lint)",   "soybean": "Soybean",
    "soyabean": "Soybean",      "maize": "Maize",
    "sugarcane": "Sugarcane",   "groundnut": "Groundnut",
    "banana": "Banana",         "mango": "Mango",
    "coconut": "Coconut",       "pomegranate": "Pomegranate",
    "grapes": "Grapes",         "onion": "Onion",
    "tomato": "Tomato",         "chickpea": "Gram",
    "gram": "Gram",             "pigeonpeas": "Arhar(Tur/Red Gram)(Whole)",
    "arhar/tur": "Arhar(Tur/Red Gram)(Whole)",
    "lentil": "Lentil (Masur)(Whole)",
    "orange": "Orange",         "mungbean": "Moong(Green Gram)(Whole)",
    "blackgram": "Black Gram (Urd Beans)(Whole)",
    "urad": "Black Gram (Urd Beans)(Whole)",
    "watermelon": "Water Melon","muskmelon": "Musk Melon",
    "papaya": "Papaya",         "bajra": "Bajra(Pearl Millet/Cumbu)",
    "jowar": "Jowar(Sorghum)",  "sunflower": "Sunflower Seed",
    "ragi": "Ragi (Finger Millet/Nagli/Ragi)",
}

MARKET_PRICES_FALLBACK: dict[str, float] = {
    "rice": 2183,        "wheat": 2275,       "cotton": 6680,
    "soybean": 4600,     "soyabean": 4600,    "maize": 2090,
    "sugarcane": 3150,   "groundnut": 5550,   "banana": 1400,
    "mango": 3200,       "coconut": 2800,     "pomegranate": 8000,
    "grapes": 5500,      "onion": 1800,       "tomato": 2500,
    "chickpea": 5440,    "gram": 5440,        "pigeonpeas": 7000,
    "arhar/tur": 7000,   "lentil": 6425,      "orange": 3500,
    "coffee": 9000,      "jute": 4750,        "mungbean": 8558,
    "blackgram": 7400,   "urad": 7400,        "watermelon": 800,
    "muskmelon": 1200,   "papaya": 1500,      "mothbeans": 8558,
    "kidneybeans": 6000, "apple": 12000,      "bajra": 2500,
    "jowar": 2800,       "sunflower": 7280,   "ragi": 3000,
    "sesamum": 7830,     "safflower": 5800,
}

MSP_FALLBACK: dict[str, float] = {
    "rice": 2300,     "wheat": 2275,     "cotton": 7121,
    "soybean": 4892,  "soyabean": 4892,  "maize": 2225,
    "groundnut": 6783,"chickpea": 5440,  "gram": 5440,
    "pigeonpeas": 7000,"lentil": 6425,   "mungbean": 8682,
    "blackgram": 7400, "urad": 7400,     "sugarcane": 3400,
    "jute": 5335,     "sunflower": 7280, "bajra": 2625,
    "jowar": 3371,    "ragi": 4290,      "sesamum": 9267,
    "safflower": 5800,
}

IRRIGATION_YIELD_FACTOR = {"Full": 1.20, "Partial": 1.0, "None": 0.80}

YIELD_BENCHMARKS: dict[str, dict] = {
    "rice": dict(low=900, high=1600),     "wheat": dict(low=1000, high=1800),
    "cotton": dict(low=200, high=500),    "soybean": dict(low=600, high=1100),
    "soyabean": dict(low=600, high=1100), "maize": dict(low=900, high=1700),
    "sugarcane": dict(low=20000, high=40000),
    "groundnut": dict(low=500, high=900), "banana": dict(low=7000, high=15000),
    "mango": dict(low=2000, high=6000),   "coconut": dict(low=3000, high=8000),
    "pomegranate": dict(low=3000, high=8000),
    "grapes": dict(low=4000, high=10000), "onion": dict(low=5000, high=12000),
    "tomato": dict(low=6000, high=15000), "chickpea": dict(low=350, high=700),
    "gram": dict(low=350, high=700),      "pigeonpeas": dict(low=400, high=800),
    "arhar/tur": dict(low=400, high=800), "lentil": dict(low=300, high=600),
    "orange": dict(low=2500, high=6000),  "coffee": dict(low=300, high=700),
    "jute": dict(low=1500, high=2800),    "mungbean": dict(low=200, high=450),
    "blackgram": dict(low=200, high=450), "urad": dict(low=200, high=450),
    "watermelon": dict(low=8000, high=18000),
    "muskmelon": dict(low=4000, high=10000),
    "papaya": dict(low=8000, high=20000), "bajra": dict(low=700, high=1400),
    "jowar": dict(low=800, high=1500),    "sunflower": dict(low=400, high=800),
    "ragi": dict(low=600, high=1200),
}


# ══════════════════════════════════════════════════════════════════════
#  DB + lifespan
# ══════════════════════════════════════════════════════════════════════

def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sightings (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            district TEXT    NOT NULL,
            crop     TEXT    NOT NULL,
            pest     TEXT    NOT NULL,
            severity TEXT    NOT NULL,
            ts       DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit(); conn.close()
    print("✅ DB initialised")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global weather_svc, soil_clf, crop_rec, disease_det, risk_eng, pest_eng
    print("🌱 Starting SmartCrop backend v5...")
    _init_db()
    weather_svc = WeatherService(os.getenv("OPENWEATHER_API_KEY", ""))
    soil_clf    = SoilClassifier()
    crop_rec    = CropRecommender(os.getenv("HF_API_TOKEN", ""))
    disease_det = DiseaseDetector(os.getenv("HF_API_TOKEN", ""))
    risk_eng    = RiskEngine()
    pest_eng    = PestEngine()
    asyncio.create_task(_warm_msp_cache())
    print("✅ All services ready.")
    yield


app = FastAPI(title="Smart Crop DSS", version="5.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ══════════════════════════════════════════════════════════════════════
#  Agmarknet helpers (unchanged from original)
# ══════════════════════════════════════════════════════════════════════

_AGMARKNET_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"


async def _fetch_mandi_prices(crop_key: str, district: str) -> Optional[dict]:
    if not DATA_GOV_API_KEY:
        return None
    cache_key = f"mandi:{district.lower()}:{crop_key}"
    cached = _price_cache.get(cache_key)
    if cached and (datetime.now() - cached["_ts"]).seconds < _CACHE_TTL:
        return {k: v for k, v in cached.items() if k != "_ts"}
    commodity = _COMMODITY_MAP.get(crop_key)
    if not commodity:
        return None
    mandis = DISTRICT_MANDI.get(district, ["Pune"])
    all_records: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            for mandi in mandis[:2]:
                params = {
                    "api-key"            : DATA_GOV_API_KEY,
                    "format"             : "json",
                    "limit"              : "10",
                    "filters[State]"     : "Maharashtra",
                    "filters[Commodity]" : commodity,
                    "filters[Market]"    : mandi,
                }
                r = await client.get(_AGMARKNET_URL, params=params)
                r.raise_for_status()
                for rec in r.json().get("records", []):
                    v = _parse_record(rec)
                    if v:
                        all_records.append(v)
                if all_records:
                    break
            if not all_records:
                params = {
                    "api-key"            : DATA_GOV_API_KEY,
                    "format"             : "json",
                    "limit"              : "5",
                    "filters[State]"     : "Maharashtra",
                    "filters[Commodity]" : commodity,
                }
                r = await client.get(_AGMARKNET_URL, params=params)
                r.raise_for_status()
                for rec in r.json().get("records", []):
                    v = _parse_record(rec)
                    if v:
                        all_records.append(v)
    except Exception as e:
        print(f"⚠️  Agmarknet error {district}/{crop_key}: {e}")
        return None
    if not all_records:
        return None
    best = sorted(all_records, key=lambda r: (r["arrival_date"], r["modal_price"]), reverse=True)[0]
    result = {
        "mandi_name"  : best["mandi"],
        "commodity"   : commodity,
        "variety"     : best["variety"],
        "min_price"   : best["min_price"],
        "modal_price" : best["modal_price"],
        "max_price"   : best["max_price"],
        "arrival_date": best["arrival_date"],
        "all_mandis"  : all_records[:5],
        "source"      : "agmarknet_live",
        "_ts"         : datetime.now(),
    }
    _price_cache[cache_key] = result
    return {k: v for k, v in result.items() if k != "_ts"}


def _parse_record(rec: dict) -> Optional[dict]:
    modal = _safe_float(rec.get("Modal_Price"))
    if not modal or modal <= 0:
        return None
    return {
        "mandi"       : rec.get("Market", "Maharashtra APMC"),
        "variety"     : rec.get("Variety", "Common"),
        "min_price"   : _safe_float(rec.get("Min_Price")) or modal * 0.75,
        "modal_price" : modal,
        "max_price"   : _safe_float(rec.get("Max_Price")) or modal * 1.25,
        "arrival_date": rec.get("Arrival_Date", ""),
    }


def _safe_float(val) -> Optional[float]:
    try:
        return float(str(val).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


async def _fetch_agmarknet_modal(crop_key: str) -> Optional[float]:
    cache_key = f"modal:{crop_key}"
    cached = _price_cache.get(cache_key)
    if cached and (datetime.now() - cached["_ts"]).seconds < _CACHE_TTL:
        return cached.get("price")
    commodity = _COMMODITY_MAP.get(crop_key)
    if not commodity or not DATA_GOV_API_KEY:
        return None
    try:
        params = {
            "api-key"            : DATA_GOV_API_KEY,
            "format"             : "json",
            "limit"              : "20",
            "filters[State]"     : "Maharashtra",
            "filters[Commodity]" : commodity,
        }
        async with httpx.AsyncClient(timeout=6.0) as client:
            r = await client.get(_AGMARKNET_URL, params=params)
            r.raise_for_status()
            prices = [_safe_float(rec.get("Modal_Price")) for rec in r.json().get("records", [])]
            prices = [p for p in prices if p and p > 0]
            if not prices:
                return None
            avg = sum(prices) / len(prices)
            _price_cache[cache_key] = {"price": avg, "_ts": datetime.now()}
            return avg
    except Exception:
        return None


_MSP_RESOURCE = "35be93cd-ab6f-4fdd-859a-e1d2f04b8571"
_MSP_COMMODITY_MAP = {
    "rice": "Paddy (Common)",   "wheat": "Wheat",
    "cotton": "Cotton (Medium Staple)", "soybean": "Soybean (Yellow)",
    "soyabean": "Soybean (Yellow)",     "maize": "Maize",
    "groundnut": "Groundnut",  "chickpea": "Gram",  "gram": "Gram",
    "pigeonpeas": "Arhar/Tur", "lentil": "Masur (Lentil)",
    "mungbean": "Moong",       "blackgram": "Urad", "urad": "Urad",
    "sugarcane": "Sugarcane (FRP)", "bajra": "Bajra",
    "jowar": "Jowar (Hybrid)", "sunflower": "Sunflower Seed",
}


async def _warm_msp_cache():
    if not DATA_GOV_API_KEY:
        return
    try:
        url = f"https://api.data.gov.in/resource/{_MSP_RESOURCE}"
        params = {"api-key": DATA_GOV_API_KEY, "format": "json", "limit": "100"}
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, params=params)
        records = r.json().get("records", [])
        live: dict[str, float] = {}
        for rec in records:
            name  = (rec.get("commodity") or rec.get("Commodity", "")).strip().lower()
            price = rec.get("msp") or rec.get("MSP") or rec.get("Price", 0)
            try:
                live[name] = float(str(price).replace(",", ""))
            except (ValueError, TypeError):
                pass
        for crop_key, api_name in _MSP_COMMODITY_MAP.items():
            match = live.get(api_name.lower())
            if match:
                _msp_cache[crop_key] = match
        print(f"✅ MSP cache: {len(_msp_cache)} crops loaded")
    except Exception as e:
        print(f"⚠️  MSP cache warm failed: {e}")


# ══════════════════════════════════════════════════════════════════════
#  Pydantic schemas
# ══════════════════════════════════════════════════════════════════════

class CropRequest(BaseModel):
    N          : float = Field(..., ge=0, le=145)
    P          : float = Field(..., ge=0, le=145)
    K          : float = Field(..., ge=0, le=210)
    temperature: float = Field(..., ge=0, le=50)
    humidity   : float = Field(..., ge=0, le=100)
    ph         : float = Field(..., ge=0, le=14)
    rainfall   : float = Field(..., ge=0, le=3000)
    season     : str   = "Kharif"
    irrigation : str   = "Partial"
    land_acres : float = Field(1.0, ge=0.1, le=1000)
    budget     : Optional[float] = Field(None, ge=0)
    district   : Optional[str]   = None
    w_npk      : float = Field(0.4, ge=0.0, le=1.0)
    top_n      : int   = Field(5, ge=1, le=10)


class SightingRequest(BaseModel):
    district: str
    crop    : str
    pest    : str
    severity: str


# ══════════════════════════════════════════════════════════════════════
#  Endpoints
# ══════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status"        : "ok",
        "version"       : "5.0.0",
        "soil_model"    : soil_clf.source,
        "crop_model"    : crop_rec.source,
        "disease_model" : disease_det.source,
        "data_gov_key"  : "configured" if DATA_GOV_API_KEY else "missing",
        "msp_cached"    : len(_msp_cache),
    }


@app.get("/weather/{district}")
async def get_weather(district: str):
    return await weather_svc.fetch(district)


@app.get("/district-defaults/{district}")
def district_defaults(district: str):
    d = DISTRICT_DATA.get(district)
    if not d:
        raise HTTPException(404, f"No data for district: {district}")
    return d


# ── FIX: Relaxed content-type check for soil image upload ─────────────
# Some Android/iOS devices send 'application/octet-stream' instead of
# 'image/jpeg'. We now validate by actually attempting to open the image
# with Pillow rather than relying solely on the MIME type header.
@app.post("/analyze-soil-image")
async def analyze_soil(file: UploadFile = File(...)):
    data = await file.read()

    # Validate minimum size first
    if len(data) < 1000:
        raise HTTPException(400, "Image appears empty or corrupt")

    # Try to validate as image using Pillow — much more reliable than MIME type
    try:
        from PIL import Image as PILImage
        import io
        img = PILImage.open(io.BytesIO(data))
        img.verify()  # raises if not a valid image
    except Exception:
        # Only reject if it truly cannot be opened as an image
        # content_type check as secondary fallback
        ct = file.content_type or ""
        # Allow octet-stream through since many mobile clients send this
        # for camera images
        if ct and not ct.startswith("image/") and ct not in (
            "application/octet-stream", "binary/octet-stream", ""
        ):
            raise HTTPException(400, "Must be an image file")

    return soil_clf.classify(data)


@app.post("/recommend-crops")
async def recommend_crops(req: CropRequest):
    candidates = crop_rec.recommend(
        req.N, req.P, req.K,
        req.temperature, req.humidity,
        req.ph, req.rainfall,
        district = req.district or "",
        season   = req.season,
        w_npk    = req.w_npk,
        top_n    = min(req.top_n * 2, 10),
    )

    async def _enrich(c: dict) -> dict:
        name     = c["crop_name"]
        crop_key = name.lower()
        risk   = risk_eng.score(name, req.season, req.temperature,
                                req.humidity, req.rainfall, req.land_acres, req.budget)
        afford = risk_eng.affordability(name, req.land_acres, req.budget)
        yield_d = _yield_estimate(name, req.land_acres, req.irrigation)
        market  = _market_info(name)
        live_price = await _fetch_agmarknet_modal(crop_key)
        price      = live_price or _msp_cache.get(crop_key) or MARKET_PRICES_FALLBACK.get(crop_key, 3000)
        revenue    = _revenue_from_price(name, req.land_acres, req.irrigation, price)
        mandi = await _fetch_mandi_prices(crop_key, req.district or "Pune")
        if mandi is None:
            fp = MARKET_PRICES_FALLBACK.get(crop_key)
            if fp:
                mandi = {
                    "mandi_name"  : f"{req.district or 'Maharashtra'} APMC",
                    "commodity"   : name,
                    "variety"     : "Common",
                    "min_price"   : round(fp * 0.75),
                    "modal_price" : fp,
                    "max_price"   : round(fp * 1.25),
                    "arrival_date": "2023-24 average",
                    "all_mandis"  : [],
                    "source"      : "static_fallback",
                }
        harvest_days = HARVEST_DAYS.get(crop_key, 120)
        return {
            **c,
            "risk_score"          : risk["total"],
            "risk_level"          : risk["level"],
            "risk_breakdown"      : risk["breakdown"],
            "affordability"       : afford,
            "harvest_days"        : harvest_days,
            "mandi_prices"        : mandi,
            "explanation"         : _explain(name, req),
            "market_signal"       : market,
            "yield_estimate"      : yield_d,
            "revenue_estimate"    : revenue,
            "input_cost_estimate" : _input_cost_str(name, req.land_acres),
        }

    enriched = list(await asyncio.gather(*[_enrich(c) for c in candidates]))

    def _composite(crop: dict) -> float:
        ai_conf    = float(crop["confidence"])
        risk_inv   = 100.0 - float(crop["risk_score"])
        af         = crop["affordability"]
        if af["budget_ratio"] is None:
            budget_fit = 75.0
        else:
            r = af["budget_ratio"]
            if r < 0.50:   budget_fit = 100.0
            elif r < 0.85: budget_fit = 85.0
            elif r < 1.00: budget_fit = 70.0
            elif r < 1.20: budget_fit = 35.0
            else:          budget_fit = 10.0
        return ai_conf * 0.40 + risk_inv * 0.35 + budget_fit * 0.25

    ranked = sorted(enriched, key=_composite, reverse=True)[: req.top_n]
    for i, crop in enumerate(ranked):
        crop["rank"]            = i + 1
        crop["composite_score"] = round(_composite(crop), 1)

    weather = await weather_svc.fetch_or_use(req.temperature, req.humidity, req.rainfall)
    return {"crops": ranked, "total": len(ranked), "weather": weather}


@app.get("/mandi-prices/{district}/{crop}")
async def mandi_prices(district: str, crop: str):
    live = await _fetch_mandi_prices(crop.lower(), district)
    if live:
        return live
    fp = MARKET_PRICES_FALLBACK.get(crop.lower())
    if not fp:
        raise HTTPException(404, f"No price data for {crop}")
    return {
        "mandi_name"  : f"{district} APMC",
        "commodity"   : crop.title(),
        "variety"     : "Common",
        "min_price"   : round(fp * 0.75),
        "modal_price" : fp,
        "max_price"   : round(fp * 1.25),
        "arrival_date": "2023-24 average",
        "all_mandis"  : [],
        "source"      : "static_fallback",
    }


@app.post("/diagnose-crop-image")
async def diagnose(
    file     : UploadFile = File(...),
    crop_name: str        = Form("Unknown"),
):
    return await disease_det.diagnose(await file.read(), crop_name)


@app.get("/pest-alerts/{district}/{crop}")
async def pest_alerts(district: str, crop: str, season: str = "Kharif"):
    weather  = await weather_svc.fetch(district)
    w_alerts = pest_eng.weather_alerts(crop, weather, season)
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """SELECT pest, severity, COUNT(*) as cnt FROM sightings
           WHERE LOWER(district)=LOWER(?) AND LOWER(crop)=LOWER(?)
             AND ts > datetime('now','-7 days')
           GROUP BY pest, severity ORDER BY cnt DESC""",
        (district, crop),
    ).fetchall()
    conn.close()
    c_alerts = [{
        "pest_name"      : pest,
        "crop"           : crop,
        "severity"       : sev,
        "alert_type"     : "community",
        "report_count"   : cnt,
        "trigger_reason" : f"{cnt} farmer(s) in {district} reported {pest} in last 7 days",
        "action"         : pest_eng.action(pest),
        "organic"        : pest_eng.organic(pest),
        "days_until_peak": 3,
        "time_posted"    : "Recent",
    } for (pest, sev, cnt) in rows]
    return {"alerts": w_alerts + c_alerts, "district": district, "crop": crop}


@app.post("/report-sighting")
def report_sighting(req: SightingRequest):
    if not req.pest.strip():
        raise HTTPException(400, "Pest name required")
    if req.severity.upper() not in ("LOW", "MEDIUM", "HIGH"):
        raise HTTPException(400, "Severity must be LOW, MEDIUM, or HIGH")
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO sightings (district, crop, pest, severity) VALUES (?,?,?,?)",
        (req.district.strip(), req.crop.strip(), req.pest.strip(), req.severity.upper()),
    )
    conn.commit()
    n = conn.execute(
        "SELECT COUNT(*) FROM sightings WHERE LOWER(district)=LOWER(?) AND LOWER(crop)=LOWER(?)",
        (req.district, req.crop),
    ).fetchone()[0]
    conn.close()
    return {"success": True, "message": "Sighting recorded", "farmers_alerted": n}


@app.get("/crop-recommender-meta")
def crop_recommender_meta():
    return {"districts": crop_rec.districts, "seasons": crop_rec.seasons, "source": crop_rec.source}


@app.get("/market-prices/{crop}")
async def get_market_price(crop: str):
    live = await _fetch_agmarknet_modal(crop.lower())
    if live:
        return {"crop": crop, "price": live, "source": "agmarknet_live", "unit": "₹/quintal"}
    fb = MARKET_PRICES_FALLBACK.get(crop.lower())
    return {"crop": crop, "price": fb, "source": "static_fallback", "unit": "₹/quintal"}


# ══════════════════════════════════════════════════════════════════════
#  Helper functions
# ══════════════════════════════════════════════════════════════════════

def _yield_estimate(crop: str, land: float, irrigation: str) -> str:
    d = YIELD_BENCHMARKS.get(crop.lower())
    if not d:
        return "Data not available"
    f  = IRRIGATION_YIELD_FACTOR.get(irrigation, 1.0)
    lo = int(d["low"]  * land * f)
    hi = int(d["high"] * land * f)
    if d["high"] > 5000:
        return f"{lo/1000:.1f}–{hi/1000:.1f} tonnes"
    return f"{lo}–{hi} kg  ({lo//100}–{hi//100} qtl)"


def _revenue_from_price(crop: str, land: float, irrigation: str, price: float) -> str:
    d = YIELD_BENCHMARKS.get(crop.lower())
    if not d:
        return "N/A"
    f     = IRRIGATION_YIELD_FACTOR.get(irrigation, 1.0)
    lo_kg = d["low"]  * land * f
    hi_kg = d["high"] * land * f
    return f"₹{int(lo_kg/100*price):,}–₹{int(hi_kg/100*price):,}"


def _input_cost_str(crop: str, land: float) -> str:
    d = INPUT_COSTS.get(crop.lower(), DEFAULT_INPUT_COST)
    return f"₹{int(d['total'] * land):,}"


def _market_info(crop: str) -> dict:
    crop_key = crop.lower()
    mkt      = MARKET_DATA.get(crop_key)
    msp      = _msp_cache.get(crop_key) or MSP_FALLBACK.get(crop_key)
    if mkt:
        trend, demand, oversupply, _ = mkt
    else:
        trend, demand, oversupply = "STABLE", "MEDIUM", False
    return {
        "price_trend"    : trend,
        "demand_level"   : demand,
        "oversupply_risk": oversupply,
        "msp_price"      : msp or MARKET_PRICES_FALLBACK.get(crop_key),
    }


def _explain(crop: str, req: CropRequest) -> list[str]:
    pts  = []
    prof = CROP_PROFILES.get(crop.lower(), {})
    tlo, thi = prof.get("temp", (20, 33))
    rlo, rhi = prof.get("rain", (400, 1200))
    if req.N >= 70:
        pts.append(f"Good nitrogen levels ({req.N:.0f} mg/kg) support strong {crop} growth")
    else:
        pts.append(f"Moderate nitrogen — consider urea top-dressing for {crop}")
    if tlo <= req.temperature <= thi:
        pts.append(f"Current temperature ({req.temperature:.0f}°C) is ideal for {crop}")
    else:
        pts.append(f"Temperature ({req.temperature:.0f}°C) is outside ideal range; monitor closely")
    if rlo <= req.rainfall <= rhi:
        pts.append(f"Rainfall ({req.rainfall:.0f} mm) meets {crop}'s water requirements")
    elif req.rainfall < rlo:
        pts.append(f"Low rainfall ({req.rainfall:.0f} mm) — irrigation essential for {crop}")
    else:
        pts.append(f"High rainfall ({req.rainfall:.0f} mm) — ensure good drainage for {crop}")
    seasons = prof.get("seasons", [])
    if req.season in seasons or not seasons:
        pts.append(f"{req.season} season aligns with {crop}'s optimal growing cycle")
    else:
        pts.append(f"{crop} can be grown in {req.season} but performs best in {'/'.join(seasons)}")
    return pts[:4]

DISTRICT_DATA: dict[str, dict] = {
    # ── Vidarbha ─────────────────────────────────────────────────────
    "Nagpur": dict(
        region="Vidarbha", typical_soil_type="Black Soil",
        npk_range=dict(N_low=60,N_high=85,P_low=50,P_high=80,K_low=80,K_high=130),
        avg_ph_range=dict(low=7.0,high=8.5), avg_annual_rainfall_mm=1034,
        primary_season="Kharif", weather_city="Nagpur",
        common_crops=["Cotton","Soybean","Orange","Wheat","Pigeonpeas"],
    ),
    "Wardha": dict(
        region="Vidarbha", typical_soil_type="Black Soil",
        npk_range=dict(N_low=55,N_high=80,P_low=45,P_high=75,K_low=75,K_high=120),
        avg_ph_range=dict(low=7.0,high=8.5), avg_annual_rainfall_mm=920,
        primary_season="Kharif", weather_city="Wardha",
        common_crops=["Cotton","Soybean","Wheat","Pigeonpeas"],
    ),
    "Amravati": dict(
        region="Vidarbha", typical_soil_type="Black Soil",
        npk_range=dict(N_low=55,N_high=80,P_low=45,P_high=75,K_low=75,K_high=120),
        avg_ph_range=dict(low=7.0,high=8.5), avg_annual_rainfall_mm=870,
        primary_season="Kharif", weather_city="Amravati",
        common_crops=["Cotton","Soybean","Orange","Pigeonpeas"],
    ),
    "Akola": dict(
        region="Vidarbha", typical_soil_type="Black Soil",
        npk_range=dict(N_low=55,N_high=78,P_low=48,P_high=75,K_low=78,K_high=118),
        avg_ph_range=dict(low=7.2,high=8.6), avg_annual_rainfall_mm=790,
        primary_season="Kharif", weather_city="Akola",
        common_crops=["Cotton","Soybean","Wheat","Pigeonpeas"],
    ),
    "Washim": dict(
        region="Vidarbha", typical_soil_type="Black Soil",
        npk_range=dict(N_low=52,N_high=75,P_low=44,P_high=70,K_low=72,K_high=115),
        avg_ph_range=dict(low=7.2,high=8.5), avg_annual_rainfall_mm=760,
        primary_season="Kharif", weather_city="Washim",
        common_crops=["Cotton","Soybean","Pigeonpeas","Mungbean"],
    ),
    "Buldhana": dict(
        region="Vidarbha", typical_soil_type="Black Soil",
        npk_range=dict(N_low=52,N_high=78,P_low=46,P_high=72,K_low=74,K_high=116),
        avg_ph_range=dict(low=7.0,high=8.5), avg_annual_rainfall_mm=800,
        primary_season="Kharif", weather_city="Buldhana",
        common_crops=["Cotton","Soybean","Wheat","Orange"],
    ),
    "Yavatmal": dict(
        region="Vidarbha", typical_soil_type="Black Soil",
        npk_range=dict(N_low=58,N_high=82,P_low=48,P_high=76,K_low=76,K_high=122),
        avg_ph_range=dict(low=7.0,high=8.4), avg_annual_rainfall_mm=920,
        primary_season="Kharif", weather_city="Yavatmal",
        common_crops=["Cotton","Soybean","Pigeonpeas","Wheat"],
    ),
    "Chandrapur": dict(
        region="Vidarbha", typical_soil_type="Red Soil",
        npk_range=dict(N_low=40,N_high=65,P_low=25,P_high=50,K_low=55,K_high=90),
        avg_ph_range=dict(low=6.0,high=7.5), avg_annual_rainfall_mm=1250,
        primary_season="Kharif", weather_city="Chandrapur",
        common_crops=["Rice","Cotton","Soybean","Maize"],
    ),
    "Gadchiroli": dict(
        region="Vidarbha", typical_soil_type="Red Soil",
        npk_range=dict(N_low=35,N_high=60,P_low=20,P_high=45,K_low=50,K_high=85),
        avg_ph_range=dict(low=5.5,high=7.0), avg_annual_rainfall_mm=1500,
        primary_season="Kharif", weather_city="Gadchiroli",
        common_crops=["Rice","Maize","Sorghum"],
    ),
    "Gondia": dict(
        region="Vidarbha", typical_soil_type="Alluvial Soil",
        npk_range=dict(N_low=65,N_high=95,P_low=38,P_high=65,K_low=80,K_high=125),
        avg_ph_range=dict(low=6.5,high=7.8), avg_annual_rainfall_mm=1350,
        primary_season="Kharif", weather_city="Gondia",
        common_crops=["Rice","Wheat","Soybean"],
    ),
    "Bhandara": dict(
        region="Vidarbha", typical_soil_type="Alluvial Soil",
        npk_range=dict(N_low=68,N_high=98,P_low=40,P_high=68,K_low=82,K_high=128),
        avg_ph_range=dict(low=6.5,high=7.8), avg_annual_rainfall_mm=1320,
        primary_season="Kharif", weather_city="Bhandara",
        common_crops=["Rice","Wheat","Maize","Soybean"],
    ),
    "Chhatrapati Sambhajinagar": dict(
        region="Marathwada", typical_soil_type="Black Soil",
        npk_range=dict(N_low=50,N_high=75,P_low=40,P_high=68,K_low=70,K_high=110),
        avg_ph_range=dict(low=7.2,high=8.6), avg_annual_rainfall_mm=710,
        primary_season="Kharif", weather_city="Aurangabad",
        common_crops=["Cotton","Soybean","Sugarcane","Pigeonpeas","Wheat"],
    ),
    "Dharashiv": dict(
        region="Marathwada", typical_soil_type="Black Soil",
        npk_range=dict(N_low=48,N_high=72,P_low=38,P_high=65,K_low=68,K_high=108),
        avg_ph_range=dict(low=7.2,high=8.6), avg_annual_rainfall_mm=670,
        primary_season="Kharif", weather_city="Osmanabad",
        common_crops=["Soybean","Pigeonpeas","Cotton","Sugarcane"],
    ),
    "Beed": dict(
        region="Marathwada", typical_soil_type="Black Soil",
        npk_range=dict(N_low=48,N_high=73,P_low=40,P_high=66,K_low=68,K_high=108),
        avg_ph_range=dict(low=7.2,high=8.6), avg_annual_rainfall_mm=680,
        primary_season="Kharif", weather_city="Beed",
        common_crops=["Sugarcane","Cotton","Soybean","Pomegranate"],
    ),
    "Hingoli": dict(
        region="Marathwada", typical_soil_type="Black Soil",
        npk_range=dict(N_low=50,N_high=74,P_low=42,P_high=68,K_low=70,K_high=112),
        avg_ph_range=dict(low=7.0,high=8.4), avg_annual_rainfall_mm=830,
        primary_season="Kharif", weather_city="Hingoli",
        common_crops=["Soybean","Cotton","Pigeonpeas","Wheat"],
    ),
    "Jalna": dict(
        region="Marathwada", typical_soil_type="Black Soil",
        npk_range=dict(N_low=50,N_high=74,P_low=42,P_high=68,K_low=70,K_high=110),
        avg_ph_range=dict(low=7.2,high=8.5), avg_annual_rainfall_mm=720,
        primary_season="Kharif", weather_city="Jalna",
        common_crops=["Cotton","Soybean","Mungbean","Wheat"],
    ),
    "Latur": dict(
        region="Marathwada", typical_soil_type="Black Soil",
        npk_range=dict(N_low=48,N_high=72,P_low=40,P_high=66,K_low=66,K_high=106),
        avg_ph_range=dict(low=7.2,high=8.6), avg_annual_rainfall_mm=680,
        primary_season="Kharif", weather_city="Latur",
        common_crops=["Soybean","Pigeonpeas","Sugarcane","Cotton"],
    ),
    "Nanded": dict(
        region="Marathwada", typical_soil_type="Black Soil",
        npk_range=dict(N_low=52,N_high=76,P_low=44,P_high=70,K_low=72,K_high=112),
        avg_ph_range=dict(low=7.0,high=8.5), avg_annual_rainfall_mm=860,
        primary_season="Kharif", weather_city="Nanded",
        common_crops=["Soybean","Cotton","Sugarcane","Banana"],
    ),
    "Parbhani": dict(
        region="Marathwada", typical_soil_type="Black Soil",
        npk_range=dict(N_low=50,N_high=74,P_low=42,P_high=68,K_low=70,K_high=110),
        avg_ph_range=dict(low=7.2,high=8.6), avg_annual_rainfall_mm=760,
        primary_season="Kharif", weather_city="Parbhani",
        common_crops=["Soybean","Cotton","Pigeonpeas","Wheat"],
    ),
    "Pune": dict(
        region="Western Maharashtra", typical_soil_type="Black Soil",
        npk_range=dict(N_low=55,N_high=80,P_low=40,P_high=70,K_low=70,K_high=115),
        avg_ph_range=dict(low=6.5,high=8.0), avg_annual_rainfall_mm=725,
        primary_season="Kharif", weather_city="Pune",
        common_crops=["Sugarcane","Grapes","Onion","Wheat","Tomato"],
    ),
    "Nashik": dict(
        region="Northern Maharashtra", typical_soil_type="Red Soil",
        npk_range=dict(N_low=30,N_high=55,P_low=18,P_high=40,K_low=48,K_high=82),
        avg_ph_range=dict(low=5.8,high=7.2), avg_annual_rainfall_mm=680,
        primary_season="Kharif", weather_city="Nashik",
        common_crops=["Grapes","Onion","Tomato","Wheat","Maize"],
    ),
    "Ahilyanagar": dict(
        region="Western Maharashtra", typical_soil_type="Black Soil",
        npk_range=dict(N_low=52,N_high=78,P_low=42,P_high=68,K_low=72,K_high=112),
        avg_ph_range=dict(low=6.8,high=8.2), avg_annual_rainfall_mm=590,
        primary_season="Kharif", weather_city="Ahmednagar",
        common_crops=["Sugarcane","Onion","Cotton","Pomegranate"],
    ),
    "Solapur": dict(
        region="Western Maharashtra", typical_soil_type="Black Soil",
        npk_range=dict(N_low=45,N_high=70,P_low=38,P_high=62,K_low=65,K_high=105),
        avg_ph_range=dict(low=7.0,high=8.5), avg_annual_rainfall_mm=540,
        primary_season="Kharif", weather_city="Solapur",
        common_crops=["Pomegranate","Sugarcane","Onion","Sorghum"],
    ),
    "Satara": dict(
        region="Western Maharashtra", typical_soil_type="Loamy Soil",
        npk_range=dict(N_low=60,N_high=88,P_low=42,P_high=70,K_low=75,K_high=118),
        avg_ph_range=dict(low=6.2,high=7.8), avg_annual_rainfall_mm=780,
        primary_season="Kharif", weather_city="Satara",
        common_crops=["Sugarcane","Onion","Wheat","Maize","Tomato"],
    ),
    "Sangli": dict(
        region="Western Maharashtra", typical_soil_type="Black Soil",
        npk_range=dict(N_low=55,N_high=82,P_low=42,P_high=70,K_low=72,K_high=115),
        avg_ph_range=dict(low=7.0,high=8.2), avg_annual_rainfall_mm=520,
        primary_season="Kharif", weather_city="Sangli",
        common_crops=["Sugarcane","Grapes","Turmeric","Onion","Sorghum"],
    ),
    "Kolhapur": dict(
        region="Western Maharashtra", typical_soil_type="Loamy Soil",
        npk_range=dict(N_low=65,N_high=95,P_low=45,P_high=75,K_low=80,K_high=128),
        avg_ph_range=dict(low=5.8,high=7.5), avg_annual_rainfall_mm=1100,
        primary_season="Kharif", weather_city="Kolhapur",
        common_crops=["Sugarcane","Rice","Groundnut","Soybean"],
    ),
    "Raigad": dict(
        region="Konkan", typical_soil_type="Laterite Soil",
        npk_range=dict(N_low=22,N_high=45,P_low=12,P_high=28,K_low=35,K_high=65),
        avg_ph_range=dict(low=5.0,high=6.5), avg_annual_rainfall_mm=2500,
        primary_season="Kharif", weather_city="Alibaug",
        common_crops=["Rice","Coconut","Mango","Cashew","Banana"],
    ),
    "Ratnagiri": dict(
        region="Konkan", typical_soil_type="Laterite Soil",
        npk_range=dict(N_low=20,N_high=42,P_low=10,P_high=25,K_low=32,K_high=60),
        avg_ph_range=dict(low=5.0,high=6.5), avg_annual_rainfall_mm=3000,
        primary_season="Kharif", weather_city="Ratnagiri",
        common_crops=["Alphonso Mango","Cashew","Coconut","Rice"],
    ),
    "Sindhudurg": dict(
        region="Konkan", typical_soil_type="Laterite Soil",
        npk_range=dict(N_low=20,N_high=42,P_low=10,P_high=26,K_low=33,K_high=62),
        avg_ph_range=dict(low=5.0,high=6.5), avg_annual_rainfall_mm=3200,
        primary_season="Kharif", weather_city="Sindhudurg",
        common_crops=["Coconut","Cashew","Rice","Mango"],
    ),
    "Thane": dict(
        region="Konkan", typical_soil_type="Laterite Soil",
        npk_range=dict(N_low=25,N_high=48,P_low=12,P_high=30,K_low=38,K_high=68),
        avg_ph_range=dict(low=5.2,high=6.8), avg_annual_rainfall_mm=2600,
        primary_season="Kharif", weather_city="Thane",
        common_crops=["Rice","Maize","Vegetables","Coconut"],
    ),
    "Palghar": dict(
        region="Konkan", typical_soil_type="Laterite Soil",
        npk_range=dict(N_low=22,N_high=46,P_low=11,P_high=28,K_low=36,K_high=66),
        avg_ph_range=dict(low=5.2,high=6.8), avg_annual_rainfall_mm=2800,
        primary_season="Kharif", weather_city="Palghar",
        common_crops=["Rice","Vegetables","Banana","Coconut"],
    ),
    "Mumbai suburban": dict(
        region="Konkan", typical_soil_type="Laterite Soil",
        npk_range=dict(N_low=20,N_high=40,P_low=10,P_high=25,K_low=30,K_high=58),
        avg_ph_range=dict(low=5.0,high=6.5), avg_annual_rainfall_mm=2400,
        primary_season="Kharif", weather_city="Mumbai",
        common_crops=["Vegetables","Rice"],
    ),
    "Dhule": dict(
        region="Northern Maharashtra", typical_soil_type="Red Soil",
        npk_range=dict(N_low=25,N_high=50,P_low=15,P_high=35,K_low=42,K_high=75),
        avg_ph_range=dict(low=6.5,high=8.0), avg_annual_rainfall_mm=575,
        primary_season="Kharif", weather_city="Dhule",
        common_crops=["Maize","Cotton","Onion","Wheat","Groundnut"],
    ),
    "Nandurbar": dict(
        region="Northern Maharashtra", typical_soil_type="Red Soil",
        npk_range=dict(N_low=25,N_high=48,P_low=14,P_high=32,K_low=40,K_high=72),
        avg_ph_range=dict(low=6.0,high=7.8), avg_annual_rainfall_mm=900,
        primary_season="Kharif", weather_city="Nandurbar",
        common_crops=["Maize","Cotton","Sorghum","Banana"],
    ),
    "Jalgaon": dict(
        region="Northern Maharashtra", typical_soil_type="Black Soil",
        npk_range=dict(N_low=55,N_high=80,P_low=44,P_high=72,K_low=74,K_high=118),
        avg_ph_range=dict(low=7.0,high=8.4), avg_annual_rainfall_mm=680,
        primary_season="Kharif", weather_city="Jalgaon",
        common_crops=["Banana","Cotton","Maize","Wheat","Onion"],
    ),
}