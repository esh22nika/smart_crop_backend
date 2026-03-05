from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from contextlib import asynccontextmanager
import sqlite3, os
from dotenv import load_dotenv

load_dotenv()

from services.weather_service   import WeatherService
from services.soil_classifier   import SoilClassifier
from services.crop_recommender  import CropRecommender
from services.disease_detector  import DiseaseDetector
from services.risk_engine       import RiskEngine
from services.pest_engine       import PestEngine

# ── Globals ───────────────────────────────────────────────────────────
weather_svc : WeatherService  = None
soil_clf    : SoilClassifier  = None
crop_rec    : CropRecommender = None
disease_det : DiseaseDetector = None
risk_eng    : RiskEngine      = None
pest_eng    : PestEngine      = None

def _init_db():
    conn = sqlite3.connect("sightings.db")
    conn.execute("""CREATE TABLE IF NOT EXISTS sightings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        district TEXT, crop TEXT, pest TEXT,
        severity TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit(); conn.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global weather_svc, soil_clf, crop_rec, disease_det, risk_eng, pest_eng
    print("🌱 Starting backend...")
    _init_db()
    weather_svc = WeatherService(os.getenv("OPENWEATHER_API_KEY", ""))
    soil_clf    = SoilClassifier()
    crop_rec    = CropRecommender(os.getenv("HF_API_TOKEN", ""))
    disease_det = DiseaseDetector(os.getenv("HF_API_TOKEN", ""))
    risk_eng    = RiskEngine()
    pest_eng    = PestEngine()
    print("✅ All services ready.")
    yield

app = FastAPI(title="Smart Crop DSS", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"], allow_headers=["*"])

# ── Pydantic schemas ──────────────────────────────────────────────────

class CropRequest(BaseModel):
    # These 7 are the EXACT inputs the Arko007 crop model needs
    N          : float = Field(..., ge=0,   le=140, description="Nitrogen kg/ha")
    P          : float = Field(..., ge=0,   le=145, description="Phosphorus kg/ha")
    K          : float = Field(..., ge=0,   le=205, description="Potassium kg/ha")
    temperature: float = Field(..., ge=0,   le=50)
    humidity   : float = Field(..., ge=0,   le=100)
    ph         : float = Field(..., ge=0,   le=14)
    rainfall   : float = Field(..., ge=0,   le=500)
    # These extras are for risk engine & revenue calc — not fed to ML model
    season     : str   = "Kharif"
    land_acres : float = 1.0
    budget     : Optional[float] = None

class SightingRequest(BaseModel):
    district: str; crop: str; pest: str; severity: str

# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "soil_model"  : soil_clf.source,
        "crop_model"  : crop_rec.source,
        "disease_model": disease_det.source,
    }

@app.get("/weather/{district}")
async def get_weather(district: str):
    return await weather_svc.fetch(district)

@app.get("/district-defaults/{district}")
def district_defaults(district: str):
    d = DISTRICT_DATA.get(district)
    if not d: raise HTTPException(404, f"No data for {district}")
    return d

@app.post("/analyze-soil-image")
async def analyze_soil(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Must be an image")
    return soil_clf.classify(await file.read())

@app.post("/recommend-crops")
async def recommend_crops(req: CropRequest):
    # Step 1 — run the HF crop model (7 numeric inputs only)
    top3 = crop_rec.recommend(
        req.N, req.P, req.K,
        req.temperature, req.humidity, req.ph, req.rainfall
    )
    # Step 2 — enrich each result with risk, yield, market data
    results = []
    for c in top3:
        risk    = risk_eng.score(c["crop_name"], req.season,
                                  req.temperature, req.humidity,
                                  req.rainfall, req.land_acres, req.budget)
        yield_d = YIELD_DATA.get(c["crop_name"].lower(), DEFAULT_YIELD)
        market  = MARKET_DATA.get(c["crop_name"].lower(), DEFAULT_MARKET)
        rev_lo  = yield_d["yield_low"]  * req.land_acres * yield_d["price"] / 100
        rev_hi  = yield_d["yield_high"] * req.land_acres * yield_d["price"] / 100
        results.append({
            **c,
            "risk_score"    : risk["total"],
            "risk_level"    : risk["level"],
            "risk_breakdown": risk["breakdown"],
            "explanation"   : _explain(c["crop_name"], req),
            "market"        : market,
            "yield_estimate": f"{yield_d['yield_low']}–{yield_d['yield_high']} kg/acre",
            "revenue_estimate": f"₹{int(rev_lo):,}–₹{int(rev_hi):,}",
        })
    weather = await weather_svc.fetch_or_use(req.temperature, req.humidity, req.rainfall)
    return {"top_3": results, "weather": weather}

@app.post("/diagnose-crop-image")
async def diagnose(
    file     : UploadFile = File(...),
    crop_name: str        = Form("Unknown"),
):
    return await disease_det.diagnose(await file.read(), crop_name)

@app.get("/pest-alerts/{district}/{crop}")
async def pest_alerts(district: str, crop: str):
    weather  = await weather_svc.fetch(district)
    w_alerts = pest_eng.weather_alerts(crop, weather)
    conn     = sqlite3.connect("sightings.db")
    rows     = conn.execute(
        "SELECT pest,severity,COUNT(*) FROM sightings "
        "WHERE district=? AND crop=? AND ts>datetime('now','-7 days') "
        "GROUP BY pest,severity", (district, crop)
    ).fetchall(); conn.close()
    c_alerts = [{"pest_name":r[0],"severity":r[1],"report_count":r[2],
                 "alert_type":"community","crop":crop,
                 "trigger_reason":f"{r[2]} farmer(s) reported in 7 days",
                 "action": pest_eng.action(r[0]),
                 "organic": pest_eng.organic(r[0])} for r in rows]
    return {"alerts": w_alerts + c_alerts}

@app.post("/report-sighting")
def report_sighting(req: SightingRequest):
    conn = sqlite3.connect("sightings.db")
    conn.execute("INSERT INTO sightings(district,crop,pest,severity) VALUES(?,?,?,?)",
                 (req.district, req.crop, req.pest, req.severity))
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM sightings WHERE district=? AND crop=?",
                     (req.district, req.crop)).fetchone()[0]
    conn.close()
    return {"success": True, "farmers_alerted": n}

# ── Helper: plain-language explanation ───────────────────────────────
def _explain(crop: str, req: CropRequest) -> list[str]:
    pts = [f"Soil NPK profile suits {crop} cultivation"]
    if req.rainfall > 150: pts.append(f"Rainfall ({int(req.rainfall)}mm) meets {crop}'s water needs")
    else: pts.append(f"Low rainfall — consider irrigation for {crop}")
    if 6.0 <= req.ph <= 7.5: pts.append(f"Soil pH ({req.ph}) is in ideal range for {crop}")
    pts.append(f"{req.season} season aligns with {crop}'s growing cycle")
    return pts[:3]

# ═══════════════════════════════════════════════════════════════════════
# Static data embedded in main.py (move to JSON files if preferred)
# ═══════════════════════════════════════════════════════════════════════

DISTRICT_DATA = {
    "Nagpur"  :{"typical_soil_type":"Black","npk_range":{"N_low":60,"N_high":80,"P_low":50,"P_high":80,"K_low":80,"K_high":120},"avg_ph_range":{"low":7.0,"high":8.5},"avg_annual_rainfall_mm":1034,"primary_season":"Kharif","common_crops":["Cotton","Soybean","Orange","Wheat"],"weather_city":"Nagpur"},
    "Pune"    :{"typical_soil_type":"Black","npk_range":{"N_low":55,"N_high":75,"P_low":40,"P_high":70,"K_low":70,"K_high":110},"avg_ph_range":{"low":6.5,"high":8.0},"avg_annual_rainfall_mm":720,"primary_season":"Kharif","common_crops":["Sugarcane","Grapes","Onion","Wheat"],"weather_city":"Pune"},
    "Nashik"  :{"typical_soil_type":"Red","npk_range":{"N_low":30,"N_high":50,"P_low":20,"P_high":40,"K_low":50,"K_high":80},"avg_ph_range":{"low":6.0,"high":7.5},"avg_annual_rainfall_mm":680,"primary_season":"Kharif","common_crops":["Grapes","Onion","Tomato"],"weather_city":"Nashik"},
    "Ludhiana":{"typical_soil_type":"Alluvial","npk_range":{"N_low":90,"N_high":120,"P_low":45,"P_high":70,"K_low":100,"K_high":140},"avg_ph_range":{"low":7.0,"high":8.0},"avg_annual_rainfall_mm":680,"primary_season":"Rabi","common_crops":["Wheat","Rice","Maize","Cotton"],"weather_city":"Ludhiana"},
    "Amritsar":{"typical_soil_type":"Alluvial","npk_range":{"N_low":85,"N_high":115,"P_low":40,"P_high":65,"K_low":95,"K_high":135},"avg_ph_range":{"low":7.0,"high":8.0},"avg_annual_rainfall_mm":680,"primary_season":"Rabi","common_crops":["Wheat","Rice","Sugarcane"],"weather_city":"Amritsar"},
    "Patna"   :{"typical_soil_type":"Alluvial","npk_range":{"N_low":70,"N_high":100,"P_low":35,"P_high":60,"K_low":80,"K_high":120},"avg_ph_range":{"low":6.5,"high":7.5},"avg_annual_rainfall_mm":1050,"primary_season":"Kharif","common_crops":["Rice","Wheat","Maize","Lentil"],"weather_city":"Patna"},
    "Varanasi":{"typical_soil_type":"Alluvial","npk_range":{"N_low":75,"N_high":105,"P_low":38,"P_high":62,"K_low":85,"K_high":125},"avg_ph_range":{"low":7.0,"high":8.0},"avg_annual_rainfall_mm":1000,"primary_season":"Kharif","common_crops":["Rice","Wheat","Sugarcane"],"weather_city":"Varanasi"},
    "Hyderabad":{"typical_soil_type":"Red","npk_range":{"N_low":25,"N_high":45,"P_low":15,"P_high":35,"K_low":45,"K_high":75},"avg_ph_range":{"low":5.5,"high":7.0},"avg_annual_rainfall_mm":780,"primary_season":"Kharif","common_crops":["Rice","Cotton","Maize","Groundnut"],"weather_city":"Hyderabad"},
    "Coimbatore":{"typical_soil_type":"Red","npk_range":{"N_low":20,"N_high":40,"P_low":12,"P_high":28,"K_low":40,"K_high":70},"avg_ph_range":{"low":5.5,"high":7.0},"avg_annual_rainfall_mm":700,"primary_season":"Kharif","common_crops":["Cotton","Groundnut","Maize","Coconut"],"weather_city":"Coimbatore"},
    "Jaipur"  :{"typical_soil_type":"Red","npk_range":{"N_low":15,"N_high":35,"P_low":10,"P_high":25,"K_low":30,"K_high":60},"avg_ph_range":{"low":7.0,"high":8.5},"avg_annual_rainfall_mm":530,"primary_season":"Kharif","common_crops":["Millet","Groundnut","Mustard","Wheat"],"weather_city":"Jaipur"},
    "Bhopal"  :{"typical_soil_type":"Black","npk_range":{"N_low":55,"N_high":75,"P_low":45,"P_high":70,"K_low":75,"K_high":115},"avg_ph_range":{"low":6.5,"high":8.0},"avg_annual_rainfall_mm":1150,"primary_season":"Kharif","common_crops":["Soybean","Wheat","Cotton","Maize"],"weather_city":"Bhopal"},
    "Ahmedabad":{"typical_soil_type":"Black","npk_range":{"N_low":50,"N_high":70,"P_low":40,"P_high":65,"K_low":70,"K_high":110},"avg_ph_range":{"low":7.0,"high":8.5},"avg_annual_rainfall_mm":780,"primary_season":"Kharif","common_crops":["Cotton","Groundnut","Wheat","Castor"],"weather_city":"Ahmedabad"},
}

YIELD_DATA = {
    "rice"    :{"yield_low":1400,"yield_high":2200,"price":2100,"input_cost":18000},
    "wheat"   :{"yield_low":1200,"yield_high":2000,"price":2200,"input_cost":15000},
    "cotton"  :{"yield_low":280, "yield_high":520, "price":6600,"input_cost":22000},
    "soybean" :{"yield_low":700, "yield_high":1100,"price":4000,"input_cost":12000},
    "maize"   :{"yield_low":1000,"yield_high":1800,"price":1800,"input_cost":13000},
    "groundnut":{"yield_low":600,"yield_high":1000,"price":5000,"input_cost":14000},
    "banana"  :{"yield_low":8000,"yield_high":14000,"price":1500,"input_cost":25000},
    "mango"   :{"yield_low":3000,"yield_high":6000,"price":3000,"input_cost":20000},
    "coconut" :{"yield_low":5000,"yield_high":9000,"price":1200,"input_cost":18000},
    "coffee"  :{"yield_low":400, "yield_high":700, "price":8000,"input_cost":30000},
    "jute"    :{"yield_low":1800,"yield_high":2800,"price":3500,"input_cost":14000},
    "lentil"  :{"yield_low":400, "yield_high":700, "price":5500,"input_cost":10000},
}
DEFAULT_YIELD = {"yield_low":600,"yield_high":1000,"price":3000,"input_cost":15000}

MARKET_DATA = {
    "rice"    :{"price_trend":"STABLE","demand_level":"HIGH",  "oversupply_risk":False},
    "wheat"   :{"price_trend":"STABLE","demand_level":"HIGH",  "oversupply_risk":False},
    "cotton"  :{"price_trend":"DOWN",  "demand_level":"MEDIUM","oversupply_risk":True},
    "soybean" :{"price_trend":"UP",    "demand_level":"HIGH",  "oversupply_risk":False},
    "maize"   :{"price_trend":"UP",    "demand_level":"HIGH",  "oversupply_risk":False},
    "mango"   :{"price_trend":"UP",    "demand_level":"HIGH",  "oversupply_risk":False},
    "banana"  :{"price_trend":"STABLE","demand_level":"HIGH",  "oversupply_risk":False},
    "coffee"  :{"price_trend":"UP",    "demand_level":"MEDIUM","oversupply_risk":False},
    "coconut" :{"price_trend":"STABLE","demand_level":"MEDIUM","oversupply_risk":False},
    "jute"    :{"price_trend":"STABLE","demand_level":"MEDIUM","oversupply_risk":False},
}
DEFAULT_MARKET = {"price_trend":"STABLE","demand_level":"MEDIUM","oversupply_risk":False}



