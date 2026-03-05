PEST_DB = {
    "Whitefly":        ("Apply imidacloprid 0.3ml/L. Remove infested leaves. Install yellow traps.",
                        "Neem oil 5ml/L every 7 days"),
    "Brown Planthopper":("Drain excess water. Apply buprofezin 15% SC.",
                         "Release spider predators, use light traps"),
    "Fall Armyworm":   ("Apply chlorantraniliprole. Scout every 3 days.",
                        "Apply Bt-based biopesticide early"),
    "Powdery Mildew":  ("Apply propiconazole. Remove infected parts.",
                        "Diluted milk (1:9) or baking soda spray"),
    "Aphid":           ("Apply dimethoate if >20/tiller.", "Soap solution spray"),
    "Late Blight":     ("Apply metalaxyl urgently. Destroy infected plants.",
                        "Copper sulfate spray"),
}

PEST_RULES = [
    ("Cotton","Kharif",28,35,70,"Whitefly",     "HIGH",3),
    ("Rice",  "Kharif",25,30,85,"Brown Planthopper","HIGH",5),
    ("Wheat", "Rabi",   8,15,75,"Powdery Mildew","MEDIUM",7),
    ("Maize", "Kharif",28,38,50,"Fall Armyworm","HIGH",2),
    ("Tomato","Kharif",15,25,80,"Late Blight",  "HIGH",3),
    ("Potato","Kharif",15,25,80,"Late Blight",  "HIGH",3),
]

class PestEngine:
    def weather_alerts(self, crop: str, weather: dict) -> list:
        temp, hum = weather.get("temperature",28), weather.get("humidity",70)
        alerts = []
        for c,s,tlo,thi,hmin,pest,sev,days in PEST_RULES:
            if c.lower() != crop.lower(): continue
            if tlo <= temp <= thi and hum >= hmin:
                a,o = PEST_DB.get(pest,("Consult officer","Neem spray"))
                alerts.append({
                    "pest_name":pest,"crop":crop,"severity":sev,
                    "alert_type":"weather_prediction","report_count":0,
                    "trigger_reason":f"Temp {temp}°C + Humidity {hum}% match {pest} outbreak conditions",
                    "action":a,"organic":o,"days_until_peak":days,"time_posted":"Now",
                })
        return alerts

    def action(self, pest: str) -> str:
        return PEST_DB.get(pest,("Consult agricultural officer","Neem oil spray"))[0]

    def organic(self, pest: str) -> str:
        return PEST_DB.get(pest,("","Neem oil spray"))[1]
