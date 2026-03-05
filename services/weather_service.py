import httpx

class WeatherService:
    BASE = "https://api.openweathermap.org/data/2.5"

    def __init__(self, api_key: str):
        self.key   = api_key
        self._cache: dict = {}

    async def fetch(self, district: str) -> dict:
        if district in self._cache:
            return self._cache[district]
        if not self.key:
            return self._fallback(district)
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r1 = await c.get(f"{self.BASE}/weather",
                    params={"q":f"{district},IN","appid":self.key,"units":"metric"})
                r1.raise_for_status()
                d  = r1.json()
                r2 = await c.get(f"{self.BASE}/forecast",
                    params={"q":f"{district},IN","appid":self.key,"units":"metric","cnt":40})
                r2.raise_for_status()
                rain = sum(i.get("rain",{}).get("3h",0) for i in r2.json()["list"])
            res = {
                "district":district,
                "temperature": round(d["main"]["temp"],1),
                "humidity"   : round(d["main"]["humidity"],1),
                "rainfall"   : round(rain, 1),
                "condition"  : d["weather"][0]["description"].title(),
                "drought_flag": rain < 20,
                "flood_flag"  : rain > 200,
            }
            self._cache[district] = res
            return res
        except Exception:
            return self._fallback(district)

    async def fetch_or_use(self, temp, hum, rain) -> dict:
        return {"temperature":temp,"humidity":hum,"rainfall":rain,
                "drought_flag":rain<20,"flood_flag":rain>200,"condition":"User provided"}

    def _fallback(self, district: str) -> dict:
        D = {"Nagpur":(29.5,72,820),"Ludhiana":(25,60,680),"Patna":(30,75,1050)}
        t,h,r = D.get(district,(28,70,750))
        return {"district":district,"temperature":t,"humidity":h,"rainfall":r,
                "condition":"Partly Cloudy","drought_flag":r<100,"flood_flag":False}

