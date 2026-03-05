class RiskEngine:
    IDEAL_TEMP = {
        "rice":(20,35),"wheat":(15,25),"cotton":(25,35),"maize":(18,32),
        "soybean":(20,30),"groundnut":(25,35),"sugarcane":(20,35),
    }
    MARKET = {
        "rice"    :("STABLE","HIGH",False),
        "wheat"   :("STABLE","HIGH",False),
        "cotton"  :("DOWN",  "MEDIUM",True),
        "soybean" :("UP",    "HIGH",  False),
        "maize"   :("UP",    "HIGH",  False),
        "mango"   :("UP",    "HIGH",  False),
        "banana"  :("STABLE","HIGH",  False),
        "coconut" :("STABLE","MEDIUM",False),
        "coffee"  :("UP",    "MEDIUM",False),
    }

    def score(self, crop:str, season:str, temp:float, humidity:float,
              rainfall:float, land:float, budget=None) -> dict:
        w = self._weather(crop, temp, rainfall)
        m = self._market(crop)
        c = self._cost(crop, land, budget)
        p = self._pest(crop, season, temp, humidity)
        total = round(w*0.35 + m*0.30 + c*0.20 + p*0.15, 1)
        return {
            "total": total,
            "level": "LOW" if total<=30 else "MEDIUM" if total<=60 else "HIGH",
            "breakdown": {"weather":round(w,1),"market":round(m,1),
                          "input_cost":round(c,1),"pest":round(p,1)},
        }

    def _weather(self, crop, temp, rainfall):
        lo,hi = self.IDEAL_TEMP.get(crop.lower(),(20,33))
        s = 0 if lo<=temp<=hi else (30 if abs(temp-(lo+hi)/2)<8 else 70)
        if rainfall < 20:  s += 60
        if rainfall > 200: s += 50
        return min(s, 100)

    def _market(self, crop):
        t,d,o = self.MARKET.get(crop.lower(),("STABLE","MEDIUM",False))
        ts = {"UP":10,"STABLE":30,"DOWN":65}[t]
        ds = {"HIGH":0,"MEDIUM":25,"LOW":60}[d]
        return min((ts+ds+(40 if o else 0))/3, 100)

    def _cost(self, crop, land, budget):
        costs = {"rice":18000,"wheat":15000,"cotton":22000,"soybean":12000,
                 "maize":13000,"banana":20000,"mango":18000}
        cost = costs.get(crop.lower(), 15000) * land
        if not budget: return 40.0
        r = cost / budget
        return 10 if r<0.6 else 40 if r<0.8 else 70 if r<1.0 else 100

    def _pest(self, crop, season, temp, humidity):
        rules = [
            ("Cotton","Kharif",28,35,70,80.0),
            ("Rice",  "Kharif",25,30,85,80.0),
            ("Wheat", "Rabi",   8,15,75,45.0),
            ("Maize", "Kharif",28,38,50,80.0),
            ("Tomato","Kharif",15,20,80,80.0),
        ]
        for c,s,tlo,thi,hmin,risk in rules:
            if c.lower()==crop.lower() and s==season and tlo<=temp<=thi and humidity>=hmin:
                return risk
        return 10.0
