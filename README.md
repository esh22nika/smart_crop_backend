# Smart Crop DSS — FastAPI Backend

Backend service for the **Smart Crop Decision Support System (DSS)**.
This API powers the Flutter mobile application by providing crop recommendations, soil analysis, disease detection, pest alerts, and weather data.

The backend integrates multiple machine learning models and rule-based systems to provide intelligent agricultural insights.

---

# Models and Services Used (as of demo 1 version)

| Component               | Model / Service                        | Purpose                                         |
| ----------------------- | -------------------------------------- | ----------------------------------------------- |
| Crop Recommendation     | `Arko007/agromind-crop-recommendation` | Predict best crops from soil and climate data   |
| Plant Disease Detection | `Arko007/nfnet-f1-plant-disease`       | Identify plant diseases from leaf images        |
| Soil Classification     | `EricEchemane/Pixsoil`                 | Classify soil type from soil images             |
| Weather Data            | OpenWeatherMap API                     | Fetch live weather conditions                   |
| Risk Engine             | Rule-based logic                       | Estimate crop risk levels                       |
| Pest Alerts             | Rule-based + SQLite                    | Predict pest outbreaks and store farmer reports |

The system combines machine learning predictions, rule-based logic, and external APIs to support agricultural decision-making.

---

# Python Version

This project requires:

```
Python 3.11
```

Using other versions may cause compatibility issues with TensorFlow or PyTorch models.

---

# Project Structure

```
backend/
│
├── main.py
├── requirements.txt
├── Procfile
├── conversion.ipynb
├── .env
│
├── services/
│   ├── crop_recommender.py
│   ├── disease_detector.py
│   ├── soil_classifier.py
│   ├── weather_service.py
│   ├── pest_engine.py
│   └── risk_engine.py
│
├── models/
│   └── soil_model/
│
└── sightings.db
```

---

# Environment Variables

Create a `.env` file in the project root.

Example:

```
OPENWEATHER_API_KEY=your_openweathermap_api_key
HF_API_TOKEN=your_huggingface_token
```

### Purpose

| Variable            | Purpose                              |
| ------------------- | ------------------------------------ |
| OPENWEATHER_API_KEY | Fetch live weather data              |
| HF_API_TOKEN        | Download models from HuggingFace Hub |

---

# Install Dependencies

Create and activate a virtual environment.

### Windows

```
python -m venv venv
venv\Scripts\activate
```

### Mac / Linux

```
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

Dependencies include FastAPI, TensorFlow, PyTorch, HuggingFace Hub, timm, scikit-learn, Pillow, and httpx.

---

# Soil Model Setup (Pixsoil)

The soil classifier uses the **Pixsoil model**, which must be converted from TensorFlow.js format before running the backend.

A conversion notebook is included:

```
conversion.ipynb
```

## Step 1 — Run the notebook

Open and run:

```
conversion.ipynb
```

This notebook converts the Pixsoil TensorFlow.js model into a **TensorFlow SavedModel**.

It will generate a file:

```
soil_model.zip
```

---

## Step 2 — Extract the model

Create the following directory:

```
models/soil_model/
```

Extract the contents of the zip file into that folder.

Final structure should look like:

```
backend/
│
├── models/
│   └── soil_model/
│       ├── saved_model.pb
│       └── variables/
│           ├── variables.data-00000-of-00001
│           └── variables.index
```

---

## Step 3 — Verify model loading

When the server starts you should see:

```
Pixsoil model loaded
```

If the model is missing, the backend will fall back to a **color-based heuristic soil classifier**.

---

# Running the Server

Start the FastAPI server:

```
uvicorn main:app --reload
```

Server will run at:

```
http://127.0.0.1:8000
```

Interactive API documentation:

```
http://127.0.0.1:8000/docs
```

---

# Main API Endpoints

## Health Check

```
GET /health
```

Returns model loading status.

---

## Weather Data

```
GET /weather/{district}
```

Returns temperature, humidity, rainfall, and weather conditions.

---

## District Defaults

```
GET /district-defaults/{district}
```

Returns soil ranges, rainfall, and common crops for a district.

---

## Soil Analysis

```
POST /analyze-soil-image
```

Input:

- Soil image

Output:

- Soil type
- Estimated NPK values
- Recommended crops

---

## Crop Recommendation

```
POST /recommend-crops
```

Input parameters:

```
N
P
K
temperature
humidity
ph
rainfall
season
land_acres
budget
```

Output:

- Top 3 crop recommendations
- Risk score
- Market data
- Yield estimate
- Revenue estimate

---

## Plant Disease Detection

```
POST /diagnose-crop-image
```

Input:

- Leaf image
- Crop name

Output:

- Disease name
- Severity
- Treatment recommendation
- Organic treatment option

---

## Pest Alerts

```
GET /pest-alerts/{district}/{crop}
```

Returns pest outbreak predictions based on weather and community reports.

---

## Report Pest Sighting

```
POST /report-sighting
```

Stores farmer-reported pest sightings in SQLite.

---

# Database

The backend automatically creates:

```
sightings.db
```

SQLite is used to store farmer pest reports and generate community alerts.

---

# Flutter Integration

The Flutter frontend communicates with these endpoints.

| Screen             | Endpoint                               |
| ------------------ | -------------------------------------- |
| Wizard Step 1      | `/district-defaults`                   |
| Wizard Step 1      | `/weather/{district}`                  |
| Wizard Step 2      | `/analyze-soil-image`                  |
| Wizard Submit      | `/recommend-crops`                     |
| Results Screen     | Uses data from recommendation endpoint |
| Pest Alerts Screen | `/pest-alerts/{district}/{crop}`       |
| Disease Scanner    | `/diagnose-crop-image`                 |
| Report Sighting    | `/report-sighting`                     |

---

# Deployment

For deployment (e.g., Render), the repository includes a Procfile:

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

# Testing the API

Check server health:

```
http://127.0.0.1:8000/health
```

Test endpoints via Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

# License

This project is intended for educational and research purposes.
