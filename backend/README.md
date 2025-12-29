# TB Risk Detection System – Backend

This backend exposes an API for **early tuberculosis (TB) risk screening** using a machine learning model and location-based hospital recommendations.

> ⚠️ **Disclaimer:** This system is for screening support only and does not diagnose tuberculosis.

---

## Tech Stack

- Python 3.9
- Flask
- scikit-learn
- joblib
- NumPy

---

## Project Structure
```
backend/
├── app.py
├── requirements.txt
├── README.md
├── model/
│ └── tb_risk_new_logic_model.joblib
└── data/
└── hospitals.csv
```

---

## Running Locally

### 1. Create and activate virtual environment

python -m venv venv
venv\Scripts\activate


### 2. Install dependencies

pip install -r requirements.txt


### 3. Start the server

python app.py


Server runs at:

http://127.0.0.1:5000


---

## API Reference

### POST `/predict`

---

## Request Format (JSON)

- 12 binary symptoms (`0` or `1`)
- Optional district and state for hospital recommendations

```
{
  "symptom_1": 0,
  "symptom_2": 1,
  "symptom_3": 0,
  "symptom_4": 1,
  "symptom_5": 0,
  "symptom_6": 0,
  "symptom_7": 1,
  "symptom_8": 0,
  "symptom_9": 0,
  "symptom_10": 1,
  "symptom_11": 0,
  "symptom_12": 0,
  "district": "howrah",
  "state": "west bengal"
}
```


---

## Response Format (JSON)

```
{
  "risk_level": "Medium",
  "confidence_percent": 50.12,
  "hospitals": [],
  "disclaimer": "This system provides early TB risk screening only. It does not diagnose tuberculosis."
}


```
---

## Notes

- Risk levels: **Low | Medium | High**
- Confidence is model confidence, not diagnosis probability
- Hospitals list is empty for Low risk
- Disclaimer must always be displayed

---

## Error Handling

### 400 – Invalid Input

{ "error": "Missing feature: symptom_3" }


### 500 – Server Error

{
"error": "Server error. Please try again later.",
"disclaimer": "This system provides early TB risk screening only."
}

---

## Hospital Recommendation Logic

- Triggered only for **Medium and High** risk
- District-first matching
- State fallback if district match fails
- Maximum of **5 hospitals**

---

