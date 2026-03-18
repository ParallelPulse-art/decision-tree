# 🚢 Titanic Survival Predictor

A Flask web application that predicts Titanic passenger survival using a Random Forest classifier (~84% accuracy).

## Files
| File | Purpose |
|------|---------|
| `app.py` | Flask web app + prediction API |
| `train_model.py` | Script to retrain the model |
| `model.pkl` | Pre-trained Random Forest model |
| `requirements.txt` | Python dependencies |

---

## Run Locally

```bash
pip install -r requirements.txt
python app.py
# → open http://localhost:5000
```

---

## 🚀 Deploy to Render (Free)

1. Push this folder to a GitHub repo
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your repo
4. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
5. Click **Deploy** — your app gets a public URL like `https://titanic-predictor.onrender.com`

---

## 🚀 Deploy to Railway (Free tier)

```bash
npm i -g @railway/cli
railway login
railway init
railway up
```

---

## 🚀 Deploy to Fly.io

```bash
# Install flyctl, then:
fly launch
fly deploy
```

---

## API

`POST /predict` with JSON body:

```json
{
  "pclass": 1,
  "sex": "female",
  "age": 28,
  "sibsp": 0,
  "parch": 0,
  "fare": 100.0,
  "embarked": "C"
}
```

Response:
```json
{
  "survived": true,
  "survival_probability": 0.99,
  "feature_importance": [
    { "feature": "Sex", "importance": 0.269 },
    ...
  ]
}
```

---

## Model Details
- **Algorithm:** Random Forest (100 trees)
- **Accuracy:** ~84%
- **Features:** Passenger class, sex, age, siblings/spouses, parents/children, fare, port of embarkation

## 🚢 Live Demo

👉 [Titanic Survival Predictor](https://decision-tree-txs3.onrender.com/)
