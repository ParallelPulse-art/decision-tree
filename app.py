from flask import Flask, request, jsonify, render_template_string
import pickle, numpy as np

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    data = pickle.load(f)
model = data['model']
le_sex = data['le_sex']
le_emb = data['le_emb']
age_median = data['age_median']
fare_median = data['fare_median']

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Titanic Survival Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Crimson+Pro:ital,wght@0,300;0,400;1,300&display=swap" rel="stylesheet"/>
<style>
  :root {
    --ink: #0d0d0d;
    --paper: #f5f0e8;
    --aged: #e8dfc8;
    --rust: #8b3a2f;
    --gold: #c9a84c;
    --deep: #1a2744;
    --mist: #6b7fa3;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--paper);
    color: var(--ink);
    font-family: 'Crimson Pro', Georgia, serif;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Ocean background */
  body::before {
    content: '';
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background:
      radial-gradient(ellipse at 20% 80%, rgba(26,39,68,0.12) 0%, transparent 60%),
      radial-gradient(ellipse at 80% 20%, rgba(139,58,47,0.07) 0%, transparent 50%),
      linear-gradient(170deg, var(--paper) 0%, #ede5d0 50%, #e0d5bc 100%);
    pointer-events: none; z-index: 0;
  }

  /* Grain texture */
  body::after {
    content: '';
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none; z-index: 0; opacity: 0.6;
  }

  .wrapper { position: relative; z-index: 1; max-width: 680px; margin: 0 auto; padding: 60px 24px 80px; }

  /* Header */
  header { text-align: center; margin-bottom: 56px; }
  .eyebrow {
    font-family: 'Crimson Pro', serif;
    font-size: 11px; letter-spacing: 0.35em; text-transform: uppercase;
    color: var(--mist); margin-bottom: 16px;
  }
  h1 {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: clamp(42px, 8vw, 72px);
    font-weight: 900; line-height: 0.92;
    color: var(--deep);
    margin-bottom: 6px;
  }
  h1 em { color: var(--rust); font-style: normal; }
  .subtitle {
    font-size: 17px; color: #6b6050; font-style: italic;
    margin-top: 14px; font-weight: 300;
  }
  .rule {
    display: flex; align-items: center; gap: 16px;
    margin: 28px auto 0; max-width: 320px;
  }
  .rule::before, .rule::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, transparent, var(--gold), transparent); }
  .rule-diamond { width: 6px; height: 6px; background: var(--gold); transform: rotate(45deg); flex-shrink: 0; }

  /* Form card */
  .card {
    background: rgba(255,255,255,0.55);
    border: 1px solid rgba(201,168,76,0.3);
    border-radius: 2px;
    padding: 44px 40px;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 40px rgba(26,39,68,0.08), inset 0 1px 0 rgba(255,255,255,0.8);
  }

  .section-title {
    font-family: 'Playfair Display', serif;
    font-size: 11px; letter-spacing: 0.3em; text-transform: uppercase;
    color: var(--gold); margin-bottom: 24px; margin-top: 32px;
    display: flex; align-items: center; gap: 12px;
  }
  .section-title:first-child { margin-top: 0; }
  .section-title::after { content: ''; flex: 1; height: 1px; background: rgba(201,168,76,0.25); }

  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .field { display: flex; flex-direction: column; gap: 7px; }
  .field.full { grid-column: 1 / -1; }

  label {
    font-size: 11px; letter-spacing: 0.2em; text-transform: uppercase;
    color: var(--mist); font-family: 'Crimson Pro', serif;
  }

  input, select {
    background: rgba(245,240,232,0.8);
    border: 1px solid rgba(107,127,163,0.25);
    border-radius: 1px;
    padding: 11px 14px;
    font-family: 'Crimson Pro', serif;
    font-size: 16px; color: var(--ink);
    transition: border-color 0.2s, background 0.2s;
    width: 100%;
    appearance: none; -webkit-appearance: none;
  }
  input:focus, select:focus {
    outline: none;
    border-color: var(--gold);
    background: rgba(255,255,255,0.9);
  }
  select { cursor: pointer; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%236b7fa3'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: right 14px center; padding-right: 36px; }

  .hint { font-size: 12px; color: #9e917c; font-style: italic; margin-top: 2px; }

  .btn {
    width: 100%; margin-top: 36px;
    padding: 16px 32px;
    background: var(--deep);
    color: var(--paper);
    border: none; border-radius: 1px;
    font-family: 'Playfair Display', serif;
    font-size: 15px; letter-spacing: 0.15em; text-transform: uppercase;
    cursor: pointer;
    transition: background 0.25s, transform 0.15s, box-shadow 0.25s;
    box-shadow: 0 4px 20px rgba(26,39,68,0.3);
    position: relative; overflow: hidden;
  }
  .btn::before {
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(201,168,76,0.15), transparent);
    opacity: 0; transition: opacity 0.25s;
  }
  .btn:hover { background: #243360; transform: translateY(-1px); box-shadow: 0 6px 28px rgba(26,39,68,0.4); }
  .btn:hover::before { opacity: 1; }
  .btn:active { transform: translateY(0); }
  .btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }

  /* Result */
  #result { margin-top: 32px; display: none; }
  .result-box {
    padding: 32px 36px; border-radius: 2px;
    text-align: center;
    animation: fadeUp 0.5s ease both;
  }
  @keyframes fadeUp { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }

  .result-box.survived {
    background: linear-gradient(135deg, rgba(26,39,68,0.06), rgba(46,90,60,0.08));
    border: 1px solid rgba(46,90,60,0.3);
  }
  .result-box.perished {
    background: linear-gradient(135deg, rgba(139,58,47,0.08), rgba(26,39,68,0.06));
    border: 1px solid rgba(139,58,47,0.25);
  }

  .result-icon { font-size: 48px; margin-bottom: 14px; line-height: 1; }
  .result-verdict {
    font-family: 'Playfair Display', serif;
    font-size: 28px; font-weight: 700;
    margin-bottom: 8px;
  }
  .result-box.survived .result-verdict { color: #2a5a3a; }
  .result-box.perished .result-verdict { color: var(--rust); }

  .result-prob {
    font-size: 15px; color: #6b6050; font-style: italic;
  }
  .prob-bar-wrap {
    margin: 20px auto 0; max-width: 280px;
    background: rgba(107,127,163,0.15); height: 4px; border-radius: 2px; overflow: hidden;
  }
  .prob-bar { height: 100%; border-radius: 2px; transition: width 0.8s cubic-bezier(0.4,0,0.2,1); }
  .result-box.survived .prob-bar { background: linear-gradient(90deg, #3a7a4f, #5aad74); }
  .result-box.perished .prob-bar { background: linear-gradient(90deg, var(--rust), #c45e50); }

  .factors {
    margin-top: 28px; text-align: left;
    border-top: 1px solid rgba(107,127,163,0.2); padding-top: 20px;
  }
  .factors-title {
    font-size: 10px; letter-spacing: 0.3em; text-transform: uppercase;
    color: var(--mist); margin-bottom: 14px;
  }
  .factor-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
  .factor-name { font-size: 13px; color: #6b6050; }
  .factor-bar-wrap { flex: 1; margin: 0 12px; background: rgba(107,127,163,0.12); height: 3px; border-radius: 2px; overflow: hidden; }
  .factor-bar { height: 100%; background: var(--gold); border-radius: 2px; }
  .factor-pct { font-size: 12px; color: var(--mist); min-width: 32px; text-align: right; }

  .error-box {
    background: rgba(139,58,47,0.08); border: 1px solid rgba(139,58,47,0.2);
    border-radius: 2px; padding: 16px 20px;
    color: var(--rust); font-size: 14px; font-style: italic;
    animation: fadeUp 0.3s ease both;
  }

  @media (max-width: 520px) {
    .card { padding: 28px 20px; }
    .grid-2 { grid-template-columns: 1fr; }
    .field.full { grid-column: 1; }
  }
</style>
</head>
<body>
<div class="wrapper">
  <header>
    <p class="eyebrow">April 15, 1912 · R.M.S. Titanic</p>
    <h1>Will You<br><em>Survive?</em></h1>
    <p class="subtitle">A machine learning oracle consults the passenger manifest</p>
    <div class="rule"><div class="rule-diamond"></div></div>
  </header>

  <div class="card">
    <p class="section-title">Passenger Details</p>
    <div class="grid-2">
      <div class="field">
        <label for="sex">Sex</label>
        <select id="sex">
          <option value="female">Female</option>
          <option value="male" selected>Male</option>
        </select>
      </div>
      <div class="field">
        <label for="age">Age</label>
        <input type="number" id="age" min="0" max="100" step="1" placeholder="e.g. 29" value="29"/>
      </div>
      <div class="field">
        <label for="pclass">Ticket Class</label>
        <select id="pclass">
          <option value="1">First Class</option>
          <option value="2">Second Class</option>
          <option value="3" selected>Third Class</option>
        </select>
      </div>
      <div class="field">
        <label for="embarked">Port of Embarkation</label>
        <select id="embarked">
          <option value="S" selected>Southampton</option>
          <option value="C">Cherbourg</option>
          <option value="Q">Queenstown</option>
        </select>
      </div>
    </div>

    <p class="section-title" style="margin-top:28px">Family Aboard</p>
    <div class="grid-2">
      <div class="field">
        <label for="sibsp">Siblings / Spouses</label>
        <input type="number" id="sibsp" min="0" max="10" value="0"/>
        <span class="hint">Number aboard with you</span>
      </div>
      <div class="field">
        <label for="parch">Parents / Children</label>
        <input type="number" id="parch" min="0" max="10" value="0"/>
        <span class="hint">Number aboard with you</span>
      </div>
    </div>

    <p class="section-title" style="margin-top:28px">Fare</p>
    <div class="grid-2">
      <div class="field full">
        <label for="fare">Ticket Fare (£)</label>
        <input type="number" id="fare" min="0" step="0.01" placeholder="e.g. 32.50" value="32.50"/>
      </div>
    </div>

    <button class="btn" onclick="predict()">⚓ Consult the Oracle</button>
  </div>

  <div id="result"></div>
</div>

<script>
async function predict() {
  const btn = document.querySelector('.btn');
  btn.disabled = true;
  btn.textContent = 'Consulting…';

  const payload = {
    pclass: parseInt(document.getElementById('pclass').value),
    sex: document.getElementById('sex').value,
    age: parseFloat(document.getElementById('age').value) || null,
    sibsp: parseInt(document.getElementById('sibsp').value) || 0,
    parch: parseInt(document.getElementById('parch').value) || 0,
    fare: parseFloat(document.getElementById('fare').value) || null,
    embarked: document.getElementById('embarked').value
  };

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    showResult(data);
  } catch(e) {
    document.getElementById('result').style.display = 'block';
    document.getElementById('result').innerHTML = `<div class="error-box">Something went wrong. Please try again.</div>`;
  }

  btn.disabled = false;
  btn.textContent = '⚓ Consult the Oracle';
}

function showResult(data) {
  const el = document.getElementById('result');
  el.style.display = 'block';

  if (data.error) {
    el.innerHTML = `<div class="error-box">${data.error}</div>`;
    return;
  }

  const survived = data.survived;
  const prob = Math.round((survived ? data.survival_probability : (1 - data.survival_probability)) * 100);
  const survProb = Math.round(data.survival_probability * 100);

  const factors = data.feature_importance.map(f => `
    <div class="factor-row">
      <span class="factor-name">${f.feature}</span>
      <div class="factor-bar-wrap"><div class="factor-bar" style="width:${Math.round(f.importance*100)}%"></div></div>
      <span class="factor-pct">${Math.round(f.importance*100)}%</span>
    </div>
  `).join('');

  el.innerHTML = `
    <div class="result-box ${survived ? 'survived' : 'perished'}">
      <div class="result-icon">${survived ? '🛟' : '🌊'}</div>
      <div class="result-verdict">${survived ? 'Survived' : 'Did Not Survive'}</div>
      <div class="result-prob">Survival probability: <strong>${survProb}%</strong></div>
      <div class="prob-bar-wrap">
        <div class="prob-bar" style="width:0%" id="pbar"></div>
      </div>
      <div class="factors">
        <div class="factors-title">Key Factors in This Prediction</div>
        ${factors}
      </div>
    </div>
  `;

  // Animate bar
  setTimeout(() => {
    document.getElementById('pbar').style.width = survProb + '%';
  }, 100);

  el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
</script>
</body>
</html>"""

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        d = request.get_json()
        age = d.get('age') or age_median
        fare = d.get('fare') or fare_median
        sex_enc = le_sex.transform([d['sex']])[0]
        emb_enc = le_emb.transform([d['embarked']])[0]

        X = [[d['pclass'], sex_enc, age, d['sibsp'], d['parch'], fare, emb_enc]]
        pred = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][1])

        feat_names = ['Ticket Class', 'Sex', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare', 'Embarkation']
        importances = model.feature_importances_
        total = sum(importances)
        feat_imp = sorted(
            [{'feature': n, 'importance': float(i/total)} for n, i in zip(feat_names, importances)],
            key=lambda x: -x['importance']
        )[:5]

        return jsonify({'survived': pred == 1, 'survival_probability': proba, 'feature_importance': feat_imp})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
