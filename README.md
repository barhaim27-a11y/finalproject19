# 🧠 Parkinsons – ML App (v8.6)

Streamlit app for Parkinson’s disease modeling & prediction — with:
- DATA/EDA
- Single/Multi model training (metrics + ROC/PR/CM + exports)
- Best Dashboard (shows bundled Production model only)
- Predict (always uses Production model; threshold from metadata)
- Retrain (train on uploaded CSV; Promote if better to replace Production)

**No manual upload is needed for Predict.**  
**Data loads via `mp.load_data`** — if `data/parkinsons.csv` is missing, a small demo file is created so the app boots.

## Run
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Replace Production model with your notebook model
Place your model at `models/best_model.joblib` and metadata `assets/best_model.json` with:
```json
{"opt_thr": 0.47}
```
(according to your EDA notebook). From then on, Predict will use it until a better model is promoted via Retrain.
