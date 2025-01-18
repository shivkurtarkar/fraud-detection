# MODEL_FILE=models/best_model.pkl uvicorn api:app --reload --host 0.0.0.0 --port 8000
MODEL_FILE=models/best_model.pkl gunicorn api:app --reload  -b 0.0.0.0:8000