# Static Malware Detection with Machine Learning
## Run locally

### 1. Install dependencies
pip install -r requirements.txt

### 2. Train final model
python -m malware_detector.modeling.final_train_and_test

### 3. Run API
uvicorn malware_detector.api.app:app --reload

### 4. Run Streamlit UI
streamlit run web/streamlit_app.py

### 5. Run tests
pytest -q