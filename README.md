# Morgan Stanley — Publication Readership Predictor

Predicts expected reads for analyst publications before they are sent, enabling editorial teams to optimise distribution strategy.

## Live Demo
Streamlit App: https://ms-readership-streamlit-771249737445.europe-west1.run.app

## Model Performance
| Metric | Value |
|--------|-------|
| R2 (test set) | 0.937 |
| R2 (5-fold CV) | 0.856 |
| RMSE | 115 reads |
| MAE | 89 reads |

## Project Structure
    ms_readership_project/
    ├── api/                    FastAPI REST endpoint
    │   └── main.py
    ├── model/                  Trained model artifacts
    │   ├── model.pkl
    │   ├── feature_names.pkl
    │   └── mappings.json
    ├── streamlit_app/          Streamlit UI
    │   └── app.py
    ├── Untitled-1.ipynb        Full EDA, modelling and SHAP notebook
    ├── Dockerfile              Container configuration
    └── requirements.txt        Dependencies

## Methodology
1. EDA — analysed 1,000 publications across channels, seniority levels and content types
2. Statistical validation — t-tests and ANOVA to select statistically significant features (p<0.05)
3. Feature Engineering — ordinal encoding, one-hot encoding, log transformation of target
4. Modelling — compared baseline vs tuned XGBRegressor with 5-fold cross-validation
5. Explainability — SHAP global and local feature importance
6. Deployment — FastAPI + Docker + GCP Cloud Run

## Key Findings (SHAP)
- Analyst seniority — top driver, MDs generate 2.6x more reads than Junior analysts
- 30-day avg reads — recent momentum is the strongest engagement signal
- Subscriber count — audience size directly drives readership volume
- Email channel — outperforms Web by 53% (push vs pull dynamic)
- Send hour — peak reads at 9am, drops significantly after 2pm

## Tech Stack
- ML: XGBoost, scikit-learn, SHAP
- Tracking: MLflow
- API: FastAPI
- UI: Streamlit
- Containerisation: Docker
- Cloud: GCP Cloud Run (europe-west1)
- Language: Python 3.13

## Run Locally
git clone https://github.com/pankajdawani/ms_readership_project.git
cd ms_readership_project
pip install -r requirements.txt
streamlit run streamlit_app/app.py
uvicorn api.main:app --reload --port 8000

## Docker
docker build -t ms-readership-api .
docker run -p 8000:8000 ms-readership-api
