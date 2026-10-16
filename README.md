# Prepaid to Postpaid Conversion Prediction

End-to-end machine learning project that predicts which prepaid telecom
customers are likely to convert to postpaid within the next 60 days, so the
marketing team can target the right customers.

## Business Problem
Postpaid customers bring higher and more stable revenue. Calling every prepaid
customer is expensive (Rs 60 per call), so we rank customers by conversion
probability and contact only the top segment.

- Target: `1` if the customer converts to postpaid within 60 days after the
  observation date, otherwise `0`
- Features: customer behaviour in the 90 days before the observation date
- Base conversion rate: about 3% (highly imbalanced)
- Break-even precision: cost per call / value per conversion = 60 / 2000 = 3%

## Success Metrics
| Type | Metric |
|---|---|
| Primary ML | PR-AUC, Precision@Top-10%, Lift@Top-10% |
| Secondary ML | Recall@Top-10%, ROC-AUC |
| Business | Campaign ROI, cost per acquisition |

## Tech Stack
Python, pandas, scikit-learn, MLflow, FastAPI, Docker, GitHub Actions, AWS

## Project Structure
```
├── configs/        # YAML configuration
├── data/           # raw / interim / processed (not tracked by Git)
├── docs/           # business and design documents
├── models/         # trained models (not tracked by Git)
├── notebooks/      # exploration only
├── reports/        # figures and evaluation reports
├── src/            # production code (data, features, models, api, monitoring)
└── tests/          # pytest tests
```

## Setup
```powershell
git clone https://github.com/kavi-rahangdale/prepaid-to-postpaid-conversion.git
cd prepaid-to-postpaid-conversion
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```
Python version used: 3.13

## Project Status
- [x] Phase 1: Business understanding
- [x] Phase 2: Project setup
- [ ] Phase 3: Synthetic data generation
- [ ] Phase 4: Data validation and EDA
- [ ] Phase 5 onwards: features, training, tracking, API, Docker, CI/CD, AWS, monitoring
