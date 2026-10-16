# Business Understanding

## 1. Problem Statement
(अपने शब्दों में: कौन सी समस्या, किसके लिए)

## 2. Target Definition
- target = 1: converted to postpaid within 60 days after observation date
- target = 0: did not convert
- Features come only from the 90 days before the observation date (no leakage)

## 3. Business Case
- Cost per call: Rs 60, value per conversion: Rs 2000
- Break-even precision = 60 / 2000 = 3%
- Random targeting profit: about 0; model targeting top 10% at 12% precision: large positive

## 4. Success Metrics
- ML: PR-AUC, Precision@Top-10%, Lift@Top-10%
- Business: ROI, cost per acquisition
- Targets: Precision@10% >= 10%, Lift@10% >= 3x

## 5. Stakeholders
Marketing, Business head, Finance, Data Engineering, Compliance, IT/DevOps

## 6. Constraints and Risks
Privacy (no PII), batch scoring, explainability (SHAP), class imbalance,
low AWS budget, data drift, probability is not the same as uplift
