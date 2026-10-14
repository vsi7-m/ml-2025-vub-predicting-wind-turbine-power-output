# Predicting Wind Turbine Power Output — Machine Learning

Project voor het vak **Machine Learning** aan de **Vrije Universiteit Brussel (VUB)**.

## Projectoverzicht

Het doel van dit project is het accuraat voorspellen van het gegenereerde vermogen (*power output*) van windturbines op basis van meteorologische omstandigheden en operationele sensordata (SCADA-data, waaronder windsnelheid, windrichting en omgevingsfactoren). 

Het project omvat het volledige machine learning proces: van exploratieve data-analyse en data-cleaning tot hyperparameter optimalisatie en geavanceerde ensemble-technieken.

---

## Aanpak & Experimenten

### 1. Baselines & Data-analyse
* **Manufacturer Power Curve Baseline:** Vergelijking tegen de theoretische vermogenscurve van de fabrikant.
* **Bias-Variance & Learning Curves:** Analyse van onder- en overfittering over verschillende datafolds en iteraties.

### 2. Modellen & Hyperparameter Tuning
Diverse regressiemodellen zijn getraind en geoptimaliseerd met behulp van **Optuna** voor geautomatiseerde hyperparameter optimalisatie:
* **Gradient Boosting:** 
  * LightGBM (`lgbm_optuna`)
  * XGBoost (`xgboost_optuna`)
  * CatBoost (`catboost_info`)
* **Tree-based Ensembles:** Random Forest Regressor (`randomforest_optuna`)
* **Neurale Netwerken:** Multi-Layer Perceptron (`mlp_model`)

### 3. Ensembling & Stacking
Om de voorspellende nauwkeurigheid te maximaliseren zijn verschillende modellen gecombineerd:
* **Averaging Ensembles:** Gewogen en ongewogen gemiddeldes van sterke individuele modellen.
* **Triple Threat Ensemble:** Samengesteld ensemble gericht op complementaire voorspelfouten van verschillende modelarchitecturen.
* **Super Stacker:** Een 2-traps stacking-regressor waarbij de out-of-fold voorspellingen van basismodellen als meta-features worden gebruikt.

---

## Repository Structuur

| Bestand / Map | Omschrijving |
| :--- | :--- |
| `start_here.ipynb` | Introductienotebook met basis data-exploratie |
| `biasvariance_learningcurve.ipynb` | Analyse van modelcomplexiteit en leercurves |
| `example_classification_pipeline.ipynb` | Referentie-pipeline voor dataverwerking |
| `xgboost_optuna_*.ipynb` | XGBoost modelontwikkeling & Optuna tuning |
| `lgbm_optuna_*.ipynb` | LightGBM modelontwikkeling & Optuna tuning |
| `randomforest_optuna_*.ipynb` | Random Forest regressie & tuning |
| `mlp_model_*.ipynb` | Deep learning / MLP implementatie |
| `super_stacker_*.ipynb` | Stacking regressor implementatie |
| `final_triple_threat_ensemble_*.ipynb` | Finale ensemble-pipeline voor test-submissies |

---

## Vereisten & Gebruikte Bibliotheken

* Python 3.9+
* `numpy`, `pandas`
* `scikit-learn`
* `lightgbm`, `xgboost`, `catboost`
* `optuna`
* `matplotlib`, `seaborn`
