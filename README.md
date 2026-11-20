# Can GLP-1 Drugs Solve the Obesity Epidemic?
### An Exploratory Data Analysis

## Headline Result

Predicting 7 obesity categories from 10 lifestyle features (no weight or height), tested with 5-fold cross-validation:

- **XGBoost reaches 71.6%** against a **19.5%** majority-class baseline (always guessing Obesity Type III) and 52.0% for logistic regression
- **65.1%** is the figure to trust. It is the same model scored on the 6 remaining categories after Obesity Type III, a synthetic-data artefact, is excluded
- The 85.5% from a model that also sees weight and height is **target leakage**, because BMI is computed from those two columns
- The dataset is partly synthetic (SMOTE-expanded), and the model predicts but does not explain

## Live Dashboard

**[View the interactive dashboard here](https://glp1-obesity.streamlit.app/)**

Built with Streamlit and Plotly. It covers global obesity trends and GLP-1 drug effectiveness, plus a cross-validated comparison of a logistic regression against two tree models (random forest and XGBoost) predicting obesity level from lifestyle factors.

This project explores the global obesity epidemic and the rapid rise of GLP-1 receptor agonist drugs (Wegovy, Mounjaro, Ozempic) as a treatment. By combining multiple data sources, we investigate whether these drugs can address the growing obesity crisis.

## Key Findings

- Global obesity has **more than tripled** since 1980, affecting every region worldwide
- **Age, family history and diet** rank among the top predictors of obesity category across the models tested. These are predictive associations, and none of them is a demonstrated cause
- GLP-1 drugs are highly effective: **Mounjaro achieves 20.9% average weight loss**, nearly 9x the figure for lifestyle changes alone
- GLP-1 prescriptions grew **10x from 2018 to 2024**, with spending reaching $52 billion
- Including weight and height gives **85.5% accuracy** on a single 80/20 split, but that figure reflects **target leakage**. BMI is calculated from those two columns, so the model recomputes the label it is being asked to predict
- Using **lifestyle features only** (7 classes, 5-fold CV), logistic regression manages **52.0%**, while XGBoost reaches **71.6%**, against a 19.5% majority-class baseline. That falls to **65.1%** on the other 6 classes once Obesity Type III is excluded as a data-generation artefact. The weak linear result says more about the model than about the lifestyle data
- The case for GLP-1 drugs rests on the clinical trials: **2.4%** average weight loss from lifestyle programmes against **20.9%** from tirzepatide. The predictive model plays no part in that case

## Project Structure

```
glp1-obesity-analysis/
├── glp1_obesity_analysis.ipynb   # Main analysis notebook
├── data/
│   ├── obesity_global.csv        # WHO global obesity data (1980-2024)
│   ├── obesity_level.csv         # Individual obesity risk factors (Kaggle)
│   ├── google_trends_glp1.csv    # Search interest for GLP-1 drugs
│   ├── glp1_effectiveness.csv    # Clinical trial results
│   └── glp1_prescriptions.csv   # Prescription and spending trends
├── images/                       # Saved chart images
├── app.py                        # Streamlit dashboard
├── requirements.txt              # Dashboard dependencies
└── README.md
```

## Analysis Overview

| Section | Description | Charts |
|---------|-------------|--------|
| 1. Data Exploration | Load and inspect 5 datasets | n/a |
| 2. Data Cleaning | Separate countries/regions, rename columns, fix typos | n/a |
| 3. Global Obesity Trends | WHO data analysis across countries and regions | Charts 1-3 |
| 4. Risk Factor Analysis | Lifestyle factors, family history, correlation heatmap | Charts 4-8 |
| 5. Rise of GLP-1 Drugs | Prescriptions, Google Trends, spending, cross-dataset analysis | Charts 9-12 |
| 6. Drug Effectiveness | Clinical trial weight loss comparison | Chart 13 |
| 7. Predictive Model | Model A vs Model B, then a 5-fold cross-validated comparison of logistic regression against random forest and XGBoost, with per-class evaluation and cross-model feature importance | Charts 14-17 |
| 8. Conclusions | Data-backed answer to the title question | n/a |

## Limitations

The first two items are separate problems, and they are easy to confuse. One is about how the data was generated. The other is about which columns were fed to a model.

- **The risk-factor dataset is partly synthetic, and it inflates the results.** Within Obesity Type III, six of the ten lifestyle features are constant or near-constant, and 99.9% of that class is female. What shows through there is a SMOTE fingerprint. The model identifies the class at 98.6% recall for the wrong reason. Excluding it, accuracy falls from 71.6% to **65.1%**, which is the honest figure quoted throughout.
- **Model A leaks its target by construction.** Weight and height mechanically determine BMI, which in turn defines the categories being predicted. Model A exists to demonstrate that contrast. It has no connection to the synthetic-data issue above.
- **The models predict, and they do not explain.** Feature importance shows what predicts the label in this dataset. It says nothing about what causes obesity, or about what would change if someone changed a habit.
- **The screening signal is uneven.** It holds up at the severe end (Obesity Type II F1 = 0.89) and weakens badly in the middle (Overweight Level I F1 = 0.42). 51% of errors land in an adjacent severity category.
- Google Trends and prescription figures are reconstructions built from published research.

## Data Sources

- **WHO Global Health Observatory** via [Our World in Data](https://ourworldindata.org/grapher/share-of-adults-defined-as-obese)
- **Kaggle** [Obesity Risk Dataset](https://www.kaggle.com/datasets/jpkochar/obesity-risk-dataset) (partly synthetic, SMOTE-expanded; see Limitations)
- **STEP trials** (semaglutide) and **SURMOUNT trials** (tirzepatide)
- **FAIR Health** [GLP-1 White Paper](https://s3.amazonaws.com/media2.fairhealth.org/whitepaper/asset/Obesity%20and%20GLP-1%20Drugs%20-%20A%20FAIR%20Health%20White%20Paper.pdf)
- **KFF** [Medicare GLP-1 Report](https://www.kff.org/medicare/recent-trends-in-glp-1-use-and-spending-in-medicare/)

## Tools Used

- Python (pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost)
- Jupyter Notebook
