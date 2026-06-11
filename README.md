# Can GLP-1 Drugs Solve the Obesity Epidemic?
### An Exploratory Data Analysis

This project explores the global obesity epidemic and the rapid rise of GLP-1 receptor agonist drugs (Wegovy, Mounjaro, Ozempic) as a treatment. By combining multiple data sources, we investigate whether these drugs can address the growing obesity crisis.

## Key Findings

- Global obesity has **more than tripled** since 1980, affecting every region worldwide
- Lifestyle factors alone **cannot fully explain obesity** — family history and genetics play a major role
- GLP-1 drugs are highly effective: **Mounjaro achieves 20.9% average weight loss**, nearly 9x more than lifestyle changes alone
- GLP-1 prescriptions grew **10x from 2018 to 2024**, with spending reaching $52 billion
- A logistic regression model achieved **85.5% accuracy** in predicting obesity level, but accuracy drops to **52.5%** using lifestyle factors alone — confirming the need for biological interventions

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
└── README.md
```

## Analysis Overview

| Section | Description | Charts |
|---------|-------------|--------|
| 1. Data Exploration | Load and inspect 5 datasets | — |
| 2. Data Cleaning | Separate countries/regions, rename columns, fix typos | — |
| 3. Global Obesity Trends | WHO data analysis across countries and regions | Charts 1-3 |
| 4. Risk Factor Analysis | Lifestyle factors, family history, correlation heatmap | Charts 4-8 |
| 5. Rise of GLP-1 Drugs | Prescriptions, Google Trends, spending, cross-dataset analysis | Charts 9-12 |
| 6. Drug Effectiveness | Clinical trial weight loss comparison | Chart 13 |
| 7. Predictive Model | Two logistic regression models comparing all features vs lifestyle-only | Charts 14-15 |
| 8. Conclusions | Data-backed answer to the title question | — |

## Data Sources

- **WHO Global Health Observatory** via [Our World in Data](https://ourworldindata.org/grapher/share-of-adults-defined-as-obese)
- **Kaggle** [Obesity Risk Dataset](https://www.kaggle.com/datasets/jpkochar/obesity-risk-dataset)
- **STEP trials** (semaglutide) and **SURMOUNT trials** (tirzepatide)
- **FAIR Health** [GLP-1 White Paper](https://s3.amazonaws.com/media2.fairhealth.org/whitepaper/asset/Obesity%20and%20GLP-1%20Drugs%20-%20A%20FAIR%20Health%20White%20Paper.pdf)
- **KFF** [Medicare GLP-1 Report](https://www.kff.org/medicare/recent-trends-in-glp-1-use-and-spending-in-medicare/)

## Tools Used

- Python (pandas, numpy, matplotlib, seaborn, scikit-learn)
- Jupyter Notebook
