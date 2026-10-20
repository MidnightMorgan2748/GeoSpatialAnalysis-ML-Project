# 🌍 GeoSpatial Analysis & Machine Learning Project
### Global Socio-Economic, Demographic & Geospatial Analytics — Comparative Study of 195 Sovereign Nations

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![GeoPandas](https://img.shields.io/badge/GeoPandas-Spatial-green.svg)](https://geopandas.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-purple.svg)](https://plotly.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine--Learning-yellow.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

This project implements an end-to-end data science and geospatial analytics lifecycle on a global dataset comprising **195 sovereign nations**. Combining cartographic visualization, exploratory data analysis, and machine learning, this repository investigates macro-demographic, economic, and digital connectivity patterns worldwide.

### Key Capabilities
- **Geospatial & Cartographic Modeling:** Interactive multi-layer Folium maps (satellite, marker clustering, population-density heatmaps), GeoPandas spatial projections (EPSG:4326 / EPSG:3857), Contextily satellite tile overlays, and Plotly 3D/orthographic projections.
- **Statistical Analytics:** Continental comparative analysis, kernel density estimations (KDE), Cumulative Distribution Functions (CDF), violin plots, box plots, and correlation heatmaps.
- **Unsupervised Machine Learning:** K-Means clustering ($k=4$) with `StandardScaler` feature normalization to classify nations into development archetypes based on socio-economic indicators.
- **Supervised Machine Learning:** Bivariate Linear Regression modeling the relationship between national digital adoption (% internet users) and the UN Human Development Index (HDI), complete with residual diagnostics and $R^2$ evaluation.

---

## 📂 Repository Structure

```
GeoSpatialAnalysis-ML-Project/
├── country_stats.csv                 # Core dataset: socio-economic & geospatial indicators (195 nations)
├── fds_lab9.ipynb                    # Master production Jupyter Notebook (96 comprehensive cells)
├── generate_fds_lab9_notebook.py     # Reproducible Python generator script for the master notebook
├── Lab_9_FDS.ipynb                   # Exploratory analysis notebook
├── requirements.txt                  # Python package dependencies
├── .gitignore                        # Git ignore rules for artifacts & checkpoints
└── README.md                         # Project documentation
```

---

## 📊 Dataset Description

The dataset (`country_stats.csv`) covers 195 sovereign states with indicators across multiple dimensions:

| Column | Description |
|---|---|
| `Country name` | Official sovereign state name |
| `Total in km2` | Total national geographical area |
| `Land in km2` | Total land area excluding inland waters |
| `Inland water in km2` | Surface area occupied by inland waters |
| `% water` | Percentage of territorial surface consisting of water |
| `Population` | Total sovereign population |
| `% of world population` | Percentage share of aggregate global population |
| `Population density` | Inhabitants per square kilometer |
| `HDI` | United Nations Human Development Index score (0 to 1) |
| `Fertility rate` | Mean births per woman |
| `Mortality rate` | Crude death rate per 1,000 individuals |
| `% of internet users in population` | Percentage of population with digital connectivity |
| `External debt (billions)` | Total external debt liability in billions USD |

---

## 🔬 Analytical Workflow

```
┌─────────────────────────────────┐
│     Multi-Source Ingestion      │ -> Local / Google Colab / Google Drive Detection
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│   Data Cleaning & Engineering   │ -> Coordinates, HDI Tiers, Debt-per-Capita
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│      Geospatial Cartography     │ -> Folium, GeoPandas, Contextily, Plotly
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│    Multivariate Statistical EDA │ -> Correlation, CDF, Violin, Box Plots
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│         Machine Learning        │ -> K-Means Clustering & Linear Regression
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│    Master Styled Scorecards     │ -> Executive Ranking & Comparative Metrics
└─────────────────────────────────┘
```

1. **Environment Setup & Resilient Ingestion:** Dynamic runtime detection ensuring seamless execution locally or within Google Colaboratory.
2. **Data Cleaning & Multi-Dimensional Harmonization:** Type coercion, missing value imputation, coordinate imputation (latitude/longitude centroids), and continent mapping.
3. **Feature Engineering:**
   - Classification into UN HDI tiers: *Low* ($<0.550$), *Medium* ($0.550 - 0.699$), *High* ($0.700 - 0.799$), *Very High* ($\ge 0.800$).
   - Debt-per-capita and land-to-water ratio metrics.
4. **Interactive Geospatial Cartography:**
   - Folium MarkerCluster and weighted heatmaps.
   - Contextily satellite imagery with GeoPandas geometry.
   - Plotly Natural Earth and orthographic 3D scatter geo charts.
5. **Statistical Distribution Analysis:**
   - Continental grouping, box and violin plots for demographic transitions.
   - Cumulative distribution analysis showing population concentration.
6. **Machine Learning:**
   - **K-Means Clustering ($k=4$):** Grouping nations based on population, HDI, internet penetration, fertility, mortality, and external debt.
   - **Linear Regression:** Quantitative modeling of $HDI = f(\text{Internet Penetration})$.

---

## 🚀 Installation & Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/MidnightMorgan2748/GeoSpatialAnalysis-ML-Project.git
cd GeoSpatialAnalysis-ML-Project
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook fds_lab9.ipynb
```

*(Alternatively, open `fds_lab9.ipynb` in [Google Colaboratory](https://colab.research.google.com/)).*

---

## 📈 Key Findings & Insights

- **Digital Connectivity & Human Development:** A strong positive linear correlation ($R^2 > 0.65$) exists between internet adoption rates and HDI, indicating digital infrastructure is strongly coupled with educational, health, and economic development.
- **Demographic Polarization:** The two most populous nations (India and China) comprise over 35% of the aggregate world population, demonstrating extreme right-skewed demographic distributions compared to a median country size of ~8 million.
- **Demographic Transition Dividends:** Nations with Very High HDI exhibit sub-replacement fertility rates ($<1.6$), while Low HDI states present high fertility rates ($>4.0$) and high youth dependency ratios.
- **Macroeconomic Debt Distribution:** External debt liabilities are heavily concentrated among high-income, mature economies reflecting credit market depth rather than direct default risk.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
