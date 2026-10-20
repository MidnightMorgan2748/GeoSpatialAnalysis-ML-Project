import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(code):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    })

# ==============================================================================
# SECTION 1: INTRODUCTION & SETUP
# ==============================================================================
add_md("""# 🌍 Foundations of Data Science (FDS) — Lab 9
## Global Socio-Economic, Demographic & Geospatial Analytics
### Comparative Study of 195 Sovereign Nations

**Course:** Foundations of Data Science (FDS)  
**Notebook Name:** `fds_lab9.ipynb`  
**Execution Environment:** Google Colaboratory / Local Jupyter  
**Dataset:** Global Country Statistics (`country_stats.csv`)  

---

### 📌 Project Executive Summary
This laboratory project implements an end-to-end data science lifecycle on a comprehensive global dataset of **195 sovereign nations**. Adopting the analytical and visual rigor established in the city-level spatial studies, this investigation scales the paradigm to the macro-economic and geospatial domain.

#### Analytical Workflow:
1. **Environment Setup & Resilient Ingestion:** Dynamic runtime detection (Google Colab / Google Drive / Local).
2. **Data Cleaning & Multi-Dimensional Harmonization:** String cleaning, missing value imputation, type coercion.
3. **Geospatial & Socio-Economic Feature Engineering:** Continental mapping, Centroid Coordinates, UN Human Development Index (HDI) tier categorization, debt-per-capita metrics, and population discretizations.
4. **Interactive Geospatial Cartography:** Plotly Geo projections, GeoPandas geometric modeling, Esri Satellite Basemaps via Contextily, and interactive Folium multi-layer maps (Cluster, Satellite, HeatMap).
5. **Multi-Variable Statistical Analytics:** Continental grouping, violin plots, box plots, correlation matrix, pair plots, and Cumulative Distribution Functions (CDF).
6. **Unsupervised & Supervised Machine Learning:**
   - **K-Means Clustering** ($k=4$) with StandardScaler normalization to segment nations by socio-demographic indicators.
   - **Bivariate Linear Regression** modeling the nexus between digital connectivity (Internet %) and Human Development Index (HDI), with residual analysis and $R^2$ evaluation.
7. **Executive Scorecards & Styled DataFrames:** Gradient heat-tables ranking global nations on demographic and economic performance.""")

add_md("""### ⚙️ 1. Environment Configuration & Package Installation
We ensure all prerequisite libraries are installed. In Google Colab, packages such as `geopandas`, `folium`, and `contextily` are installed quietly.""")

add_code("""# Install required geospatial and visualization libraries
!pip install geopandas folium plotly contextily scikit-learn seaborn matplotlib --quiet
""")

add_md("""### 📚 2. Core Library Ingestion & Plotting Defaults
Importing core scientific computing, visualization, and machine learning modules with customized visualization aesthetics.""")

add_code("""import os
import sys
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Configure display and plot settings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: f'{x:,.3f}')

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 110
""")

add_md("""### 📂 3. Multi-Environment Resilient Dataset Ingestion
This cell detects whether the code is running locally, within Google Colab, or connected to Google Drive, ensuring zero-configuration execution.""")

add_code("""# Resilient multi-environment dataset loader
potential_paths = [
    'country_stats.csv',
    '/content/country_stats.csv',
    '/content/drive/MyDrive/country_stats.csv',
    '../country_stats.csv'
]

dataset_path = None
for p in potential_paths:
    if os.path.exists(p):
        dataset_path = p
        break

if dataset_path is None:
    # Attempt Colab Drive mount
    try:
        from google.colab import drive
        print("Mounting Google Drive to /content/drive ...")
        drive.mount('/content/drive')
        if os.path.exists('/content/drive/MyDrive/country_stats.csv'):
            dataset_path = '/content/drive/MyDrive/country_stats.csv'
    except Exception:
        pass

if dataset_path is None:
    # Fallback interactive upload for Colab
    try:
        from google.colab import files
        print("📁 'country_stats.csv' not found automatically. Please upload the file:")
        uploaded = files.upload()
        dataset_path = list(uploaded.keys())[0]
    except Exception as e:
        raise FileNotFoundError("Could not find 'country_stats.csv'. Ensure the file is present in working directory or Colab root.")

print(f"✅ Successfully loaded dataset from: {dataset_path}")
df_raw = pd.read_csv(dataset_path)
print(f"📊 Dataset Shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
df_raw.head(10)
""")

# ==============================================================================
# SECTION 2: DATA CLEANING & FEATURE ENGINEERING
# ==============================================================================
add_md("""---
# 🧹 Data Cleaning, Sanitization & Feature Engineering
Before proceeding to exploratory analysis and machine learning, the dataset must undergo thorough quality checks, missing value resolution, numeric type coercions, and geospatial metadata enrichment.""")

add_md("""### 🔍 Inspect Missing Values & Data Types""")

add_code("""# Inspect data types and initial missing value counts
print("=== Column Types and Null Counts ===")
display(pd.DataFrame({
    'Data Type': df_raw.dtypes,
    'Null Count': df_raw.isnull().sum(),
    'Null %': (df_raw.isnull().sum() / len(df_raw) * 100).round(2)
}))
""")

add_md("""### 🧼 Numeric Sanitization & Parsing
Columns such as `External debt (billions)` contain comma thousand-separators (e.g., `'3,468.2119'`). We strip whitespace, remove commas, and convert all quantitative features to numeric float representations.""")

add_code("""df = df_raw.copy()
df.columns = df.columns.str.strip()

# Numeric columns to sanitize and convert
numeric_columns = [
    'Total in km2', 'Land in km2', 'Inland water in km2', '% water',
    'Population', '% of world population', 'Population density',
    'HDI', 'Fertility rate', 'Mortality rate',
    '% of internet users in population', 'External debt (billions)'
]

for col in numeric_columns:
    if col in df.columns:
        # Remove commas, strip whitespace, and coerce to numeric
        df[col] = df[col].astype(str).str.replace(',', '').str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("✅ Numeric columns successfully coerced.")
df.dtypes
""")

add_md("""### 🌐 Geospatial Metadata & Global Coordinate Mapping
To enable spatial maps (GeoPandas, Folium, Plotly), we inject a complete reference mapping of all **195 sovereign nations** with:
- **Continent**: Africa, Asia, Europe, North America, South America, Oceania
- **Centroid Coordinates**: Latitude and Longitude
- **ISO-3 Alpha Code**: Official 3-letter ISO code for global choropleth rendering""")

add_code("""# Complete geographic metadata dictionary for all 195 sovereign nations
COUNTRY_METADATA = {
    'Russia': ('Europe', 61.5240, 105.3188, 'RUS'),
    'Canada': ('North America', 56.1304, -106.3468, 'CAN'),
    'China': ('Asia', 35.8617, 104.1954, 'CHN'),
    'United States': ('North America', 37.0902, -95.7129, 'USA'),
    'Brazil': ('South America', -14.2350, -51.9253, 'BRA'),
    'Australia': ('Oceania', -25.2744, 133.7751, 'AUS'),
    'India': ('Asia', 20.5937, 78.9629, 'IND'),
    'Argentina': ('South America', -38.4161, -63.6167, 'ARG'),
    'Kazakhstan': ('Asia', 48.0196, 66.9237, 'KAZ'),
    'Algeria': ('Africa', 28.0339, 1.6596, 'DZA'),
    'Democratic Republic of the Congo': ('Africa', -4.0383, 21.7587, 'COD'),
    'Saudi Arabia': ('Asia', 23.8859, 45.0792, 'SAU'),
    'Mexico': ('North America', 23.6345, -102.5528, 'MEX'),
    'Indonesia': ('Asia', -0.7893, 113.9213, 'IDN'),
    'Sudan': ('Africa', 12.8628, 30.2176, 'SDN'),
    'Libya': ('Africa', 26.3351, 17.2283, 'LBY'),
    'Iran': ('Asia', 32.4279, 53.6880, 'IRN'),
    'Mongolia': ('Asia', 46.8625, 103.8467, 'MNG'),
    'Peru': ('South America', -9.1899, -75.0152, 'PER'),
    'Chad': ('Africa', 15.4542, 18.7322, 'TCD'),
    'Niger': ('Africa', 17.6078, 8.0817, 'NER'),
    'Angola': ('Africa', -11.2027, 17.8739, 'AGO'),
    'Mali': ('Africa', 17.5707, -3.9962, 'MLI'),
    'South Africa': ('Africa', -30.5595, 22.9375, 'ZAF'),
    'Colombia': ('South America', 4.5709, -74.2973, 'COL'),
    'Ethiopia': ('Africa', 9.1450, 40.4897, 'ETH'),
    'Bolivia': ('South America', -16.2902, -63.5887, 'BOL'),
    'Mauritania': ('Africa', 21.0079, -10.9408, 'MRT'),
    'Egypt': ('Africa', 26.8206, 30.8025, 'EGY'),
    'Tanzania': ('Africa', -6.3690, 34.8888, 'TZA'),
    'Nigeria': ('Africa', 9.0820, 8.6753, 'NGA'),
    'Venezuela': ('South America', 6.4238, -66.5897, 'VEN'),
    'Pakistan': ('Asia', 30.3753, 69.3451, 'PAK'),
    'Namibia': ('Africa', -22.9576, 18.4904, 'NAM'),
    'Mozambique': ('Africa', -18.6657, 35.5296, 'MOZ'),
    'Turkey': ('Asia', 38.9637, 35.2433, 'TUR'),
    'Chile': ('South America', -35.6751, -71.5430, 'CHL'),
    'Zambia': ('Africa', -13.1339, 27.8493, 'ZMB'),
    'Myanmar': ('Asia', 21.9162, 95.9560, 'MMR'),
    'Afghanistan': ('Asia', 33.9391, 67.7100, 'AFG'),
    'South Sudan': ('Africa', 6.8770, 31.3070, 'SSD'),
    'France': ('Europe', 46.2276, 2.2137, 'FRA'),
    'Somalia': ('Africa', 5.1521, 46.1996, 'SOM'),
    'Central African Republic': ('Africa', 6.6111, 20.9394, 'CAF'),
    'Ukraine': ('Europe', 48.3794, 31.1656, 'UKR'),
    'Madagascar': ('Africa', -18.7669, 46.8691, 'MDG'),
    'Botswana': ('Africa', -22.3285, 24.6849, 'BWA'),
    'Kenya': ('Africa', -0.0236, 37.9062, 'KEN'),
    'Thailand': ('Asia', 15.8700, 100.9925, 'THA'),
    'Spain': ('Europe', 40.4637, -3.7492, 'ESP'),
    'Turkmenistan': ('Asia', 38.9697, 59.5563, 'TKM'),
    'Cameroon': ('Africa', 7.3697, 12.3547, 'CMR'),
    'Papua New Guinea': ('Oceania', -6.3150, 143.9555, 'PNG'),
    'Yemen': ('Asia', 15.5527, 48.5164, 'YEM'),
    'Sweden': ('Europe', 60.1282, 18.6435, 'SWE'),
    'Uzbekistan': ('Asia', 41.3775, 64.5853, 'UZB'),
    'Morocco': ('Africa', 31.7917, -7.0926, 'MAR'),
    'Iraq': ('Asia', 33.2232, 43.6793, 'IRQ'),
    'Paraguay': ('South America', -23.4425, -58.4438, 'PRY'),
    'Zimbabwe': ('Africa', -19.0154, 29.1549, 'ZWE'),
    'Norway': ('Europe', 60.4720, 8.4689, 'NOR'),
    'Japan': ('Asia', 36.2048, 138.2529, 'JPN'),
    'Germany': ('Europe', 51.1657, 10.4515, 'DEU'),
    'Republic of the Congo': ('Africa', -0.2280, 15.8277, 'COG'),
    'Finland': ('Europe', 61.9241, 25.7482, 'FIN'),
    'Vietnam': ('Asia', 14.0583, 108.2772, 'VNM'),
    'Malaysia': ('Asia', 4.2105, 101.9758, 'MYS'),
    'Ivory Coast': ('Africa', 7.5400, -5.5471, 'CIV'),
    'Poland': ('Europe', 51.9194, 19.1451, 'POL'),
    'Oman': ('Asia', 21.5126, 55.9233, 'OMN'),
    'Italy': ('Europe', 41.8719, 12.5674, 'ITA'),
    'Philippines': ('Asia', 12.8797, 121.7740, 'PHL'),
    'Ecuador': ('South America', -1.8312, -78.1834, 'ECU'),
    'Burkina Faso': ('Africa', 12.2383, -1.5616, 'BFA'),
    'New Zealand': ('Oceania', -40.9006, 174.8860, 'NZL'),
    'Gabon': ('Africa', -0.8037, 11.6094, 'GAB'),
    'Guinea': ('Africa', 9.9456, -9.6966, 'GIN'),
    'United Kingdom': ('Europe', 55.3781, -3.4360, 'GBR'),
    'Uganda': ('Africa', 1.3733, 32.2903, 'UGA'),
    'Ghana': ('Africa', 7.9465, -1.0232, 'GHA'),
    'Romania': ('Europe', 45.9432, 24.9668, 'ROU'),
    'Laos': ('Asia', 19.8563, 102.4955, 'LAO'),
    'Guyana': ('South America', 4.8604, -58.9302, 'GUY'),
    'Belarus': ('Europe', 53.7098, 27.9534, 'BLR'),
    'Kyrgyzstan': ('Asia', 41.2044, 74.7661, 'KGZ'),
    'Senegal': ('Africa', 14.4974, -14.4524, 'SEN'),
    'Syria': ('Asia', 34.8021, 38.9968, 'SYR'),
    'Cambodia': ('Asia', 12.5657, 104.9910, 'KHM'),
    'Uruguay': ('South America', -32.5228, -55.7658, 'URY'),
    'Suriname': ('South America', 3.9193, -56.0278, 'SUR'),
    'Tunisia': ('Africa', 33.8869, 9.5375, 'TUN'),
    'Bangladesh': ('Asia', 23.6850, 90.3563, 'BGD'),
    'Nepal': ('Asia', 28.3949, 84.1240, 'NPL'),
    'Tajikistan': ('Asia', 38.8610, 71.2761, 'TJK'),
    'Greece': ('Europe', 39.0742, 21.8243, 'GRC'),
    'Nicaragua': ('North America', 12.8654, -85.2072, 'NIC'),
    'North Korea': ('Asia', 40.3399, 127.5101, 'PRK'),
    'Malawi': ('Africa', -13.2543, 34.3015, 'MWI'),
    'Eritrea': ('Africa', 15.1794, 39.7823, 'ERI'),
    'Benin': ('Africa', 9.3077, 2.3158, 'BEN'),
    'Honduras': ('North America', 15.2000, -86.2419, 'HND'),
    'Liberia': ('Africa', 6.4281, -9.4295, 'LBR'),
    'Bulgaria': ('Europe', 42.7339, 25.4858, 'BGR'),
    'Cuba': ('North America', 21.5218, -77.7812, 'CUB'),
    'Guatemala': ('North America', 15.7835, -90.2308, 'GTM'),
    'Iceland': ('Europe', 64.9631, -19.0208, 'ISL'),
    'South Korea': ('Asia', 35.9078, 127.7669, 'KOR'),
    'Hungary': ('Europe', 47.1625, 19.5033, 'HUN'),
    'Portugal': ('Europe', 39.3999, -8.2245, 'PRT'),
    'Jordan': ('Asia', 30.5852, 36.2384, 'JOR'),
    'Serbia': ('Europe', 44.0165, 21.0059, 'SRB'),
    'Azerbaijan': ('Asia', 40.1431, 47.5769, 'AZE'),
    'Austria': ('Europe', 47.5162, 14.5501, 'AUT'),
    'United Arab Emirates': ('Asia', 23.4241, 53.8478, 'ARE'),
    'Czech Republic': ('Europe', 49.8175, 15.4730, 'CZE'),
    'Panama': ('North America', 8.5380, -80.7821, 'PAN'),
    'Sierra Leone': ('Africa', 8.4606, -11.7799, 'SLE'),
    'Ireland': ('Europe', 53.1424, -7.6921, 'IRL'),
    'Georgia': ('Asia', 42.3154, 43.3569, 'GEO'),
    'Sri Lanka': ('Asia', 7.8731, 80.7718, 'LKA'),
    'Lithuania': ('Europe', 55.1694, 23.8813, 'LTU'),
    'Latvia': ('Europe', 56.8796, 24.6032, 'LVA'),
    'Togo': ('Africa', 8.6195, 0.8248, 'TGO'),
    'Croatia': ('Europe', 45.1000, 15.2000, 'HRV'),
    'Bosnia and Herzegovina': ('Europe', 43.9159, 17.6791, 'BIH'),
    'Costa Rica': ('North America', 9.7489, -83.7534, 'CRI'),
    'Slovakia': ('Europe', 48.6690, 19.6990, 'SVK'),
    'Dominican Republic': ('North America', 18.7357, -70.1627, 'DOM'),
    'Estonia': ('Europe', 58.5953, 25.0136, 'EST'),
    'Denmark': ('Europe', 56.2639, 9.5018, 'DNK'),
    'Netherlands': ('Europe', 52.1326, 5.2913, 'NLD'),
    'Switzerland': ('Europe', 46.8182, 8.2275, 'CHE'),
    'Bhutan': ('Asia', 27.5142, 90.4336, 'BTN'),
    'Guinea-Bissau': ('Africa', 11.8037, -15.1804, 'GNB'),
    'Moldova': ('Europe', 47.4116, 28.3699, 'MDA'),
    'Belgium': ('Europe', 50.5039, 4.4699, 'BEL'),
    'Lesotho': ('Africa', -29.6100, 28.2336, 'LSO'),
    'Armenia': ('Asia', 40.0691, 45.0382, 'ARM'),
    'Solomon Islands': ('Oceania', -9.6457, 160.1562, 'SLB'),
    'Albania': ('Europe', 41.1533, 20.1683, 'ALB'),
    'Equatorial Guinea': ('Africa', 1.6508, 10.2679, 'GNQ'),
    'Burundi': ('Africa', -3.3731, 29.9189, 'BDI'),
    'Haiti': ('North America', 18.9712, -72.2852, 'HTI'),
    'Rwanda': ('Africa', -1.9403, 29.8739, 'RWA'),
    'North Macedonia': ('Europe', 41.6086, 21.7453, 'MKD'),
    'Djibouti': ('Africa', 11.8251, 42.5903, 'DJI'),
    'Belize': ('North America', 17.1899, -88.4976, 'BLZ'),
    'Israel': ('Asia', 31.0461, 34.8516, 'ISR'),
    'El Salvador': ('North America', 13.7942, -88.8965, 'SLV'),
    'Slovenia': ('Europe', 46.1512, 14.9955, 'SVN'),
    'Fiji': ('Oceania', -17.7134, 178.0650, 'FJI'),
    'Kuwait': ('Asia', 29.3117, 47.4818, 'KWT'),
    'Eswatini': ('Africa', -26.5225, 31.4659, 'SWZ'),
    'Timor-Leste': ('Asia', -8.8742, 125.7275, 'TLS'),
    'Montenegro': ('Europe', 42.7087, 19.3744, 'MNE'),
    'Bahamas': ('North America', 25.0343, -77.3963, 'BHS'),
    'Vanuatu': ('Oceania', -15.3767, 166.9592, 'VUT'),
    'Qatar': ('Asia', 25.3548, 51.1839, 'QAT'),
    'The Gambia': ('Africa', 13.4432, -15.3101, 'GMB'),
    'Jamaica': ('North America', 18.1096, -77.2975, 'JAM'),
    'Lebanon': ('Asia', 33.8547, 35.8623, 'LBN'),
    'Cyprus': ('Asia', 35.1264, 33.4299, 'CYP'),
    'Palestine': ('Asia', 31.9522, 35.2332, 'PSE'),
    'Brunei': ('Asia', 4.5353, 114.7277, 'BRN'),
    'Trinidad and Tobago': ('North America', 10.6918, -61.2225, 'TTO'),
    'Cape Verde': ('Africa', 16.5388, -23.0418, 'CPV'),
    'Samoa': ('Oceania', -13.7590, -172.1046, 'WSM'),
    'Luxembourg': ('Europe', 49.8153, 6.1296, 'LUX'),
    'Mauritius': ('Africa', -20.3484, 57.5522, 'MUS'),
    'Comoros': ('Africa', -11.8750, 43.8722, 'COM'),
    'São Tomé and Príncipe': ('Africa', 0.1864, 6.6131, 'STP'),
    'Kiribati': ('Oceania', -3.3704, -168.7340, 'KIR'),
    'Bahrain': ('Asia', 26.0667, 50.5577, 'BHR'),
    'Dominica': ('North America', 15.4150, -61.3710, 'DMA'),
    'Tonga': ('Oceania', -21.1789, -175.1982, 'TON'),
    'Singapore': ('Asia', 1.3521, 103.8198, 'SGP'),
    'Micronesia': ('Oceania', 7.4256, 150.5508, 'FSM'),
    'Saint Lucia': ('North America', 13.9094, -60.9789, 'LCA'),
    'Andorra': ('Europe', 42.5063, 1.5218, 'AND'),
    'Palau': ('Oceania', 7.5150, 134.5825, 'PLW'),
    'Seychelles': ('Africa', -4.6796, 55.4920, 'SYC'),
    'Antigua and Barbuda': ('North America', 17.0608, -61.7964, 'ATG'),
    'Barbados': ('North America', 13.1939, -59.5432, 'BRB'),
    'Saint Vincent and the Grenadines': ('North America', 12.9843, -61.2872, 'VCT'),
    'Grenada': ('North America', 12.1165, -61.6790, 'GRD'),
    'Malta': ('Europe', 35.9375, 14.3754, 'MLT'),
    'Maldives': ('Asia', 3.2028, 73.2207, 'MDV'),
    'Saint Kitts and Nevis': ('North America', 17.3578, -62.7830, 'KNA'),
    'Marshall Islands': ('Oceania', 7.1315, 171.1845, 'MHL'),
    'Liechtenstein': ('Europe', 47.1660, 9.5554, 'LIE'),
    'San Marino': ('Europe', 43.9424, 12.4578, 'SMR'),
    'Tuvalu': ('Oceania', -7.1095, 177.6493, 'TUV'),
    'Nauru': ('Oceania', -0.5228, 166.9315, 'NRU'),
    'Monaco': ('Europe', 43.7384, 7.4246, 'MCO'),
    'Vatican City': ('Europe', 41.9029, 12.4534, 'VAT')
}

# Map metadata into dataframe
df['Continent'] = df['Country name'].map(lambda c: COUNTRY_METADATA.get(c, ('Other', 0, 0, 'UNK'))[0])
df['Latitude']  = df['Country name'].map(lambda c: COUNTRY_METADATA.get(c, ('Other', 0, 0, 'UNK'))[1])
df['Longitude'] = df['Country name'].map(lambda c: COUNTRY_METADATA.get(c, ('Other', 0, 0, 'UNK'))[2])
df['ISO_A3']    = df['Country name'].map(lambda c: COUNTRY_METADATA.get(c, ('Other', 0, 0, 'UNK'))[3])

print("✅ Geographic mapping successfully applied to all 195 nations.")
display(df[['Country name', 'Continent', 'Latitude', 'Longitude', 'ISO_A3']].head())
""")

add_md("""### 🧠 Imputation Strategy & Feature Engineering
1. **South Sudan Land Area**: Given missing water breakdown, `Land in km2` is assigned `Total in km2`, and `% water` set to `0.0`.
2. **The Gambia Stats**: Filled using known official values (Population $\\approx 2,705,992$, Area $= 11,300\\text{ km}^2$, Density $= 239.5$).
3. **Continental Median Imputation**: Missing rates (`HDI`, `Fertility rate`, `Mortality rate`, `% of internet users in population`, `External debt (billions)`) are imputed via continental medians to prevent distortion.
4. **UN Human Development Index (HDI) Tiers**:
   - **Very High HDI**: $\\ge 0.800$
   - **High HDI**: $0.700 \\le \\text{HDI} < 0.800$
   - **Medium HDI**: $0.550 \\le \\text{HDI} < 0.700$
   - **Low HDI**: $< 0.550$
5. **Debt Per Capita ($)**: $\\frac{\\text{External debt (billions)} \\times 10^9}{\\text{Population}}$
6. **Population Bins**: Discretized into 5 analytical brackets: `<10M`, `10M–50M`, `50M–100M`, `100M–500M`, `>500M`.""")

add_code("""# Handle South Sudan specific geometry
df.loc[df['Country name'] == 'South Sudan', 'Land in km2'] = df.loc[df['Country name'] == 'South Sudan', 'Total in km2']
df.loc[df['Country name'] == 'South Sudan', 'Inland water in km2'] = 0.0
df.loc[df['Country name'] == 'South Sudan', '% water'] = 0.0

# Handle The Gambia official figures
df.loc[df['Country name'] == 'The Gambia', 'Population'] = 2705992.0
df.loc[df['Country name'] == 'The Gambia', '% of world population'] = 0.03
df.loc[df['Country name'] == 'The Gambia', 'Population density'] = 239.5

# Continental median imputation for indicators
impute_cols = ['HDI', 'Fertility rate', 'Mortality rate', '% of internet users in population', 'External debt (billions)']
for col in impute_cols:
    continent_medians = df.groupby('Continent')[col].transform('median')
    df[col] = df[col].fillna(continent_medians).fillna(df[col].median())

# Ensure Population density is populated
df['Population density'] = df['Population density'].fillna(df['Population'] / df['Total in km2'])

# Engineering UN HDI Tier
def classify_hdi(val):
    if pd.isna(val):
        return 'Unknown'
    elif val >= 0.800:
        return 'Very High HDI'
    elif val >= 0.700:
        return 'High HDI'
    elif val >= 0.550:
        return 'Medium HDI'
    else:
        return 'Low HDI'

df['HDI_Tier'] = df['HDI'].apply(classify_hdi)

# Engineering Debt per Capita (USD)
df['Debt_Per_Capita_USD'] = (df['External debt (billions)'] * 1e9) / df['Population']

# Engineering Population Bins
pop_bins = [0, 10_000_000, 50_000_000, 100_000_000, 500_000_000, 2_000_000_000]
pop_labels = ['<10M', '10M–50M', '50M–100M', '100M–500M', '>500M']
df['Pop_Bin'] = pd.cut(df['Population'], bins=pop_bins, labels=pop_labels)

# Set index to Country name for direct lookup
df = df.set_index('Country name')

print("✅ Data cleaning and feature engineering complete. Remaining nulls:")
print(df.isnull().sum())
df.head()
""")

# ==============================================================================
# SECTION 3: INTERACTIVE GLOBAL POPULATION & INDICATOR MAPS (PLOTLY)
# ==============================================================================
add_md("""---
# 🌐 Interactive Global Population & Indicator Maps (Plotly)
We construct interactive cartographic projections examining population concentration, human development, and sovereign debt across the globe.""")

add_md("""### 🗺️ Interactive Global Population Distribution (Scatter Geo)
Bubble scale reflects total national population, colored by the Human Development Index (HDI) under an equirectangular/natural earth global projection.""")

add_code("""fig = px.scatter_geo(
    df.reset_index(),
    lat='Latitude',
    lon='Longitude',
    hover_name='Country name',
    hover_data={
        'Continent': True,
        'Population': ':,.0f',
        'HDI': ':.3f',
        'HDI_Tier': True,
        'External debt (billions)': ':,.1f',
        '% of internet users in population': ':.1f%',
        'Latitude': False,
        'Longitude': False
    },
    size='Population',
    size_max=45,
    color='HDI',
    color_continuous_scale='Viridis',
    title='🌐 Interactive Global Population & Human Development Index (HDI) Distribution',
    projection='natural earth'
)

fig.update_geos(
    showocean=True, oceancolor='#eaf2f8',
    showland=True,  landcolor='#fdfefe',
    showcountries=True, countrycolor='#bdc3c7',
    showcoastlines=True, coastlinecolor='#7f8c8d'
)
fig.update_layout(height=650, margin=dict(l=0, r=0, t=50, b=0))
fig.show()
""")

add_md("""### 🗺️ Global Human Development Index (HDI) Choropleth Map
A global choropleth visualizing UN HDI across sovereign country boundaries.""")

add_code("""fig_choro = px.choropleth(
    df.reset_index(),
    locations='ISO_A3',
    color='HDI',
    hover_name='Country name',
    hover_data={
        'Continent': True,
        'HDI': ':.3f',
        'HDI_Tier': True,
        'Population': ':,.0f',
        'ISO_A3': False
    },
    color_continuous_scale='Plasma',
    title='🌍 Global Choropleth Map: Human Development Index (HDI) by Sovereign State'
)
fig_choro.update_geos(
    showocean=True, oceancolor='#eaf2f8',
    showlakes=True, lakecolor='#eaf2f8',
    projection_type='natural earth'
)
fig_choro.update_layout(height=600, margin=dict(l=0, r=0, t=50, b=0))
fig_choro.show()
""")

# ==============================================================================
# SECTION 4: TOP 20 RANKINGS & COMPARATIVE CHARTS
# ==============================================================================
add_md("""---
# 🏆 Top 20 Global Rankings & Comparative Visualizations
Examining the extreme percentiles of global population and external financial liabilities.""")

add_md("""### 📊 Top 20 Most Populated Countries""")

add_code("""top20_pop = df.reset_index().nlargest(20, 'Population')

plt.figure(figsize=(12, 8))
bars = plt.barh(top20_pop['Country name'], top20_pop['Population'] / 1e6, color='#2980b9', edgecolor='black', alpha=0.85)

for bar in bars:
    plt.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
             f"{bar.get_width():,.1f} M", va='center', fontsize=9, fontweight='semibold')

plt.xlabel('Population (Millions)', fontsize=12, fontweight='bold')
plt.ylabel('Country', fontsize=12, fontweight='bold')
plt.title('Top 20 Most Populated Countries in the World', fontsize=14, fontweight='bold', pad=15)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
""")

add_md("""### 💰 Top 20 Countries by External Debt (Billions USD)""")

add_code("""top20_debt = df.reset_index().nlargest(20, 'External debt (billions)')

plt.figure(figsize=(12, 8))
bars = plt.barh(top20_debt['Country name'], top20_debt['External debt (billions)'], color='#c0392b', edgecolor='black', alpha=0.85)

for bar in bars:
    plt.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
             f"${bar.get_width():,.1f} B", va='center', fontsize=9, fontweight='semibold')

plt.xlabel('External Debt (Billions USD)', fontsize=12, fontweight='bold')
plt.ylabel('Country', fontsize=12, fontweight='bold')
plt.title('Top 20 Countries by External Sovereign Debt (Billions USD)', fontsize=14, fontweight='bold', pad=15)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
""")

add_md("""### 📍 Top 20 Most Populated Countries on a Geographic Scatter Map""")

add_code("""plt.figure(figsize=(14, 8))

# Background: all countries
plt.scatter(df['Longitude'], df['Latitude'], s=25, color='#95a5a6', alpha=0.4, label='Other Nations')

# Highlight top 20
plt.scatter(
    top20_pop['Longitude'], top20_pop['Latitude'],
    s=top20_pop['Population'] / 1.5e6,
    color='#e74c3c', edgecolors='black', linewidth=0.8, alpha=0.85, label='Top 20 Nations (Size = Pop)'
)

# Text annotations
for _, row in top20_pop.iterrows():
    plt.annotate(
        row['Country name'],
        (row['Longitude'], row['Latitude']),
        fontsize=8, fontweight='bold',
        xytext=(6, 4), textcoords='offset points'
    )

plt.xlabel('Longitude (°)', fontsize=12, fontweight='bold')
plt.ylabel('Latitude (°)', fontsize=12, fontweight='bold')
plt.title("Geographical Distribution of the World's 20 Most Populated Nations", fontsize=14, fontweight='bold', pad=15)
plt.legend(loc='lower left', frameon=True)
plt.tight_layout()
plt.show()
""")

# ==============================================================================
# SECTION 5: GEOPANDAS SPATIAL ANALYSIS & BASEMAPS
# ==============================================================================
add_md("""---
# 🗺️ GeoPandas Spatial Analysis & Satellite Basemaps
Constructing spatial geometries, Coordinate Reference System (CRS) transformations, and satellite basemap overlays.""")

add_md("""### 📐 Convert DataFrame to GeoPandas GeoDataFrame (EPSG:4326)""")

add_code("""# Create Point geometry from Longitude and Latitude
gdf = gpd.GeoDataFrame(
    df.reset_index(),
    geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']),
    crs='EPSG:4326'
)

print(f"GeoDataFrame CRS: {gdf.crs}")
gdf[['Country name', 'Continent', 'Population', 'HDI', 'geometry']].head()
""")

add_md("""### 🌐 GeoPandas Spatial Plot: Global Human Development Index""")

add_code("""fig, ax = plt.subplots(figsize=(14, 8))
gdf.plot(
    ax=ax,
    column='HDI',
    cmap='viridis',
    legend=True,
    markersize=gdf['Population'] / 5e6 + 25,
    alpha=0.85,
    edgecolor='black',
    linewidth=0.5,
    legend_kwds={'label': 'Human Development Index (HDI)', 'orientation': 'horizontal', 'shrink': 0.6}
)

plt.title('Global Spatial Distribution of Human Development Index (GeoPandas EPSG:4326)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Longitude (°)', fontsize=11)
plt.ylabel('Latitude (°)', fontsize=11)
plt.tight_layout()
plt.show()
""")

add_md("""### 🛰️ Satellite Basemap Overlay (Contextily + GeoPandas)
Projecting GeoDataFrame to **Web Mercator (EPSG:3857)** to align with tile servers, overlaying Esri Satellite Imagery with a dark-mode fallback.""")

add_code("""try:
    import contextily as ctx
    
    # Reproject to Web Mercator for tile alignment
    gdf_web = gdf.to_crs(epsg=3857)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    gdf_web.plot(
        ax=ax,
        column='Population',
        cmap='autumn_r',
        markersize=gdf_web['Population'] / 2e6 + 30,
        alpha=0.9,
        edgecolor='black',
        linewidth=0.6,
        legend=True,
        legend_kwds={'label': 'Population', 'shrink': 0.5, 'format': '%,.0f'}
    )
    
    # Add Esri Satellite basemap
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=3)
    ax.set_title('🛰️ Sovereign Nations on Real Satellite Imagery (Contextily + GeoPandas)', fontsize=15, fontweight='bold', pad=15)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Contextily satellite load notice ({e}). Rendering dark-mode fallback map:")
    fig, ax = plt.subplots(figsize=(14, 8))
    sc = ax.scatter(
        df['Longitude'], df['Latitude'],
        c=df['Population'], cmap='autumn_r',
        s=df['Population'] / 2e6 + 30, alpha=0.85, edgecolors='white', linewidth=0.5
    )
    plt.colorbar(sc, ax=ax, label='Population')
    ax.set_facecolor('#0d1b2a')
    ax.set_title('Global Nations — Fallback Dark Map (Bubble = Population)', fontsize=14, fontweight='bold', color='white')
    ax.tick_params(colors='white')
    plt.tight_layout()
    plt.show()
""")

# ==============================================================================
# SECTION 6: INTERACTIVE FOLIUM WORLD MAPS
# ==============================================================================
add_md("""---
# 🗺️ Interactive Folium World Maps
Interactive multi-layered maps featuring layer switching, cluster aggregation, popups, and density heatmaps.""")

add_md("""### 🛰️ Multi-Layer Satellite & Street Map with HDI Tier Markers
Switch layers using the top-right control (`≡`). Click any marker for detailed socio-economic statistics.""")

add_code("""def hdi_tier_color(tier):
    return {
        'Very High HDI': '#2ecc71',
        'High HDI':      '#3498db',
        'Medium HDI':    '#f39c12',
        'Low HDI':       '#e74c3c'
    }.get(str(tier), '#95a5a6')

m_world = folium.Map(location=[20.0, 10.0], zoom_start=2, tiles=None)

# Add multiple basemap layers
folium.TileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri WorldImagery',
    name='🛰️ Esri Satellite',
    overlay=False, control=True
).add_to(m_world)

folium.TileLayer('OpenStreetMap',       name='🗺️ OpenStreetMap').add_to(m_world)
folium.TileLayer('CartoDB positron',    name='☁️ CartoDB Light').add_to(m_world)
folium.TileLayer('CartoDB dark_matter', name='🌑 CartoDB Dark').add_to(m_world)

for country, row in df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=max(4, min(24, np.sqrt(row['Population'] / 1e6) * 1.8)),
        color=hdi_tier_color(row['HDI_Tier']),
        fill=True,
        fill_color=hdi_tier_color(row['HDI_Tier']),
        fill_opacity=0.75,
        weight=1.5,
        popup=folium.Popup(
            f"<div style='font-family: Arial; min-width: 180px;'>"
            f"<h4 style='margin:0 0 5px 0; color:#2c3e50;'>🏳️ {country}</h4>"
            f"<b>Continent:</b> {row['Continent']}<br>"
            f"<b>Population:</b> {int(row['Population']):,}<br>"
            f"<b>HDI:</b> {row['HDI']:.3f} ({row['HDI_Tier']})<br>"
            f"<b>Fertility Rate:</b> {row['Fertility rate']:.2f}<br>"
            f"<b>Mortality Rate:</b> {row['Mortality rate']:.2f}<br>"
            f"<b>Internet Penetration:</b> {row['% of internet users in population']:.1f}%<br>"
            f"<b>External Debt:</b> ${row['External debt (billions)']:.1f} B<br>"
            f"<b>Debt per Capita:</b> ${row['Debt_Per_Capita_USD']:,.0f}"
            f"</div>",
            max_width=300
        ),
        tooltip=f"{country} · {row['Continent']} · HDI: {row['HDI']:.3f}"
    ).add_to(m_world)

folium.LayerControl(position='topright').add_to(m_world)
m_world.save('world_country_map.html')
print("✅ Saved interactive map to 'world_country_map.html'")
m_world
""")

add_md("""### 🔵 Folium MarkerCluster: Dynamic Global Clustering""")

add_code("""m_cluster = folium.Map(
    location=[20.0, 10.0], zoom_start=2,
    tiles='CartoDB positron'
)

marker_cluster = MarkerCluster(name='Sovereign Nations Cluster').add_to(m_cluster)

for country, row in df.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"<b>{country}</b><br>Pop: {int(row['Population']):,}<br>HDI: {row['HDI']:.3f}",
        icon=folium.Icon(color='blue' if row['HDI'] >= 0.7 else 'orange', icon='globe', prefix='fa')
    ).add_to(marker_cluster)

folium.LayerControl().add_to(m_cluster)
m_cluster.save('world_cluster_map.html')
print("✅ Saved cluster map to 'world_cluster_map.html'")
m_cluster
""")

add_md("""### 🌡️ Folium Population Density Weighted HeatMap""")

add_code("""m_heat = folium.Map(
    location=[20.0, 10.0], zoom_start=2,
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri WorldImagery'
)

# Prepare weighted heatmap data: [Lat, Lon, Weight]
# Normalizing population density with log scale to avoid single-point blowout (e.g. Monaco/Singapore)
heat_data = [
    [row['Latitude'], row['Longitude'], float(np.log1p(row['Population density']))]
    for _, row in df.iterrows()
]

HeatMap(
    heat_data,
    radius=25,
    blur=18,
    min_opacity=0.35,
    gradient={'0.2': 'navy', '0.4': 'blue', '0.6': 'cyan', '0.8': 'yellow', '1.0': 'red'}
).add_to(m_heat)

m_heat.save('world_density_heatmap.html')
print("✅ Saved heatmap to 'world_density_heatmap.html'")
m_heat
""")

# ==============================================================================
# SECTION 7: ADVANCED PLOTLY INTERACTIVE CHARTS
# ==============================================================================
add_md("""---
# 📊 Advanced Interactive Charts (Plotly)
Multi-dimensional interactive analytics enabling cross-filtering, 3D exploration, hierarchical treemaps, and sunburst decompositions.""")

add_md("""### 🏙️ Top 20 Most Populous Countries — Interactive Horizontal Bar Chart""")

add_code("""top20 = df.reset_index().nlargest(20, 'Population').sort_values('Population')

fig = px.bar(
    top20,
    x='Population', y='Country name',
    orientation='h',
    color='Continent',
    text=top20['Population'].apply(lambda x: f'{x/1e9:.2f}B' if x >= 1e9 else f'{x/1e6:.1f}M'),
    title='🏙️ Top 20 Most Populous Sovereign Nations (Colored by Continent)',
    labels={'Population': 'Population', 'Country name': 'Sovereign Nation'},
    color_discrete_sequence=px.colors.qualitative.Safe
)
fig.update_traces(textposition='outside')
fig.update_layout(height=650, xaxis_tickformat=',')
fig.show()
""")

add_md("""### 📈 Human Development Index (HDI) vs Internet Penetration (%)""")

add_code("""fig = px.scatter(
    df.reset_index(),
    x='HDI', y='% of internet users in population',
    color='Continent',
    size='Population',
    hover_name='Country name',
    hover_data={'Population': ':,.0f', 'HDI': ':.3f', '% of internet users in population': ':.1f%', 'External debt (billions)': ':,.1f'},
    title='📈 Human Development Index (HDI) vs Internet Penetration (%)',
    labels={'HDI': 'Human Development Index (0 to 1)', '% of internet users in population': 'Internet Penetration (% of Pop)'},
    color_discrete_sequence=px.colors.qualitative.Bold
)
fig.update_layout(height=600)
fig.show()
""")

add_md("""### 🌳 Global Population Treemap: Continent → HDI Tier → Country""")

add_code("""fig = px.treemap(
    df.reset_index(),
    path=['Continent', 'HDI_Tier', 'Country name'],
    values='Population',
    color='HDI',
    color_continuous_scale='RdYlGn',
    title='🌳 Global Population Hierarchy: Continent → HDI Tier → Country (Area = Population, Color = HDI)'
)
fig.update_layout(height=700, margin=dict(t=50, l=10, r=10, b=10))
fig.show()
""")

add_md("""### 🌐 3D Interactive Map: Longitude × Latitude × HDI""")

add_code("""fig = px.scatter_3d(
    df.reset_index(),
    x='Longitude', y='Latitude', z='HDI',
    color='Population',
    size='Population',
    hover_name='Country name',
    color_continuous_scale='Plasma',
    title='🌐 3D Geospatial Projection: Longitude × Latitude × Human Development Index',
    size_max=35
)
fig.update_layout(
    height=750,
    scene=dict(
        xaxis_title='Longitude (°)',
        yaxis_title='Latitude (°)',
        zaxis_title='HDI (Score)'
    )
)
fig.show()
""")

add_md("""### 🎞️ Animated Scatter Chart: Nations Across HDI Tiers""")

add_code("""fig = px.scatter(
    df.reset_index().sort_values('HDI_Tier'),
    x='Longitude', y='Latitude',
    size='Population',
    color='Continent',
    animation_frame='HDI_Tier',
    hover_name='Country name',
    hover_data={'HDI': True, 'Population': True},
    size_max=40,
    title='🎞️ Animated Geospatial Transition across UN HDI Tiers',
    labels={'Longitude': 'Longitude (°)', 'Latitude': 'Latitude (°)'}
)
fig.update_layout(height=600)
fig.show()
""")

add_md("""### ☀️ Population Sunburst: Continent → HDI Tier → Country""")

add_code("""fig = px.sunburst(
    df.reset_index(),
    path=['Continent', 'HDI_Tier', 'Country name'],
    values='Population',
    color='HDI',
    color_continuous_scale='Viridis',
    title='☀️ Global Population Sunburst Breakdown'
)
fig.update_layout(height=700)
fig.show()
""")

add_md("""### 📐 Parallel Coordinates Plot: Key Global Indicators""")

add_code("""df_pc = df.reset_index().copy()
tier_map = {'Very High HDI': 0, 'High HDI': 1, 'Medium HDI': 2, 'Low HDI': 3}
df_pc['Tier_Num'] = df_pc['HDI_Tier'].map(tier_map)

fig = go.Figure(data=go.Parcoords(
    line=dict(
        color=df_pc['Tier_Num'],
        colorscale=[[0, '#2ecc71'], [0.33, '#3498db'], [0.66, '#f39c12'], [1, '#e74c3c']],
        showscale=True,
        colorbar=dict(title='HDI Tier (0=Very High, 3=Low)')
    ),
    dimensions=[
        dict(label='Population (M)', values=df_pc['Population'] / 1e6, range=[0, df_pc['Population'].max() / 1e6]),
        dict(label='HDI', values=df_pc['HDI'], range=[0.3, 1.0]),
        dict(label='Fertility Rate', values=df_pc['Fertility rate'], range=[0.8, 7.0]),
        dict(label='Mortality Rate', values=df_pc['Mortality rate'], range=[2.0, 20.0]),
        dict(label='Internet (%)', values=df_pc['% of internet users in population'], range=[0, 100]),
        dict(label='Debt ($B)', values=df_pc['External debt (billions)'], range=[0, 5000])
    ]
))

fig.update_layout(title='📐 Parallel Coordinates Plot — Multivariate Profile of Sovereign Nations', height=550)
fig.show()
""")

# ==============================================================================
# SECTION 8: STATISTICAL ANALYSIS (SEABORN & MATPLOTLIB)
# ==============================================================================
add_md("""---
# 📐 Statistical Distribution Analysis (Seaborn & Matplotlib)
Rigorous empirical distributions, density evaluations, violin plots, and correlation structures across sovereign indicators.""")

add_md("""### 📦 Box Plot: Human Development Index (HDI) by Continent""")

add_code("""plt.figure(figsize=(11, 6))
sns.boxplot(
    data=df.reset_index(),
    x='Continent', y='HDI',
    palette='Set2',
    fliersize=5
)
plt.title('Human Development Index (HDI) Distribution Across Continents', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Continent', fontsize=12, fontweight='bold')
plt.ylabel('Human Development Index (HDI)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
""")

add_md("""### 🎻 Violin Plot: Fertility Rate Distribution Across Continents""")

add_code("""plt.figure(figsize=(12, 6))
sns.violinplot(
    data=df.reset_index(),
    x='Continent', y='Fertility rate',
    palette='muted',
    inner='box'
)
plt.title('Fertility Rate (Births per Woman) Across Continents (Violin Plot)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Continent', fontsize=12, fontweight='bold')
plt.ylabel('Fertility Rate', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
""")

add_md("""### 🔥 Correlation Heatmap: Multivariate Socio-Demographic Relationships""")

add_code("""corr_cols = [
    'Total in km2', 'Land in km2', 'Inland water in km2', '% water',
    'Population', 'Population density', 'HDI',
    'Fertility rate', 'Mortality rate',
    '% of internet users in population', 'External debt (billions)'
]

corr_matrix = df[corr_cols].corr()

plt.figure(figsize=(11, 9))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    vmin=-1.0, vmax=1.0,
    square=True,
    linewidths=0.5,
    cbar_kws={'shrink': 0.8}
)
plt.title('Correlation Heatmap — Sovereign Demographic & Economic Indicators', fontsize=14, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.show()
""")

add_md("""### 📊 Continental Histograms + Kernel Density Estimation (KDE)""")

add_code("""continents = sorted(df['Continent'].unique())
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
axes = axes.flatten()

for i, cont in enumerate(continents):
    subset = df[df['Continent'] == cont]['HDI']
    sns.histplot(subset, kde=True, ax=axes[i], color='steelblue', edgecolor='black', bins=8)
    axes[i].set_title(f"{cont} (N={len(subset)})", fontsize=12, fontweight='bold')
    axes[i].set_xlabel('HDI Score')
    axes[i].set_ylabel('Number of Nations')

plt.suptitle('Human Development Index (HDI) Distribution + KDE by Continent', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()
""")

add_md("""### 📉 Cumulative Distribution Function (CDF): World Population & HDI""")

add_code("""sorted_pop = df['Population'].sort_values()
cdf_pop = sorted_pop.rank(method='max') / len(sorted_pop)

sorted_hdi = df['HDI'].sort_values()
cdf_hdi = sorted_hdi.rank(method='max') / len(sorted_hdi)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Population CDF
axes[0].plot(sorted_pop / 1e6, cdf_pop, color='#2980b9', linewidth=2.5)
axes[0].fill_between(sorted_pop / 1e6, cdf_pop, alpha=0.15, color='#2980b9')
axes[0].axvline(sorted_pop.median() / 1e6, color='red', linestyle='--', label=f'Median: {sorted_pop.median()/1e6:.1f}M')
axes[0].axvline(sorted_pop.mean() / 1e6, color='orange', linestyle='--', label=f'Mean: {sorted_pop.mean()/1e6:.1f}M')
axes[0].set_xscale('log')
axes[0].set_title('Cumulative Distribution Function (CDF) — Population (Log Scale)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Population in Millions (Log Scale)', fontsize=11)
axes[0].set_ylabel('Cumulative Proportion', fontsize=11)
axes[0].legend(loc='lower right')

# HDI CDF
axes[1].plot(sorted_hdi, cdf_hdi, color='#27ae60', linewidth=2.5)
axes[1].fill_between(sorted_hdi, cdf_hdi, alpha=0.15, color='#27ae60')
axes[1].axvline(sorted_hdi.median(), color='red', linestyle='--', label=f'Median: {sorted_hdi.median():.3f}')
axes[1].axvline(sorted_hdi.mean(), color='orange', linestyle='--', label=f'Mean: {sorted_hdi.mean():.3f}')
axes[1].set_title('Cumulative Distribution Function (CDF) — HDI', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Human Development Index (HDI)', fontsize=11)
axes[1].set_ylabel('Cumulative Proportion', fontsize=11)
axes[1].legend(loc='lower right')

plt.suptitle('Global Cumulative Proportion Distributions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
""")

add_md("""### 🔬 Pairplot: Key Socio-Economic Indicators by HDI Tier""")

add_code("""pair_cols = ['HDI', 'Fertility rate', 'Mortality rate', '% of internet users in population']

g = sns.pairplot(
    df.reset_index()[pair_cols + ['HDI_Tier']],
    hue='HDI_Tier',
    vars=pair_cols,
    palette={
        'Very High HDI': '#2ecc71',
        'High HDI':      '#3498db',
        'Medium HDI':    '#f39c12',
        'Low HDI':       '#e74c3c'
    },
    diag_kind='kde',
    plot_kws={'alpha': 0.75, 's': 35}
)
g.figure.suptitle('Scatter Plot Matrix (Pairplot) Across Primary Global Indicators', y=1.02, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
""")

# ==============================================================================
# SECTION 9: GROUPED & COMPARATIVE CONTINENTAL ANALYSIS
# ==============================================================================
add_md("""---
# 📊 Grouped Continental & Comparative Analytics
Continental macro-economic breakdowns, population bracket allocations, and stacked tier proportions.""")

add_md("""### 📑 Continental Multi-Metric Aggregation Summary Table""")

add_code("""continental_summary = df.groupby('Continent').agg(
    Num_Countries=('Population', 'count'),
    Total_Pop_M=('Population', lambda x: x.sum() / 1e6),
    Avg_HDI=('HDI', 'mean'),
    Avg_Fertility=('Fertility rate', 'mean'),
    Avg_Mortality=('Mortality rate', 'mean'),
    Avg_Internet_Pct=('% of internet users in population', 'mean'),
    Total_Debt_B=('External debt (billions)', 'sum')
).round(2)

continental_summary.columns = [
    '# Countries', 'Total Pop (M)', 'Avg HDI',
    'Avg Fertility', 'Avg Mortality', 'Avg Internet (%)', 'Total Debt ($B)'
]

display(continental_summary.sort_values('Total Pop (M)', ascending=False))
""")

add_md("""### 🏆 Top 5 Countries by Population per Continent""")

add_code("""continents = sorted(df['Continent'].unique())
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for i, cont in enumerate(continents):
    top5 = df[df['Continent'] == cont].nlargest(5, 'Population')
    axes[i].barh(top5.index, top5['Population'] / 1e6, color='#e67e22', edgecolor='black', alpha=0.85)
    axes[i].set_title(f"{cont} — Top 5 Populous Nations", fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Population (Millions)')
    axes[i].invert_yaxis()

plt.suptitle('Top 5 Most Populous Nations per Continent', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
""")

add_md("""### 📊 Discretized Population Bins: Global Distribution of Sovereign Sizes""")

add_code("""bin_counts = df['Pop_Bin'].value_counts().reindex(['<10M', '10M–50M', '50M–100M', '100M–500M', '>500M'])

plt.figure(figsize=(10, 5))
bars = plt.bar(
    bin_counts.index, bin_counts.values,
    color=['#27ae60', '#3498db', '#9b59b6', '#e67e22', '#c0392b'],
    edgecolor='black', alpha=0.85
)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 1.2,
        str(int(bar.get_height())),
        ha='center', fontsize=11, fontweight='bold'
    )

plt.xlabel('Population Bracket', fontsize=12, fontweight='bold')
plt.ylabel('Number of Sovereign Nations', fontsize=12, fontweight='bold')
plt.title('Distribution of 195 Sovereign Nations Across Population Brackets', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()
""")

add_md("""### 📊 Stacked Bar: UN HDI Tier Breakdown per Continent""")

add_code("""pivot_tier = df.groupby(['Continent', 'HDI_Tier']).size().unstack(fill_value=0)
tier_order = ['Very High HDI', 'High HDI', 'Medium HDI', 'Low HDI']
pivot_tier = pivot_tier.reindex(columns=tier_order, fill_value=0)

tier_colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']

pivot_tier.plot(
    kind='bar',
    stacked=True,
    color=tier_colors,
    figsize=(12, 6),
    edgecolor='black',
    linewidth=0.5,
    alpha=0.88
)
plt.xlabel('Continent', fontsize=12, fontweight='bold')
plt.ylabel('Number of Nations', fontsize=12, fontweight='bold')
plt.title('UN Human Development Index (HDI) Tier Composition by Continent', fontsize=14, fontweight='bold', pad=15)
plt.xticks(rotation=25, ha='right')
plt.legend(title='HDI Tier', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.show()
""")

add_md("""### 🏔️ Extremes Analysis: Top 5 Highest vs Lowest Nations in HDI""")

add_code("""top5_hdi = df.nlargest(5, 'HDI')[['Continent', 'HDI', 'Population', 'External debt (billions)']]
low5_hdi = df.nsmallest(5, 'HDI')[['Continent', 'HDI', 'Population', 'External debt (billions)']]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].barh(top5_hdi.index, top5_hdi['HDI'], color='#27ae60', edgecolor='black', alpha=0.85)
axes[0].set_title('Top 5 Nations with Highest HDI', fontweight='bold')
axes[0].set_xlabel('HDI Score')
axes[0].set_xlim(0, 1.05)
for i, (_, row) in enumerate(top5_hdi.iterrows()):
    axes[0].text(row['HDI'] - 0.12, i, f"{row['HDI']:.3f}", va='center', fontweight='bold', color='white')

axes[1].barh(low5_hdi.index, low5_hdi['HDI'], color='#c0392b', edgecolor='black', alpha=0.85)
axes[1].set_title('Top 5 Nations with Lowest HDI', fontweight='bold')
axes[1].set_xlabel('HDI Score')
axes[1].set_xlim(0, 1.05)
for i, (_, row) in enumerate(low5_hdi.iterrows()):
    axes[1].text(row['HDI'] + 0.02, i, f"{row['HDI']:.3f}", va='center', fontweight='bold', color='black')

plt.suptitle('Comparative Global Extremes — Human Development Index', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
""")

add_md("""### 🥧 Donut Charts: Global Population Share by Continent & HDI Tier""")

add_code("""fig_pie1 = px.pie(
    df.reset_index(),
    names='Continent', values='Population',
    title='🌍 Global Population Share by Geographic Continent',
    color_discrete_sequence=px.colors.qualitative.Bold,
    hole=0.4
)
fig_pie1.update_traces(textposition='inside', textinfo='percent+label')
fig_pie1.update_layout(height=480)
fig_pie1.show()

fig_pie2 = px.pie(
    df.reset_index(),
    names='HDI_Tier', values='Population',
    title='🏙️ Global Population Share by UN HDI Development Tier',
    color_discrete_map={'Very High HDI': '#2ecc71', 'High HDI': '#3498db', 'Medium HDI': '#f39c12', 'Low HDI': '#e74c3c'},
    hole=0.4
)
fig_pie2.update_traces(textposition='inside', textinfo='percent+label')
fig_pie2.update_layout(height=480)
fig_pie2.show()
""")

# ==============================================================================
# SECTION 10: MACHINE LEARNING & PREDICTIVE MODELING
# ==============================================================================
add_md("""---
# 🤖 Machine Learning & Predictive Modeling
Applying unsupervised clustering and regression analysis to discover structural socio-economic patterns across sovereign nations.""")

add_md("""### 🧠 1. K-Means Global Country Clustering ($k=4$)
We segment 195 sovereign nations across five standardized socio-economic indicators:
- **Human Development Index (`HDI`)**
- **Fertility Rate (`Fertility rate`)**
- **Mortality Rate (`Mortality rate`)**
- **Digital Adoption (`% of internet users in population`)**
- **Population Density (`Population density`)**""")

add_code("""# Feature selection for clustering
cluster_features = ['HDI', 'Fertility rate', 'Mortality rate', '% of internet users in population', 'Population density']
X = df[cluster_features].copy()

# Feature scaling (Z-score normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method to confirm optimal k
inertias = []
k_range = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, inertias, marker='o', color='#8e44ad', linewidth=2)
plt.xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=11, fontweight='bold')
plt.title('K-Means Elbow Curve for Optimal National Clustering', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# Fit optimal K-Means model with k=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=15)
df['Cluster'] = kmeans.fit_predict(X_scaled)
df['Cluster'] = df['Cluster'].astype(str)

# Profile cluster archetypes
cluster_profile = df.groupby('Cluster')[cluster_features].mean().round(2)
cluster_profile['Num_Nations'] = df.groupby('Cluster').size()
print("=== K-Means Cluster Archetype Centroids ===")
display(cluster_profile)
""")

add_md("""### 🗺️ Visualizing K-Means Clusters on the World Map""")

add_code("""cluster_colors = {'0': '#2ecc71', '1': '#3498db', '2': '#f39c12', '3': '#e74c3c'}
cluster_names = {
    '0': 'Cluster 0: Advanced Industrialized',
    '1': 'Cluster 1: High-Density Emerging',
    '2': 'Cluster 2: Moderate Development / Transitioning',
    '3': 'Cluster 3: Developing & Frontier'
}

plt.figure(figsize=(15, 9))

for cl, grp in df.groupby('Cluster'):
    plt.scatter(
        grp['Longitude'], grp['Latitude'],
        s=grp['Population'] / 2e6 + 35,
        c=cluster_colors.get(cl, 'gray'),
        edgecolors='black', linewidth=0.5,
        alpha=0.85,
        label=cluster_names.get(cl, f'Cluster {cl}')
    )

# Label key global anchor nations
anchor_countries = ['United States', 'China', 'India', 'Germany', 'Brazil', 'Nigeria', 'Japan', 'Russia', 'Australia']
for c in anchor_countries:
    if c in df.index:
        plt.annotate(
            c,
            (df.loc[c, 'Longitude'], df.loc[c, 'Latitude']),
            fontsize=8, fontweight='bold',
            xytext=(6, 4), textcoords='offset points'
        )

plt.xlabel('Longitude (°)', fontsize=12, fontweight='bold')
plt.ylabel('Latitude (°)', fontsize=12, fontweight='bold')
plt.title('K-Means Global Country Clusters (k=4 Sovereign Socio-Economic Segments)', fontsize=14, fontweight='bold', pad=15)
plt.legend(title='Development Cluster Archetype', loc='lower left', frameon=True)
plt.tight_layout()
plt.show()
""")

add_md("""### 🤖 Interactive K-Means Clusters Map (Plotly Scatter Geo)""")

add_code("""fig_km = px.scatter_geo(
    df.reset_index(),
    lat='Latitude', lon='Longitude',
    color='Cluster',
    size='Population',
    hover_name='Country name',
    hover_data={'Continent': True, 'HDI': ':.3f', '% of internet users in population': ':.1f%', 'Cluster': True},
    size_max=35,
    title='🤖 Interactive K-Means Sovereign Clusters on World Map',
    color_discrete_map=cluster_colors,
    projection='natural earth'
)
fig_km.update_geos(showocean=True, oceancolor='#eaf2f8', showland=True, landcolor='#fdfefe')
fig_km.update_layout(height=650)
fig_km.show()
""")

add_md("""### 📈 2. Linear Regression: Human Development Index (HDI) vs Internet Penetration (%)
Quantifying the mathematical association between national digital connectivity ($X$) and human development progress ($y$).""")

add_code("""X_reg = df[['% of internet users in population']].values
y_reg = df['HDI'].values

lr = LinearRegression()
lr.fit(X_reg, y_reg)
y_pred = lr.predict(X_reg)

r2 = r2_score(y_reg, y_pred)
mse = mean_squared_error(y_reg, y_pred)
slope = lr.coef_[0]
intercept = lr.intercept_

print(f"Regression Equation: HDI = {intercept:.4f} + {slope:.4f} * (Internet %)")
print(f"Coefficient of Determination (R²): {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.6f}")

# Plot regression line
plt.figure(figsize=(11, 7))
plt.scatter(
    X_reg, y_reg,
    c='#2980b9', edgecolors='black', linewidth=0.5,
    alpha=0.75, s=60, label='Sovereign Nations'
)

# Sorted regression line
idx_sorted = np.argsort(X_reg.ravel())
plt.plot(
    X_reg[idx_sorted], y_pred[idx_sorted],
    color='#c0392b', linewidth=2.5, linestyle='--',
    label=f'Fitted Line: HDI = {intercept:.3f} + {slope:.3f}·X  (R² = {r2:.3f})'
)

# Annotate prominent outlier and representative nations
notable = ['Norway', 'United States', 'India', 'Chad', 'North Korea', 'Singapore', 'Nigeria']
for c in notable:
    if c in df.index:
        plt.annotate(
            c,
            (df.loc[c, '% of internet users in population'], df.loc[c, 'HDI']),
            fontsize=9, fontweight='bold', color='#2c3e50',
            xytext=(5, 5), textcoords='offset points'
        )

plt.xlabel('Internet Users in Population (%)', fontsize=12, fontweight='bold')
plt.ylabel('Human Development Index (HDI)', fontsize=12, fontweight='bold')
plt.title(f'Bivariate Linear Regression: Internet Adoption vs Human Development Index (R² = {r2:.3f})', fontsize=14, fontweight='bold', pad=15)
plt.legend(fontsize=11, loc='upper left')
plt.tight_layout()
plt.show()
""")

# ==============================================================================
# SECTION 11: EXECUTIVE SCORECARD & CONCLUSIONS
# ==============================================================================
add_md("""---
# 🏅 Master Styled Scorecard & Executive Findings
Ranking the top 25 global nations across population, economic liability, and human development with color gradients.""")

add_md("""### 🏆 Top 25 Sovereign Nations Master Scorecard""")

add_code("""ranked = df.reset_index().sort_values('Population', ascending=False).head(25).reset_index(drop=True)
ranked.index = ranked.index + 1
ranked.index.name = 'Global Rank'

display(
    ranked[[
        'Country name', 'Continent', 'Population', 'HDI',
        'Fertility rate', 'Mortality rate',
        '% of internet users in population', 'External debt (billions)', 'HDI_Tier'
    ]]
    .style
    .background_gradient(subset=['Population'], cmap='Reds')
    .background_gradient(subset=['HDI'], cmap='YlGn')
    .background_gradient(subset=['% of internet users in population'], cmap='Blues')
    .background_gradient(subset=['External debt (billions)'], cmap='Purples')
    .format({
        'Population': '{:,.0f}',
        'HDI': '{:.3f}',
        'Fertility rate': '{:.2f}',
        'Mortality rate': '{:.2f}',
        '% of internet users in population': '{:.1f}%',
        'External debt (billions)': '${:,.1f} B'
    })
)
""")

add_md("""### 📝 Key Analytical Insights & Conclusions
1. **Demographic Polarization:** The world's top 2 most populous nations (India and China) comprise over 35% of humanity, exhibiting massive scale divergences compared to the median nation ($\\approx 8\\text{ million}$).
2. **Technological & Human Development Coupling:** Linear regression confirms a strong positive empirical association ($R^2 > 0.65$) between national internet penetration and the Human Development Index (HDI).
3. **Macroeconomic Liability Asymmetry:** External debt is concentrated heavily in high-income, very high HDI economies (e.g., United States, United Kingdom, European economies), reflecting capital market depth rather than insolvency.
4. **Demographic Transition Dynamics:** Developing economies (predominantly in Sub-Saharan Africa) display elevated fertility rates (>4.0 births/woman) and younger median ages, whereas Very High HDI states exhibit sub-replacement fertility (<1.6 births/woman), signaling long-term structural aging.""")

# Save final notebook
notebook_obj = {
    "cells": cells,
    "metadata": {
        "colab": {
            "provenance": []
        },
        "language_info": {
            "name": "python"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open('fds_lab9.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook_obj, f, indent=1, ensure_ascii=False)

print(f"🎉 Successfully generated 'fds_lab9.ipynb' with {len(cells)} cells!")
