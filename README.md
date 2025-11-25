# Car Price Prediction

### 🔗 Live Demo: https://car-prediction-price.vercel.app/  
### 🔗 Backend API (Swagger UI): https://carpredictionprice-1.onrender.com/docs - initial 50s wait time

A complete end-to-end machine learning pipeline and web application for predicting the prices of used cars.  
It includes **custom web scraping**, dataset construction, outlier detection, feature engineering, model training,  
a FastAPI backend, and a fully deployed frontend interface.

---

# 📥 Data Collection & Custom Dataset

This project uses a **fully custom-built dataset**, created by scraping real car listings from online marketplaces.

### 🔎 Data Sources
Public automotive marketplaces containing listings for used cars.

### 🧰 Tools Used
- Python  
- Requests  
- BeautifulSoup (BS4)  
- Regex  
- Pandas  

### ✔️ Extracted Features
- Brand (marca)  
- Model  
- Year of manufacture  
- Mileage (km)  
- Fuel type  
- Transmission  
- Engine displacement / horsepower (if available)    
- Listing price (target)

### 🧹 Data Cleaning & Normalization
- Cleaned mileage and engine size into numeric values  
- Normalized categorical features  
- Removed corrupted or incomplete entries  
- Standardized price into EUR  
- Identified outliers using statistical methods (see below)

---

# ⚠️ Outlier Detection Strategy

Instead of forcing all prices into a **fixed range** (which hides real market variation),  
the project focused on **detecting and analyzing outliers**, because the dataset was **heavily skewed** due to:

- luxury brands with extremely high prices  
- damaged cars listed cheaply  
- inconsistent user-entered values  
- rare models with very low sample size  

### Why not clamp prices?
Clamping or trimming values makes the model always predict “average cars” and destroys real patterns.

### What I did instead:
- Applied **log-transformation** on price to reduce skew  
- Used **IQR filters** and **Z-score** to detect true anomalies  
- Removed only extreme, unrealistic entries  
- Tested **robust regression methods**  
- Compared models with and without outlier filtering  

This approach produced:
- higher stability  
- better generalization  
- improved predictions for high-end cars  
- reduced underestimation bias  

---

# 📊 How It Works (Architecture Diagram)
                     ┌────────────────────────┐
                     │   Car Listing Websites │
                     │   (Raw scraped data)    │
                     └──────────────┬─────────┘
                                    │
                                    ▼
                       ┌──────────────────────┐
                       │  Web Scraper (Python)│
                       │  - Requests          │
                       │  - BS4 parsing       │
                       │  - Cleaning          │
                       └───────────┬──────────┘
                                   │
                                   ▼
                  ┌──────────────────────────────────┐
                  │      Custom Dataset (CSV)         │
                  │   + Outlier detection             │
                  │   + Feature normalization         │
                  └──────────────────┬────────────────┘
                                     │
                                     ▼
                   ┌──────────────────────────────────┐
                   │   Machine Learning Pipeline      │
                   │  - Preprocessing                 │
                   │  - Feature engineering           │
                   │  - Model training & tuning       │
                   │  - Evaluation                    │
                   └──────────────────┬──────────────┘
                                     │
                                     ▼
                 ┌────────────────────────────────────┐
                 │      FastAPI Backend API           │
                 │   - `/predict`                     │
                 │   - `/predict-batch`               │
                 │   - Confidence intervals           │
                 └──────────────────┬─────────────────┘
                                    │
                                    ▼
                 ┌────────────────────────────────────┐
                 │  Frontend (Next.js on Vercel)      │
                 │  - Form UI                         │
                 │  - Fetch API                       │
                 │  - Display prediction              │
                 └────────────────────────────────────┘

---

# 🤖 Machine Learning Models Compared

Several models were trained and compared for performance and generalization.

### Models tested:

| Model Type | Notes | Result |
|------------|-------|--------|
| **Linear Regression** | Stable Curve, Misses Outliers  | Decent |
| **Random Forest** | Nonlinear model, Very good clustering | ⭐ Best overall |
| **Gradient Boosting** | Strong but sensitive to outliers | Mixed |
| **Decision Tree** | Too simple | Overfit |

### 📌 Final choice: **Random Forest*  
It gave the best balance between:
- training stability  
- explainability  
- robustness to limited dataset size  
- consistent predictions  

---

# ⭐ Feature Importance

The most impactful features on price prediction:

1. **Year of manufacture**  
2. **Age (years)**  
3. **Horsepower (marca)**  
4. **Modern Car**  
5. **Is it A Modern Car**  
6. **Engine Capacity**  
7. **Category 2010-2015**  
8. **Is Automatic**

### Key insights:
- Newer cars → significantly higher price  
- Higher mileage → strong negative correlation  
- Premium brands → consistently higher values  
- Fuel type affects value, especially electric
- 2018 > SUV's and Premium Brands very high price but after 2015 start to follow the curve  

---

# 📈 Model Performance

Final model performance:
RMSE: 3,797 EUR
MAE: 1,820 EUR
R² : 0.76
MAPE: 23.6%
Accuracy: ~76%


Performance improves with more data and better scraping coverage.

---

# 🛠️ Installation

```bash
git clone https://github.com/Zebyan/CarPredictionPrice.git
cd CarPredictionPrice
pip install -r requirements.txt
```
Go to prediction_models and train one of them
Copy the .pkl inside backend/model_storage
Start backend: uvicorn backend.main:app --reload

# 📂 Project Structure
CarPredictionPrice/
├── backend/
│   ├── models/
│   ├── services/
│   ├── routers/
│   └── ...
├── frontend/
│   ├── components/
│   ├── pages/
│   └── ...
├── prediction_models/
│   └── EDA_and_training.ipynb
├── data/
│   └── scraped_data.csv
└── README.md

# 📡 API Usage
Single Prediction:
POST /predict
{
    "marca": "Mercedes-Benz",
    "model": "C-Class",
    "an_fabricatie": 2019,
    "rulaj": 126983,
    "putere": 170,
    "capacitate_motor": 2200,
    "combustibil": "motorina",
    "caroserie": "sedan",
    "culoare": "gri",
    "cutie_viteza": "automata"
}
Batch Prediction:
[
  {
    "marca": "Skoda",
    "model": "Octavia",
    "an_fabricatie": 2004,
    "rulaj": 222549,
    "putere": 105,
    "capacitate_motor": 1900,
    "combustibil": "motorina",
    "caroserie": "sedan",
    "culoare": "rosu",
    "cutie_viteza": "manuala"
  },
  {
    "marca": "Kia",
    "model": "Sportage",
    "an_fabricatie": 2005,
    "rulaj": 211203,
    "putere": 115,
    "capacitate_motor": 1700,
    "combustibil": "motorina",
    "caroserie": "suv",
    "culoare": "verde",
    "cutie_viteza": "manuala"
  }
]

🤝 Contributing

This is a personal learning project. Feel free to fork, modify, and use the code for your own projects.

#👤 Author

Zebyan
GitHub: https://github.com/Zebyan





