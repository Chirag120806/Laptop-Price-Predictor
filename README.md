# Laptop Price Predictor

A machine learning web app that estimates a laptop's price (in INR) from its specifications: brand, type, RAM, weight, screen size and resolution, touchscreen/IPS, CPU, storage (HDD/SSD), GPU and OS.

<!-- Add your live demo link here, e.g. **Live demo:** https://your-app.streamlit.app -->

## Project structure

```
Laptop-Price-Predictor/
├── app.py                  # Streamlit web app
├── requirements.txt        # Python dependencies
├── data/
│   └── laptop_data.csv     # Raw dataset (1,303 laptops)
├── models/
│   ├── pipe.pkl            # Trained scikit-learn pipeline
│   └── df.pkl              # Cleaned dataframe used to fill the app's dropdowns
└── notebooks/
    └── laptop_price_eda_model.ipynb   # EDA, feature engineering, model training
```

## Approach

1. **EDA and cleaning:** explored the raw dataset and cleaned the text columns (RAM, weight, memory, CPU, GPU, OS).
2. **Feature engineering:** extracted touchscreen and IPS flags, computed pixels per inch (PPI) from resolution and screen size, split storage into HDD and SSD, and grouped CPU, GPU and OS into brand categories.
3. **Modeling:** a scikit-learn `Pipeline` with one-hot encoding of categorical features followed by a regression model, trained on log-transformed price (80/20 train/test split).
4. **Deployment:** the saved pipeline is served through a Streamlit app for real-time predictions.

## Results

On the held-out test set: **R² = 0.80**, **MAE = 0.20** (on log-price).

## Run locally

```bash
git clone https://github.com/Chirag120806/Laptop-Price-Predictor.git
cd Laptop-Price-Predictor
pip install -r requirements.txt
streamlit run app.py
```

To retrain the model, open `notebooks/laptop_price_eda_model.ipynb` and run it from inside the `notebooks/` folder; it reads from `data/` and writes the pickle files to `models/`.

## Tech stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit
