```markdown
# 📊 Retail Price Elasticity Prediction with RNN

## 🎯 Project Overview
A deep learning model using Recurrent Neural Networks (RNNs) to predict price elasticity of demand in retail. It analyzes historical sales, competitor prices, and promotions to forecast demand responses to price changes, supporting optimal pricing strategies.

## 📈 Business Problem
Retailers face challenges in setting optimal prices, gauging demand sensitivity, countering competitors, and optimizing promotions. This project predicts **price elasticity** to enable data-driven decisions.

## 🏗️ Architecture
Input Data → Feature Engineering → Sequence Creation → Hybrid CNN-LSTM → Elasticity Prediction → Price Optimization
```
- **Hybrid CNN-LSTM**: Conv1D for patterns + Bidirectional LSTM for sequences.
- **Input**: 14-day sequences with 40+ features.
- **Output**: Continuous elasticity value.

## 📁 Project Structure
```
retail-price-elasticity/
├── data/ (raw CSV)
├── models/ (.h5 files, metrics)
├── preprocessing/ (.pkl files)
├── notebooks/ (EDA, FE, training)
├── src/ (preprocessing, modeling, utils)
├── results/ (plots, CSV outputs)
├── requirements.txt
└── price_elasticity_rnn.py
```


## 📊 Dataset
- **Source**: `retail_store_inventory.csv` (5K+ records, 14 days, 5 stores, 20 products).
- **Key Features**:
| Feature | Type | Description |
|---------|------|-------------|
| Date, Store/Product ID | DateTime/Categorical | Identifiers |
| Units Sold, Price, Discount | Numerical | Sales & pricing |
| Competitor Pricing, Holiday | Numerical/Binary | External factors |
| Weather, Seasonality | Categorical | Contextual |

## 🔧 Feature Engineering
- **Target**: Elasticity = `-(%ΔDemand / %ΔPrice)` (categorized: Highly Elastic to Giffen).
- **Additions**: Lags (1-30 days), rolling stats, interactions (e.g., Holiday × Price), temporal encodings.

## 🧠 Model Highlights
```python
# Core layers: Conv1D → BidLSTM → Dense
Sequential([Input((14,40)), Conv1D(64), BidLSTM(128), Dense(1)])
```
- **Training**: Adam optimizer, MSE loss, 50 epochs, early stopping.
- **Metrics**: R² >0.85, MAE <0.5 (test set).

| Class | MAE | Samples |
|-------|-----|---------|
| Highly Elastic | 0.42 | 150 |
| Elastic | 0.38 | 450 |
| Inelastic | 0.45 | 300 |
| Giffen/Veblen | 0.51 | 100 |

## 💡 Usage
### Predict Elasticity
```python
model = tf.keras.models.load_model('models/price_elasticity_rnn_model.h5')
elasticity = target_scaler.inverse_transform(model.predict(sequence))[0][0]
```

### Optimize Price
```python
from src.price_optimization import optimize_price
optimal = optimize_price(current_price=100, elasticity=-1.5, cost=60, margin=0.30)
```

## 🎯 Applications
- **Dynamic Pricing**: Adjust based on elasticity thresholds.
- **Promotions**: Deep discounts for elastic goods.
- **Inventory**: Boost stock for highly elastic items.

## 🔄 Retraining
```bash
python src/retraining_pipeline.py --new-data "data/new_sales.csv" --model-path "models/model.h5"
```
Monitor drift; retrain on accuracy drop.

## 📊 Interpretation
| Range | Sensitivity | Implication |
|-------|-------------|-------------|
| <-2.0 | Highly Elastic | Big demand shifts |
| -2.0 to -1.0 | Elastic | Moderate response |
| -1.0 to 0 | Inelastic | Stable demand |
| >0 | Giffen | Luxury/upward demand |


## 👥 Authors & Contact
-  [MariamHelal](https://github.com/layla3052004)
- Issues: [GitHub](https://github.com/layla3052004/DynamicPricing_RNN_Model)

**⭐ Star if useful! Next: API, dashboard, A/B integration.
```






















































