# 📈 Sales Prediction Model for Kaggle Competition

This project is a complete data preprocessing, feature engineering, and model training pipeline for the **Rossmann Store Sales** Kaggle competition. It uses Python and Keras (TensorFlow) to build a regression model that predicts sales based on historical store data.

---

## 📖 Overview
- Merges and cleans training/test/store datasets
- Fills missing values using smart strategies (median, 0-fill, etc.)
- Converts date features into meaningful numerical indicators
- Applies label encoding to categorical columns
- Scales data with Min-Max normalization
- Trains a neural network with batch normalization and dropout
- Uses early stopping and learning rate reduction to avoid overfitting

---

## 📊 Datasets Used
- `train.csv`: Historical training data
- `test.csv`: Test data for submission
- `store.csv`: Supplementary data with store-level attributes

---

## 🌐 Libraries Required
```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

---

## ⚖️ Data Preprocessing Steps

1. **Drop Unnecessary Columns**: Removes `Id` and `Customers` columns.
2. **Handle Missing Values**: Fills missing values using median or zero-fill strategies.
3. **Merge Datasets**: Combines store info with main train/test datasets.
4. **Outlier Removal**: Applies Z-Score method to remove outliers from numeric columns.
5. **Date Feature Engineering**: Extracts year, month, week, and other date-based features.
6. **Manual Categorical Encoding**: Encodes `StateHoliday` manually (a → 1, b → 2, etc.).
7. **Label Encoding**: Encodes other categorical features using `LabelEncoder`.
8. **Scaling**: Uses `MinMaxScaler` to scale both input features and target (`Sales`).

---

## 🔧 Model Architecture (Keras)
```python
Input
└➜ Dense(256, relu) + BatchNorm + Dropout(0.4)
    └➜ Dense(128, relu) + BatchNorm + Dropout(0.4)
        └➜ Dense(64, relu) + BatchNorm + Dropout(0.4)
            └➜ Dense(32, relu)
                └➜ Dense(1)
```

- Optimizer: Adam
- Loss: Mean Squared Error
- Metrics: Mean Absolute Error
- Callbacks: EarlyStopping, ReduceLROnPlateau

---

## ⚙️ Training Configuration
- Batch size: 32
- Epochs: 10 (can be increased)
- Validation split: 20%
- Callback: Early stopping after no improvement in 10 epochs

---

## 📊 Evaluation & Prediction
- Evaluates model using Mean Absolute Error
- Plots training and validation loss
- Predicts on the test set and inverts scaling
- Generates `submission.csv` with predicted sales

---

## 📅 Custom Metrics
Custom function `rmspe` is defined (though not used in model):
```python
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))
```

---

## 📂 Output Files
- `submission.csv`: Final Kaggle submission file with columns:
  - `Id`
  - `Sales`

---

## 📗 Notes & Tips
- Z-score threshold is set to 3 for outlier removal.
- You can experiment with one-hot encoding instead of label encoding.
- Early stopping and learning rate reduction help generalization.
- Model can be improved with hyperparameter tuning and feature importance analysis.

---

## 🤝 Contributions
This code is based on initial Colab work found [here](https://colab.research.google.com/drive/1yQUilrgNoviihoA-Haknnxhgk2OTFQhw).

---

## 🚀 Future Enhancements
- [ ] Try RandomForest or XGBoost ensemble models
- [ ] Tune network depth and width
- [ ] Use TimeSeriesSplit for CV
- [ ] Integrate SHAP or LIME for explainability

---

## 🚑 License
This project is shared for educational purposes only.

