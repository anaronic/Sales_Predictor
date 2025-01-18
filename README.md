# Big Mart Sales Predictor

## Overview
This machine learning model predicts sales for Big Mart outlets using various product and store features. The model utilizes XGBoost regression to provide accurate sales forecasts based on historical data.

## Features Used
The model takes into account the following features:
- Item_Identifier: Unique product ID
- Item_Weight: Weight of product
- Item_Fat_Content: Fat content category (Low Fat/Regular)
- Item_Visibility: Product visibility index in store
- Item_Type: Category of product
- Item_MRP: Maximum Retail Price
- Outlet_Identifier: Unique store ID
- Outlet_Establishment_Year: Year store was established
- Outlet_Size: Size of outlet (Small/Medium/High)
- Outlet_Location_Type: Type of location area
- Outlet_Type: Type of outlet (Grocery Store/Supermarket)

## Technical Details

### Data Preprocessing
1. **Missing Value Treatment**:
   - Item_Weight: Filled with mean value
   - Outlet_Size: Filled using mode based on Outlet_Type

2. **Feature Engineering**:
   - Standardized Item_Fat_Content categories
   - All categorical variables encoded using Label Encoding

3. **Data Split**:
   - Training set: 80% of data
   - Testing set: 20% of data

### Model Architecture
- Algorithm: XGBoost Regressor
- Random State: 2
- Default hyperparameters

### Performance Metrics
The model achieved the following performance metrics:
- R-squared score on training data: 0.94
- R-squared score on test data: 0.55

## Requirements
- Python 3.x
- Required libraries:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - xgboost

## Usage
1. **Data Preparation**:
```python
# Load and preprocess data
big_mart_data = pd.read_csv('Train.csv')
# Handle missing values and encode categorical features
```

2. **Model Training**:
```python
# Initialize and train the model
regressor = XGBRegressor()
regressor.fit(X_train, Y_train)
```

3. **Making Predictions**:
```python
# Predict sales
predictions = regressor.predict(X_test)
```

## Data Analysis Insights

### Numerical Features
- Item_Weight shows a normal distribution
- Item_Visibility has a right-skewed distribution
- Item_MRP shows a multimodal distribution
- Item_Outlet_Sales exhibits right-skewed distribution

### Categorical Features
- Multiple outlet establishment years represented
- Two main fat content categories: Low Fat and Regular
- Various item types with different frequencies
- Three outlet sizes: Small, Medium, High

## Future Improvements
1. Feature engineering to create more informative variables
2. Hyperparameter tuning for better model performance
3. Ensemble methods to improve prediction accuracy
4. Cross-validation for more robust evaluation

## Contributing
Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss the proposed changes.
