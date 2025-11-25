# Car Price Prediction 
## Check it out here: https://car-prediction-price.vercel.app/
## Swagger UI Backend Documentation: https://carpredictionprice-1.onrender.com/docs

A machine learning project that predicts the price of used cars based on various features such as company name, model name, year of purchase, and other parameters.

## Overview

This project implements machine learning models to estimate used car prices. By analyzing historical car data and their characteristics, the model can predict market prices for new listings, helping both buyers and sellers make informed decisions.

## Features

- **Data Analysis**: Comprehensive exploratory data analysis (EDA) of car datasets
- **Feature Engineering**: Transformation and selection of relevant car features
- **Model Development**: Implementation of multiple regression models
- **Price Prediction**: Accurate prediction of used car prices
- **Linear Regression**: Primary algorithm for regression analysis

## Technologies Used

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning models and utilities
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development environment

## Dataset

The project uses car datasets containing the following typical features:
- Company/Brand name
- Model name
- Year of purchase
- Kilometers driven
- Fuel type
- Transmission type
- Owner count
- Price (target variable)

## Project Structure

```
CarPredictionPrice/
├── data/
│   └── [dataset files]
├── notebooks/
│   └── car_price_prediction.ipynb
├── models/
│   └── [trained models]
├── src/
│   └── [source code files]
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Zebyan/CarPredictionPrice.git
cd CarPredictionPrice
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Open and run the Jupyter notebook:
```bash
jupyter notebook
```

## Usage

1. Load the car dataset
2. Perform data preprocessing and cleaning
3. Execute exploratory data analysis
4. Train the regression model
5. Evaluate model performance
6. Make predictions on new data

## Model Performance

The model evaluates performance using metrics such as:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²) Score
- Mean Absolute Error (MAE)

## Contributing

This is a personal learning project. Feel free to fork, modify, and use the code for your own projects.

## License

This project is open source and available for educational and personal use.

## Author

Zebyan

## Acknowledgments

- Data sourced from automotive datasets
- Built for educational purposes in machine learning and data science

## Contact

For questions or suggestions about this project, please visit the [GitHub repository](https://github.com/Zebyan/CarPredictionPrice).
