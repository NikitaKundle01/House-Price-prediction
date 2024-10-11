
# House-Price-prediction

This project aims to develop a model to predict house prices in King County, USA, using linear regression. We will utilize a dataset from Kaggle containing various attributes of houses, including their selling price.

### Objectives:
- #### Data Exploration and Analysis:
     Gain insights into the data by performing statistical analysis, identifying missing values, visualizing relationships between features, and understanding the correlation between features and house prices.

- #### Data Preprocessing: 
    Prepare the data for modeling by handling categorical features, scaling numerical features, and splitting the data into training and testing sets.   
- #### Model Building: 
    Develop a linear regression model using the scikit-learn library. Train the model on the training data and make predictions on the test data.
- #### Model Evaluation:
    Assess the performance of the model using metrics like R-squared and Mean Squared Error (MSE).
- #### OLS Regression Analysis: 
    Perform Ordinary Least Squares (OLS) regression using the statsmodels library to gain further insights into the model's coefficients and statistical significance.
- #### Model Improvement: 
    Analyze model residuals and discuss potential improvements to enhance the prediction accuracy.



 Exploratory Data Analysis (EDA)




![App Screenshot](https://github.com/NikitaKundle01/House-Price-prediction/blob/main/output.png?raw=true)

The graph shows a scatter plot of the relationship between the price of a house and its square footage of living space. The x-axis represents the square footage of living space (sqft_living), and the y-axis represents the price.

Key Observations:

Positive Correlation: There is a clear positive correlation between price and square footage of living space. As the square footage increases, the price tends to increase as well. 

Outliers: There are a few outliers visible in the graph, which are data points that deviate significantly from the overall trend. 

Clustering: The data points seem to cluster around certain areas of the graph. This might indicate that there are different market segments or price ranges within the dataset.

Non-Linearity: While there is a general upward trend, the relationship between price and square footage doesn't appear to be perfectly linear. There might be a slight curve or non-linearity in the relationship, suggesting that the increase in price per square foot might not be constant.
Overall, the graph suggests that there is a strong positive relationship between price and square footage of living space, but there are also some variations and outliers to consider.

Additional Analysis:

Correlation Coefficient: To quantify the strength of the relationship, calculating the correlation coefficient would be helpful. A value close to 1 would indicate a strong positive correlation.

Regression Analysis: Fitting a regression line to the data could provide a more precise equation representing the relationship between price and square footage. This could be used to predict prices for houses of different sizes.




![App Screenshot](https://github.com/NikitaKundle01/House-Price-prediction/blob/main/output1.png?raw=true)

The graph shows a histogram of the distribution of house prices. The x-axis represents the price in millions of rupees, and the y-axis represents the count of houses within each price range.

Key Observations:

Right-Skewed Distribution: The distribution is heavily right-skewed, meaning that there is a long tail to the right. This indicates that most houses have relatively lower prices, while a smaller number of houses have significantly higher prices.

Peak Around 1 Million Rupees: The distribution has a clear peak around 1 million rupees, suggesting that this is the most common price range for houses in the dataset.

Long Tail: The long tail to the right indicates the presence of some high-priced houses, with prices exceeding 5 million rupees.

Skewness: The skewness of the distribution can be quantified using a statistical measure such as the skewness coefficient. A positive value would confirm the right-skewed nature of the data.
Overall, the histogram reveals that house prices in the dataset are not evenly distributed, but rather exhibit a concentration around lower prices with a significant proportion of higher-priced houses.



![App Screenshot](https://github.com/NikitaKundle01/House-Price-prediction/blob/main/output2.png?raw=true)


Key Observations:

Strong Positive Correlation: The most prominent feature is the strong positive correlation between price and sqft_living. This indicates that houses with larger square footage tend to have higher prices.

Other Positive Correlations: There are also positive correlations between price and bedrooms, bathrooms, grade, and sqft_above. 

Negative Correlations: Some variables have negative correlations with price, such as zipcode and lat. 

Weak Correlations: Many other variables show weak or no correlations with price, suggesting that they have little or no influence on the pricing of houses.

Additional Analysis:

Correlation Coefficients: The numerical values corresponding to each color in the matrix represent the correlation coefficients. These values can be used to quantify the strength of the correlations more precisely.

Hierarchical Clustering: Applying hierarchical clustering to the correlation matrix can help identify groups of variables that are highly correlated with each other.

Multiple Regression Analysis: To assess the combined influence of multiple variables on price, a multiple regression analysis can be conducted using the variables with significant correlations.


# House Price Prediction Using Linear Regression

This project explores the prediction of house prices in King County, USA, using linear regression. The model is built and evaluated using the "kc_house_data.csv" dataset, which can be downloaded from Kaggle.

## Project Overview

This project aims to predict house prices based on various features like square footage of the house, number of bedrooms, bathrooms, etc. The key steps involved are data exploration, visualization, and machine learning model implementation.

### Key Steps

1. **Data Loading and Exploration**:
    - The dataset is loaded into a pandas DataFrame.
    - Basic information such as data types, missing values, and descriptive statistics is provided.
    - Exploratory Data Analysis (EDA) is conducted to explore relationships between features, with a focus on the relationship between `price` and `sqft_living`.
    - Visualizations such as scatter plots, histograms, and correlation matrices are generated.
    - Data preparation includes handling missing values, converting data types, and dropping irrelevant columns.

2. **Machine Learning and Model Building**:
    - The data is split into training and testing sets.
    - A linear regression model is implemented using scikit-learn's `LinearRegression`.
    - The model is trained on the training data.
    - The performance of the model is evaluated using Mean Squared Error (MSE) and R-squared metrics on the test data.

3. **OLS Model (Optional)**:
    - An Ordinary Least Squares (OLS) regression model is fitted using the `statsmodels` library.
    - A detailed statistical summary of the OLS model is provided, including information about coefficients, p-values, and confidence intervals.

---

## Prerequisites

Ensure that you have the following prerequisites before running the notebook:

- **Python 3.x**
- Required Python Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - (Optional) `statsmodels` for OLS regression

---

## Installation

1. **Install Python libraries**:
   Install the necessary Python libraries by running the following command:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn statsmodels



2. **Download the Dataset**:Download the "kc_house_data.csv" dataset from Kaggle ([invalid URL removed]) and place it in the same directory as the notebook.

3.  **Run the Notebook**:
Open house_price_prediction.ipynb in a Jupyter Notebook environment.
Run the code cells sequentially (Shift+Enter or click the "Run" button).

The notebook will output various visualizations, model performance metrics, and (if applicable) the OLS model summary.


 
