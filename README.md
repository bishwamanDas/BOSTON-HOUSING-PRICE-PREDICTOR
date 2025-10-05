# 🏡 House Price Prediction – Multivariable Regression & Valuation Model

This project builds a machine learning model to predict house prices based on multiple features of residential properties. Using the classic **Boston Housing dataset**, we walk through a complete end-to-end pipeline — from data analysis and visualization to model training, evaluation, and property valuation.

The project also explores the benefits of **log-transformed regression** to improve residual distribution and predictive performance.

---

## 📊 Project Overview

The main objective is to **predict house prices** using a **multivariable linear regression model**.

### ✅ Key Highlights

- 📁 Data loading, cleaning, and validation  
- 📈 Exploratory Data Analysis (EDA) and visualization  
- 🧮 Linear regression modeling  
- 🔄 Log-transformation for better model performance  
- 🏷️ Feature coefficient analysis and interpretation  
- 💵 Predicting property valuation based on user-defined features  

---

## 🧰 Tech Stack

- **Python 3.x**  
- **Pandas** – data manipulation  
- **NumPy** – numerical computing  
- **Matplotlib & Seaborn** – data visualization  
- **Plotly** – interactive charts  
- **Scikit-learn** – machine learning tools  

---

## 📁 Dataset

The project uses the **Boston Housing Dataset**, which contains **506 data points** and various features affecting house prices, including:

| Feature  | Description |
|----------|-------------|
| `RM`     | Average number of rooms per dwelling |
| `LSTAT`  | % lower status of the population |
| `PTRATIO`| Pupil-teacher ratio by town |
| `DIS`    | Weighted distances to employment centers |
| `NOX`    | Nitric oxide concentration |
| `CHAS`   | Charles River adjacency (binary) |
| `RAD`    | Accessibility to radial highways |
| `PRICE`  | Median value of owner-occupied homes (in $1000s) – *Target Variable* |

---

## 🔬 Workflow

### 1. 📦 Data Loading & Inspection
- Import CSV file (`boston.csv`)  
- Inspect data shape, columns, types, and missing values  
- Perform data quality checks (NaNs, duplicates)

### 2. 📊 Exploratory Data Analysis (EDA)
- Distribution plots of price, distance, and rooms  
- Histograms and bar charts for categorical features  
- Correlation heatmap and pairplots  
- Joint plots exploring relationships between variables (e.g., `RM` vs `PRICE`, `LSTAT` vs `PRICE`)

### 3. 🧠 Model Building

#### ✅ Linear Regression
- Split data into training and testing sets  
- Train a `LinearRegression` model (`sklearn.linear_model.LinearRegression`)  
- Calculate **R² score** and analyze feature coefficients  

📈 **Example Insight:**  
Each additional room adds ≈ \$3,110 to the house price.

#### 🔄 Log-Transformation
- Log-transform the target (`PRICE`) to normalize skewness  
- Retrain the model and compare performance  
- Residuals become more normally distributed, improving predictions

---

## 📉 Model Evaluation

Metrics and plots include:

- **R² score** on training and test data  
- **Residual distribution** (original vs log-transformed)  
- **Actual vs Predicted** price scatter plots  
- **Residuals vs Predicted values** plots

✅ **Example Results:**

| Model | R² (Train) | R² (Test) |
|-------|------------|-----------|
| Original | ~0.74 | ~0.72 |
| Log-transformed | ~0.78 | ~0.76 |

---

## 🏷️ Property Valuation Demo

The notebook also includes a **custom valuation feature**. By modifying property attributes (e.g., number of rooms, proximity to river, pollution level), you can estimate the house price dynamically:

next_to_river = True
nr_rooms = 8
students_per_classroom = 20
distance_to_town = 5
pollution = data.NOX.quantile(0.75)
amount_of_poverty = data.LSTAT.quantile(0.25)

## 💡 Example Output

The property is estimated to be worth $452,893.425

---

## 📈 Results & Insights

- **Log-transformed regression** improves model accuracy and residual distribution.  
- Features like `RM` (rooms), `LSTAT` (poverty level), and `PTRATIO` (education quality) strongly influence house prices.  
- The model can provide **property valuation estimates** with interpretable coefficients.

---

## 📁 Project Structure

📂 House-Price-Prediction/
│
├── 📊 boston.csv
├── 📓 house price prediction.ipynb
└── 📄 README.md

---

## 🚀 Future Improvements

- Add **feature scaling** and **regularization** (Lasso/Ridge)  
- Implement advanced models like **Random Forest** or **XGBoost**  
- Deploy the model as a **web app** using **Streamlit** or **Flask**  
