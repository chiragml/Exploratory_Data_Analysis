# Exploratory_Data_Analysis
The Project gives a complete lifecycle of a project that includes: 
1. Data Analysis
2. Feature Engineering
3. Feature Selection
4. Model Building
5. Model Deployment

Each of the above components are described briefly by the phases below.
The project performs the EDA steps on the house pricing data.  
## Phase 1: Data Analysis
Data analysis is the initial phase of any data science project, aimed at understanding and exploring the dataset. The key steps include:

### 1.1 Understand the Data
In this step, you will gain a comprehensive understanding of the dataset, including its structure, size, and the features it contains.

#### Load the dataset
dataset = pd.read_csv("house_price_train.csv")

#### Display the first few rows of the dataset
print(dataset.head())

#### Check the dimensions of the dataset
print(dataset.shape)
### 1.2 Missing Values
Identify and handle missing values in the dataset. This step is crucial for ensuring data quality and reliability.

#### Find and display features with missing values
features_with_na = [feature for feature in dataset.columns if dataset[feature].isnull().sum() > 1]
for feature in features_with_na:
    print(feature, np.round(dataset[feature].isnull().mean(), 4), '% missing value')
### 1.3 Numerical Variables
Explore numerical variables in the dataset and analyze their distributions.


#### Find numerical variables and their count
numerical_features = [feature for feature in dataset.columns if dataset[feature].dtypes != 'O']
print('Number of numerical variables: ', len(numerical_features))
### 1.4 Temporal Variables
Identify and analyze temporal variables, such as year-related features, and their impact on the target variable.


#### Analyze temporal variables and their relationship with SalePrice
year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]
for feature in year_feature:
    print(feature, dataset[feature].unique())
### 1.5 Discrete Variables
Identify and analyze discrete numerical variables.

#### Find and analyze discrete variables
discrete_feature = [feature for feature in numerical_features if len(dataset[feature].unique()) < 25 and feature not in year_feature + ['Id']]
print("Discrete Variables Count: ", len(discrete_feature))
### 1.6 Continuous Variables
Identify and analyze continuous numerical variables.

#### Find and analyze continuous variables
continuous_feature = [feature for feature in numerical_features if feature not in discrete_feature + year_feature + ['Id']]
print("Total number of continuous variables: ", len(continuous_feature))
### 1.7 Exploratory Data Analysis (EDA) Part 2
Conduct further exploratory data analysis, including logarithmic transformations, to understand the relationships between features and the target variable.

### 1.8 Outliers
Identify and visualize outliers in continuous variables.

## Phase 2: Feature Engineering
Feature engineering is the process of creating new features or modifying existing ones to improve the model's performance.

## Phase 3: Feature Selection
Select the most relevant features that have the most impact on the target variable and remove unnecessary features.

## Phase 4: Model Building
Build predictive models using machine learning techniques to predict the target variable (SalePrice in this case).
