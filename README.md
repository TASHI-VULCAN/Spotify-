# Spotify song popularity analyser (Predictive model)
This model used Machine learning to find various correlations between songs' popularity related to things which may not be directly related to the song or music. It can be something completely unrelated, such as; 
- Weather 
- Types of devices the song was played on 
- Social media platform use case
- Songs that can increase sales in shopping malls and stores
- What songs should the store play based on the weather to increase sales
- Songs restaurants should play to increase sales
- Dynamic song choices based on factors, like weather, time of day
- Choosing  songs based on bpm, chord progression, modes, and  feel
- Create a custom playlist based on these dynamic song choices to drive up sales
- Dynamic playlist which changes according to sales of the store, weather, time of the day and some other factors

## Tasks
    - Data exporation and cleaning
    - Feature Engineering
    - Predective modelling 
    - Clustering  
    - Create API for extracting data
    - Find good data sources  

# Recomendation settings
- **Content based filtering**: Build a recommendation system that suggests songs based on a user’s preferences for specific attributes like genre, tempo, or energy.
- **Colloborative filtering**: If the dataset includes user data (e.g., user IDs and song IDs), implement a collaborative filtering algorithm to recommend songs to users.

## EDA (Exploratory Data Analysis)
    - Step 1: 
        - Load and preview data
        - Load the dataset and examine its structure
        - Check basic information
    - Step 2:
        - Check for missing value
        - Identify and handle missing value
    - Step 3:
        - Analyse data distribution
        - Visualise numerical and categorical data distribution
    - Step 4:
        - Correlation analysis
        - Analyse relationships between numerical features
        - Look for features that have strong positive or negative correlations with the target variables (e.g., popularity)
    - Step 5: 
        - Features expoloration
        - Identify outliers uising boxplots
        - Analyse trends
    - Step 6: 
        - Data preparation for ML
        - Convert categorical features (eg. genre) to numerical using one-hot encoding or label encoding
        - Scale numerical features for ML models

## Next step: Creating ML models
- **Regression Task**: Predict popularity using features like streams, duration_ms, and tempo.
- **Classification Task**: Classify songs as "hit" (popularity > 80) or "non-hit" (popularity ≤ 80).
- **Clustering**: Group songs into clusters based on features like danceability, energy, and tempo.

## Methodology (Machine learning Model)
- Step 1: Define the problem
- Step 2: Prepare the dataset
- Step 3: Train the linear Regression model
- Step 4: Make Prediction
- Step 5: Evaluate the model
- Step 6: Visualise Results

## Expanding on the points
1. Define the problem
    -  We aim to predict the popularity of songs using various features such as, streams,duration_ms, tempo, etc.
    - We aim to find the key factors as to why a song is popular
2. Prepare the dataset
    - Select Features and Target Variable
    - Split the Dataset: Divide the dataset into training and testing sets
3. Train the linear Regression model
    - Import the Model
    - Initialize and Fit the Model
4. Make Prediction
    - Use the trained model to predict the popularity of songs in the test set
5. Evaluate the model
    -  Assess how well your regression model performs using evaluation metrics
    -  R² Score
    - Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
6. Visualise Results
    - Compare Predictions vs. Actual Values
    - Residual Plot: Visualize errors in predictions


### Next step if the performance isn't satisfactory:
- Feature engineering: add or transform features (eg. log-transform streams or normalise features)
- Regularisation: use model like ridge or lasso regression to reduce overfitting. 
- Experiment: Try other regression models like 
    - decision trees
    - random forests
    - XGBoost 

## Installation
**Libraries and packages used**<br>
1. Pandas
2. Numpy
3. matplatlib
    - .pyplot
4. seaborn
5. scikit-learn
    - .preprocessing/StandardScaler
    - .model_selection/train_test_split
    - .linear_model/LinearRegression
    - .metrics/r2_score
    - .metrics/mean_squared_error
6. Kagglehub


To active the venv you need to first bring down the defence of windows using the command below:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
. .\.venvSong\Scripts\Activate.ps1


download ipykernel alwasy for jupyter notebook