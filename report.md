For a detailed analysis with visualizations,check out [Report.md](Report.md)

# Spotify Song Performance Report

## 🎯 Objective
Provide a clear analysis of Spotify songs to identify patterns, predict trending songs, and recommend strategies based on audio features, release timing, and seasonal trends.

## 📊 Dataset Overview
- Source: Spotify API
- Sample size: 50,000 songs
- Features: BPM, danceability, energy, valence, acousticness, release date, etc.

## 🔍 Analysis Steps
1. **Data Cleaning & Preparation**
   - Removed duplicates and missing values
   - Standardized formats for all features

2. **Exploratory Analysis**
   - Correlation analysis of features vs popularity
   - Identified high-impact variables

3. **Predictive Modeling**
   - Linear Regression for trend estimation
   - Random Forest for non-linear patterns
   - Evaluated models using R², MAE, and RMSE

## 📈 Key Insights
- Danceability strongly correlates with trending songs (R² = 0.72)
- Energy and valence moderately impact popularity
- Songs released in summer are 15% more likely to trend
- Certain chord progressions show higher engagement

## 📊 Visualizations
- **Danceability vs Popularity:** ![Danceability](results/danceability_popularity.png)  
- **Energy vs Popularity:** ![Energy](results/energy_popularity.png)  
- **Seasonal Trends:** ![Seasonal Trends](results/seasonal_trends.png)  

## 💡 Recommendations
- Prioritize highly danceable songs during summer for playlist promotion
- Focus marketing on high-energy and high-valence tracks
- Use predictive models to optimize release strategy

## 🧑‍💻 Access Full Analysis
- Interactive Jupyter Notebook: [Spotify Analysis Notebook](https://colab.research.google.com/github/USERNAME/REPO/blob/main/notebooks/spotify_analysis.ipynb)

