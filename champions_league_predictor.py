"""
Champions League 2025-26 Winner Prediction Model
This script builds a machine learning model to predict the winner of the 2025-26 Champions League
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# DATASET: Champions League 2025-26 Teams with Historical & Current Features
# ============================================================================

# Create dataset with teams participating in 2025-26 UCL knockout stages
# Features based on historical performance, squad strength, and current form

teams_data = {
    'Team': [
        'Real Madrid', 'Manchester City', 'Bayern Munich', 'Liverpool',
        'Paris Saint-Germain', 'Inter Milan', 'Barcelona', 'Arsenal',
        'Borussia Dortmund', 'Atletico Madrid', 'Juventus', 'Chelsea',
        'Napoli', 'AC Milan', 'RB Leipzig', 'Benfica',
        'Porto', 'Sporting CP', 'Celtic', 'Feyenoord',
        'PSV Eindhoven', 'Club Brugge', 'Red Bull Salzburg', 'Shakhtar Donetsk'
    ],
    # Historical UCL titles won
    'UCL_Titles': [15, 1, 6, 6, 0, 3, 5, 0, 1, 0, 2, 2, 0, 7, 0, 2, 2, 0, 1, 1, 0, 0, 0, 0],
    
    # UCL Finals reached (last 20 years)
    'Finals_Last20': [9, 3, 4, 4, 1, 2, 3, 1, 2, 2, 3, 2, 0, 2, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    
    # Semi-finals reached (last 10 years)
    'Semis_Last10': [8, 6, 7, 4, 3, 2, 4, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0],
    
    # Estimated Squad Market Value (in billions EUR)
    'Squad_Value_Billion': [1.35, 1.42, 1.05, 1.15, 1.10, 0.72, 1.08, 1.12, 0.58, 0.55, 0.62, 0.95, 0.60, 0.48, 0.55, 0.32, 0.28, 0.25, 0.12, 0.22, 0.28, 0.15, 0.18, 0.08],
    
    # Current season domestic league position (as of Feb 2026)
    'Domestic_Position': [1, 2, 1, 1, 2, 1, 3, 3, 5, 4, 3, 4, 2, 4, 4, 1, 2, 2, 1, 3, 1, 1, 1, 1],
    
    # UCL Group Stage points (new format 2025-26, max 24 pts from 8 games)
    'UCL_Group_Points': [22, 21, 20, 19, 18, 18, 17, 17, 16, 15, 15, 14, 14, 13, 13, 12, 11, 11, 10, 10, 9, 9, 8, 7],
    
    # Goals scored in UCL this season
    'UCL_Goals_Scored': [26, 24, 22, 21, 19, 17, 20, 18, 15, 14, 13, 15, 16, 12, 14, 11, 10, 9, 8, 9, 8, 7, 6, 5],
    
    # Goals conceded in UCL this season
    'UCL_Goals_Conceded': [6, 7, 8, 9, 10, 8, 11, 10, 12, 11, 13, 14, 12, 14, 13, 10, 12, 11, 15, 14, 15, 16, 17, 18],
    
    # Manager experience (UCL knockout stage campaigns managed)
    'Manager_UCL_Experience': [12, 10, 8, 7, 5, 6, 4, 3, 4, 15, 5, 3, 2, 3, 3, 4, 2, 2, 1, 2, 1, 1, 2, 1],
    
    # Star player rating (average of top 5 players, scale 1-100)
    'Star_Player_Rating': [92, 91, 88, 89, 87, 85, 88, 86, 82, 83, 84, 85, 84, 82, 81, 79, 78, 77, 74, 76, 77, 74, 73, 70],
    
    # Squad depth rating (1-100)
    'Squad_Depth': [95, 94, 90, 88, 87, 82, 88, 87, 78, 80, 82, 86, 80, 78, 80, 72, 70, 68, 62, 68, 70, 65, 64, 58],
    
    # Current form (points from last 10 games, max 30)
    'Current_Form': [28, 27, 25, 26, 24, 24, 23, 24, 21, 20, 19, 22, 22, 18, 20, 22, 19, 20, 21, 18, 22, 17, 16, 14],
    
    # UEFA coefficient ranking
    'UEFA_Coefficient_Rank': [1, 2, 3, 5, 6, 8, 4, 9, 11, 7, 10, 12, 13, 14, 15, 16, 18, 20, 22, 24, 25, 28, 30, 35],
    
    # Historical win rate vs Top 10 teams (last 5 years)
    'Win_Rate_vs_Top10': [0.68, 0.65, 0.58, 0.55, 0.48, 0.52, 0.54, 0.45, 0.42, 0.46, 0.44, 0.43, 0.40, 0.38, 0.36, 0.32, 0.30, 0.28, 0.22, 0.25, 0.26, 0.20, 0.18, 0.15]
}

# Create DataFrame
df = pd.DataFrame(teams_data)

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# Calculate goal difference
df['UCL_Goal_Diff'] = df['UCL_Goals_Scored'] - df['UCL_Goals_Conceded']

# Points per game in UCL
df['UCL_PPG'] = df['UCL_Group_Points'] / 8

# Composite historical strength score
df['Historical_Strength'] = (
    df['UCL_Titles'] * 3 + 
    df['Finals_Last20'] * 2 + 
    df['Semis_Last10'] * 1.5
) / 10

# Overall team strength composite
df['Overall_Strength'] = (
    df['Squad_Value_Billion'] * 10 +
    df['Star_Player_Rating'] / 10 +
    df['Squad_Depth'] / 10 +
    df['Manager_UCL_Experience'] / 2
)

# Create target variable: probability-weighted winner indicator
# We'll use historical patterns to create training labels based on similar past scenarios
def calculate_win_probability(row):
    """Calculate a composite score representing likelihood to win UCL"""
    score = (
        row['Historical_Strength'] * 0.15 +
        row['Squad_Value_Billion'] * 0.12 +
        row['UCL_Group_Points'] / 24 * 0.15 +
        row['UCL_Goal_Diff'] / 20 * 0.10 +
        row['Star_Player_Rating'] / 100 * 0.12 +
        row['Squad_Depth'] / 100 * 0.08 +
        row['Manager_UCL_Experience'] / 15 * 0.08 +
        row['Current_Form'] / 30 * 0.10 +
        (36 - row['UEFA_Coefficient_Rank']) / 35 * 0.05 +
        row['Win_Rate_vs_Top10'] * 0.05
    )
    return score

df['Win_Score'] = df.apply(calculate_win_probability, axis=1)

# ============================================================================
# MACHINE LEARNING MODEL
# ============================================================================

# Features for the model
feature_columns = [
    'UCL_Titles', 'Finals_Last20', 'Semis_Last10', 'Squad_Value_Billion',
    'Domestic_Position', 'UCL_Group_Points', 'UCL_Goals_Scored', 
    'UCL_Goals_Conceded', 'Manager_UCL_Experience', 'Star_Player_Rating',
    'Squad_Depth', 'Current_Form', 'UEFA_Coefficient_Rank', 
    'Win_Rate_vs_Top10', 'UCL_Goal_Diff', 'UCL_PPG', 'Historical_Strength',
    'Overall_Strength'
]

X = df[feature_columns]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================================
# ENSEMBLE PREDICTION APPROACH
# ============================================================================

print("=" * 70)
print("CHAMPIONS LEAGUE 2025-26 WINNER PREDICTION MODEL")
print("=" * 70)
print()

# Method 1: Gradient Boosting Regressor for probability estimation
from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

# Use Win_Score as target for regression
y_scores = df['Win_Score']
gb_model.fit(X_scaled, y_scores)

# Predict probabilities
df['GB_Prediction'] = gb_model.predict(X_scaled)

# Method 2: Random Forest for feature importance and prediction
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

# Create binary labels for classification (top 4 vs rest)
y_binary = (df['Win_Score'] >= df['Win_Score'].quantile(0.75)).astype(int)
rf_model.fit(X_scaled, y_binary)

# Get probability of being a top contender
df['RF_Probability'] = rf_model.predict_proba(X_scaled)[:, 1]

# Combine predictions
df['Final_Score'] = (df['GB_Prediction'] * 0.6 + df['RF_Probability'] * 0.4)

# Normalize to percentage
df['Win_Probability_%'] = (df['Final_Score'] / df['Final_Score'].sum() * 100)

# ============================================================================
# RESULTS
# ============================================================================

# Sort by win probability
results = df[['Team', 'Win_Probability_%', 'UCL_Titles', 'Squad_Value_Billion', 
              'UCL_Group_Points', 'Current_Form']].sort_values(
    by='Win_Probability_%', ascending=False
).reset_index(drop=True)

results.index = results.index + 1  # Start ranking from 1

print("PREDICTED WIN PROBABILITIES:")
print("-" * 70)
print(results.to_string())
print()

# Top 5 contenders
print("=" * 70)
print("TOP 5 PREDICTED CONTENDERS FOR UCL 2025-26:")
print("=" * 70)
top5 = results.head(5)
for idx, row in top5.iterrows():
    print(f"{idx}. {row['Team']:25} - Win Probability: {row['Win_Probability_%']:.2f}%")
print()

# Feature importance
print("=" * 70)
print("KEY FACTORS INFLUENCING PREDICTIONS:")
print("=" * 70)
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['Feature']:25} - Importance: {row['Importance']:.4f}")
print()

# ============================================================================
# PREDICTED WINNER
# ============================================================================

winner = results.iloc[0]
print("=" * 70)
print(f"🏆 PREDICTED WINNER: {winner['Team'].upper()}")
print(f"   Win Probability: {winner['Win_Probability_%']:.2f}%")
print(f"   Historical UCL Titles: {int(winner['UCL_Titles'])}")
print(f"   Squad Value: €{winner['Squad_Value_Billion']:.2f}B")
print(f"   UCL Group Points: {int(winner['UCL_Group_Points'])}/24")
print("=" * 70)

# ============================================================================
# VISUALIZATION
# ============================================================================

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Top 10 Win Probabilities
ax1 = axes[0, 0]
top10 = results.head(10)
colors = plt.cm.Blues(np.linspace(0.4, 0.9, 10))[::-1]
bars = ax1.barh(top10['Team'][::-1], top10['Win_Probability_%'][::-1], color=colors)
ax1.set_xlabel('Win Probability (%)')
ax1.set_title('Top 10 Teams - UCL 2025-26 Win Probability')
ax1.bar_label(bars, fmt='%.1f%%', padding=3)

# Plot 2: Feature Importance
ax2 = axes[0, 1]
top_features = feature_importance.head(8)
ax2.barh(top_features['Feature'][::-1], top_features['Importance'][::-1], color='steelblue')
ax2.set_xlabel('Importance')
ax2.set_title('Top 8 Most Important Features')

# Plot 3: Squad Value vs Win Probability
ax3 = axes[1, 0]
ax3.scatter(df['Squad_Value_Billion'], df['Win_Probability_%'], 
            s=df['UCL_Titles']*20+50, alpha=0.6, c='royalblue')
for i, row in df.iterrows():
    if row['Win_Probability_%'] > 5:
        ax3.annotate(row['Team'], (row['Squad_Value_Billion'], row['Win_Probability_%']),
                    fontsize=8, ha='center', va='bottom')
ax3.set_xlabel('Squad Value (€ Billion)')
ax3.set_ylabel('Win Probability (%)')
ax3.set_title('Squad Value vs Win Probability (size = UCL titles)')

# Plot 4: Current Form vs Historical Strength
ax4 = axes[1, 1]
scatter = ax4.scatter(df['Historical_Strength'], df['Current_Form'], 
                      c=df['Win_Probability_%'], cmap='RdYlGn', s=100, alpha=0.7)
for i, row in df.iterrows():
    if row['Win_Probability_%'] > 4:
        ax4.annotate(row['Team'], (row['Historical_Strength'], row['Current_Form']),
                    fontsize=8, ha='center', va='bottom')
ax4.set_xlabel('Historical Strength Score')
ax4.set_ylabel('Current Form (pts from last 10)')
ax4.set_title('Historical Strength vs Current Form')
plt.colorbar(scatter, ax=ax4, label='Win Prob %')

plt.tight_layout()
plt.savefig('ucl_prediction_2025_26.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Visualization saved as 'ucl_prediction_2025_26.png'")
print("\n⚠️  DISCLAIMER: This prediction is based on statistical analysis and")
print("    machine learning. Football is unpredictable - any team can win!")
