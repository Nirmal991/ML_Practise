'''
The Challenge:
Imagine you are a cricket data analyst.
Your organization has collected performance statistics for various international cricket players (including matches played, total runs, wickets taken, and catches). 
However, sports scouts are tired of manually categorizing players.
'''

'''
Goal: Categorize cricket players based on their performance statistics and
      accurately predict their primary playing role: Batsman, Bowler, or All-Rounder.
'''

'''
Approach:
1. Clean and prepare raw data.
2. Feature Scaling
3. Train a Model : use 80% of the data for training
4. Evaluate the Model : use the remaining 20% of the data for testing
5. Predict
6. visualize
'''

raw_csv_data = """matches,runs,wickets,catches,dominant_hand,role
150,6500,4,45,Right,Batsman
120,4800,10,38,Left,Batsman
200,8500,12,60,Right,Batsman
90,3200,2,25,Right,Batsman
180,7200,8,50,Left,Batsman
110,4200,5,30,Right,Batsman
75,2100,0,18,Right,Batsman
140,5900,15,42,Right,Batsman
130,250,180,20,Right,Bowler
160,400,220,15,Right,Bowler
95,150,130,12,Left,Bowler
210,600,310,25,Right,Bowler
85,100,110,8,Right,Bowler
175,500,260,18,Left,Bowler
105,300,145,10,Right,Bowler
190,800,290,22,Right,Bowler
140,3200,120,35,Right,Allrounder
165,4100,150,40,Left,Allrounder
115,2500,95,28,Right,Allrounder
195,5100,185,55,Right,Allrounder
80,1800,70,20,Right,Allrounder
135,3100,110,32,Left,Allrounder
170,4500,140,48,Right,Allrounder
100,2200,80,22,Right,Allrounder
160,3900,135,36,Right,Allrounder
"""

# Convert the raw CSV data into a pandas DataFrame
import pandas as pd
from io import StringIO
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# step 1 : load the data series into a DataFrame
df = pd.read_csv(StringIO(raw_csv_data))

print(df.head())
print("The target feature id 'role'\n")
print(df['role'].value_counts())

# check the missing values:
print("\nMissing values in each column:")
print(df.isnull().sum())

# step 2 : Data Preprocessing (encoding and squeezing/normalization/scaling)
 # encoder for role
 
role_encoder = LabelEncoder()
df['role_encoded'] = role_encoder.fit_transform(df['role'])

#encode for dominant hand
hand_encoder = LabelEncoder()
df['hand_encoded'] = hand_encoder.fit_transform(df['dominant_hand'])

# print("Encoded mapping: ")
# for idx, label in enumerate(role_encoder.classes_):
#     print(f"{label} -> {idx}")
    
# print("Hand Encoder mapping:",list(zip(hand_encoder.classes_, range(len(hand_encoder.classes_)))))

feature_columns = ['matches', 'runs', 'wickets', 'catches', 'hand_encoded']
X = df[feature_columns]
y = df['role_encoded']

print(X[:5])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)
# print(X_scaled_df.head())

# step 3 : Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, stratify=y, random_state=42)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)
print("=" * 60)

# step 4: Train the model
rf_clf = RandomForestClassifier(n_estimators=50, random_state=42, criterion='gini')
rf_clf.fit(X_train, y_train)

# step 5 evaluate
y_pred = rf_clf.predict(X_test)

# calculate the accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {acc * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred,target_names=role_encoder.classes_ ))

# predict for new player
new_player_raw = np.array([[130,3500,105,30,1]])  # 
new_player_scaled = scaler.transform(new_player_raw)

# redict role
predicted_code = rf_clf.predict(new_player_scaled)
print(f"Predicted role for the new player: {predicted_code}") 
predicted_role = role_encoder.inverse_transform(predicted_code)
print(f"Predicted role for the new player: {predicted_role[0]}") 

# step 6 feature Importance
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Feature Importance Bar Chart
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]

axes[0].bar(range(len(feature_columns)), importances[indices], color='skyblue', edgecolor='black')
axes[0].set_xticks(range(len(feature_columns)))
axes[0].set_xticklabels([feature_columns[i] for i in indices], rotation=30)
axes[0].set_title("Random Forest - Feature Importances", fontsize=14)
axes[0].set_ylabel("Importance Score")
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Subplot 2: Plot one of the Decision Trees inside the Forest
plot_tree(
    rf_clf.estimators_[0],
    feature_names=feature_columns,
    class_names=role_encoder.classes_,
    filled=True,
    rounded=True,
    ax=axes[1]
)

axes[1].set_title("Structure of Tree #1 inside the Random Forest", fontsize=14)

plt.tight_layout()
plt.show()

