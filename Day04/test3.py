import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. 10 rows of real-world style data: Height and Weight
df = pd.DataFrame(
    {
        'Height_cm': [150, 160, 165, 170, 175, 180, 185, 168, 172, 178],
        'Weight_kg': [50, 60, 63, 70, 75, 82, 88, 66, 71, 79],
    }
)

# 2. Standardize the data so height and weight are on the same scale
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# 3. Apply PCA to squash 2 columns down to just 1 column!
pca = PCA(n_components=1)
pca_array = pca.fit_transform(scaled_data)

# 4. Show the new single column
df_result = pd.DataFrame(pca_array, columns=['Body_Size_Score'])
print(df_result)