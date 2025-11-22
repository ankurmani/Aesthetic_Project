import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load your CSV
df = pd.read_csv("aesthetic_dataset.csv")  # Make sure the path is correct

# Encode labels
label_encoder = LabelEncoder()
df['composition_encoded'] = label_encoder.fit_transform(df['composition_rule'])

# One-hot encode
num_classes = 39
one_hot = np.eye(num_classes)[df['composition_encoded']]
one_hot_df = pd.DataFrame(one_hot, columns=[f"class_{i}" for i in range(num_classes)])

# Merge and export
final_df = pd.concat([df, one_hot_df], axis=1)
final_df.to_csv("composition_one_hot_encoded.csv", index=False)

print("✅ Saved to composition_one_hot_encoded.csv")
