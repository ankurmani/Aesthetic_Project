import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess CSV
df = pd.read_csv("composition_one_hot_encoded.csv")

regression_cols = ["ev_in", "white_balance_a", "white_balance_b", "zoom_suggestion", "dx", "dy"]
regression_gt_cols = ["ev_gt", "wb_a_gt", "wb_b_gt", "zoom_gt", "dx_gt", "dy_gt"]
composition_cols = [f"class_{i}" for i in range(39)]

# Standardize regression columns
scaler = StandardScaler()
df[regression_cols] = scaler.fit_transform(df[regression_cols])
df[regression_gt_cols] = scaler.fit_transform(df[regression_gt_cols])

# Save scaler stats for inference
mean_std = pd.DataFrame({"mean": scaler.mean_, "std": scaler.scale_}, index=regression_cols)
mean_std.to_csv("standardization_stats.csv")

# Split into train and test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Common image preprocessing
def load_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    return tf.keras.utils.img_to_array(img) / 255.0

def preprocess(row):
    image = load_image("./images/" + str(row["image_name"]))
    depthmap = tf.image.rgb_to_grayscale(image)
    regression_target = row[regression_gt_cols].values.astype(np.float32)
    composition = row[composition_cols].values.astype(np.float32)
    return image, depthmap, regression_target, composition

# Generator factories for train/test
def generator(df_subset):
    for _, row in df_subset.iterrows():
        yield preprocess(row)

def create_dataset(split='train'):
    selected_df = train_df if split == 'train' else test_df
    output_signature = (
        tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(224, 224, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(6,), dtype=tf.float32),
        tf.TensorSpec(shape=(39,), dtype=tf.float32)
    )
    return tf.data.Dataset.from_generator(
        lambda: generator(selected_df),
        output_signature=output_signature
    )
