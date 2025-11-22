import os
import pandas as pd
import tensorflow as tf
from model import FrameCorrectionModel
from util import load_image_and_depth, apply_transformations_tf
import PIL.Image

# === CONFIGURATION ===
csv_path = "composition_one_hot_encoded.csv"
image_dir = "images"  # Update if needed
output_csv = "inference_results.csv"
output_base = "outputs"
os.makedirs(output_base, exist_ok=True)

# === Load Data ===
df = pd.read_csv(csv_path)
image_paths = df["image_name"].tolist()

# === Load Model ===
model = FrameCorrectionModel()
dummy_rgb = tf.random.normal((1, 224, 224, 3))
dummy_depth = tf.random.normal((1, 224, 224, 1))
_ = model([dummy_rgb, dummy_depth])  # Build model
model.load_weights("checkpoints/model_checkpoint_epoch_5.weights.h5")  # Change path if needed
epoch = 5
# === Inference ===
results = []

for img_name in image_paths:
    img_path = os.path.join(image_dir, img_name)
    if not os.path.exists(img_path):
        print(f"[WARN] Skipping {img_path} (not found)")
        continue
    
    print(f"\n=== Epoch {epoch} ===\n")
    epoch_dir = os.path.join(output_base, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    rgb, depth = load_image_and_depth(img_path)
    pred_reg, pred_class = model([rgb, depth], training=False)
    transformed = apply_transformations_tf(rgb, pred_reg)

    output_path = os.path.join(epoch_dir, f"{img_name}")
    original_img = PIL.Image.open(img_path)
    original_size = original_img.size  # (width, height)
    resized_output = tf.image.resize(transformed[0], size=(original_size[1], original_size[0]))
    tf.keras.utils.save_img(output_path, resized_output)
    pred_class_label = tf.argmax(pred_class, axis=-1).numpy()[0]  # index of predicted class

    results.append({
        "image": img_name,
        "delta_ev": float(pred_reg[0, 0]),
        "wb_r": float(pred_reg[0, 1]),
        "wb_b": float(pred_reg[0, 2]),
        "zoom": float(pred_reg[0, 3]),
        "dx": float(pred_reg[0, 4]),
        "dy": float(pred_reg[0, 5]),
        "predicted_composition_class": int(pred_class_label)
    })

# === Save Results ===
pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"[INFO] Inference complete. Results saved to: {output_csv}")
