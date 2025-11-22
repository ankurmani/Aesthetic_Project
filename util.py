import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet
import pandas as pd
from sklearn.preprocessing import LabelEncoder

gt_df = pd.read_csv("composition_one_hot_encoded.csv")

def get_ground_truth_vector(img_filename):
    """
    Fetch ground truth correction vector for a given image filename.
    Returns:
        A numpy array: ["ev_in", "white_balance_a", "white_balance_b", "zoom_suggestion", "dx", "dy"]
    """
    row = gt_df[gt_df["image_name"] == img_filename]
    if row.empty:
        raise ValueError(f"Ground truth not found for image: {img_filename}")
    
    return row[["ev_in", "white_balance_a", "white_balance_b", "zoom_suggestion", "dx", "dy"]].values[0]

def load_image_and_depth(path, target_size=(224, 224)):
    img = tf.keras.preprocessing.image.load_img(path, target_size=target_size)
    img_arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    rgb_tensor = tf.convert_to_tensor(img_arr, dtype=tf.float32)
    depth_map = tf.image.rgb_to_grayscale(rgb_tensor)
    return tf.expand_dims(rgb_tensor, 0), tf.expand_dims(depth_map, 0)

def apply_transformations(image, preds):
    """
    Differentiable image transformation used ONLY during training.
    Uses TF ops only — no NumPy or OpenCV.
    """
    delta_ev = preds[:, 0:1]       # (1, 1)
    delta_wb_r = preds[:, 1:2]
    delta_wb_b = preds[:, 2:3]

    # -------- Exposure (EV) correction ----------
    ev_factor = tf.pow(2.0, delta_ev)
    image = image * ev_factor

    # -------- White balance correction ----------
    wb_gains = tf.concat([
        delta_wb_b + 1.0,  # Blue channel
        tf.ones_like(delta_wb_r),  # Green stays unchanged
        delta_wb_r + 1.0   # Red channel
    ], axis=-1)  # Shape: (1, 3)

    image = image * wb_gains[:, tf.newaxis, tf.newaxis, :]  # Broadcast gains
    image = tf.clip_by_value(image, 0.0, 1.0)

    # -------- Zoom ----------
    zoom = preds[:, 3:4]  # Expected to be in range [-1, +1]
    scale = 1.0 + zoom * 0.2  # Zoom in by up to 20%
    orig_h = tf.shape(image)[1]
    orig_w = tf.shape(image)[2]

    new_h = tf.cast(tf.cast(orig_h, tf.float32) / scale, tf.int32)
    new_w = tf.cast(tf.cast(orig_w, tf.float32) / scale, tf.int32)
    size = tf.concat([new_h, new_w], axis=-1)
    size = tf.squeeze(size, axis=0)  # Make it shape (2,)
    resized = tf.image.resize(image, size=size, method='bilinear')

    # -------- Center crop or pad to restore original size ----------
    restored = tf.image.resize_with_crop_or_pad(resized, orig_h, orig_w)

    return restored

def apply_transformations_tf(image, preds):
    """
    Differentiable image transformation for batch processing.
    Args:
        image: Tensor of shape (B, H, W, 3)
        preds: Tensor of shape (B, 6) — delta_ev, wb_r, wb_b, zoom, dx, dy
    Returns:
        Transformed image: Tensor of shape (B, H, W, 3)
    """
    B = tf.shape(image)[0]
    H = tf.shape(image)[1]
    W = tf.shape(image)[2]

    # --- Exposure correction ---
    delta_ev = preds[:, 0:1]  # shape (B, 1)
    ev_factor = tf.pow(2.0, delta_ev)
    image = image * ev_factor[:, tf.newaxis, tf.newaxis, :]  # broadcasted

    # --- White balance correction ---
    delta_wb_r = preds[:, 1:2]
    delta_wb_b = preds[:, 2:3]
    wb_gains = tf.stack([
        delta_wb_b[:, 0] + 1.0,         # B
        tf.ones_like(delta_wb_r[:, 0]), # G
        delta_wb_r[:, 0] + 1.0          # R
    ], axis=-1)  # shape (B, 3)

    wb_gains = tf.reshape(wb_gains, (B, 1, 1, 3))
    image = image * wb_gains
    image = tf.clip_by_value(image, 0.0, 1.0)

    # --- Zoom transform ---
    zoom = preds[:, 3:4]
    scale = 1.0 + zoom * 0.2  # shape (B, 1)
    new_heights = tf.cast(tf.cast(H, tf.float32) / scale, tf.int32)
    new_widths = tf.cast(tf.cast(W, tf.float32) / scale, tf.int32)

    # Process each image in batch
    transformed_images = []
    for i in range(B):
        resized = tf.image.resize(image[i], size=(new_heights[i][0], new_widths[i][0]))
        restored = tf.image.resize_with_crop_or_pad(resized, H, W)
        transformed_images.append(restored)

    return tf.stack(transformed_images, axis=0)  # shape (B, H, W, 3)


def dummy_nima_score_model():
    # Dummy trainable regressor
    return Sequential([
        GlobalAveragePooling2D(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Scores between 0–1
    ])

def supervised_loss_fn(preds, labels, weights=None):
    """
    Compute supervised regression loss between predicted and ground truth corrections.
    
    Args:
        preds: Tensor of shape (B, 6), predicted deltas
        labels: Tensor of shape (B, 6), ground truth deltas
        weights: Optional list or tensor of shape (6,) with per-dimension weights

    Returns:
        Scalar loss value
    """
    if weights is None:
        weights = tf.constant([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=tf.float32)
    else:
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
    
    loss_per_dim = tf.square(preds - labels) * weights
    loss = tf.reduce_mean(tf.reduce_sum(loss_per_dim, axis=-1))  # Mean over batch
    return loss

import imquality.brisque as brisque

def brisque_score_np(image_np):
    """Computes BRISQUE score"""
    score = brisque.score(image_np)
    return score

@tf.function
def brisque_loss_tf(image_batch):
    """BRISQUE loss compatible with TF graph execution"""
    image_batch = tf.clip_by_value(image_batch * 255.0, 0, 255)
    image_batch_uint8 = tf.cast(image_batch, tf.uint8)

    def _compute_brisque(img):
        return np.array([brisque_score_np(img)], dtype=np.float32)

    scores = tf.map_fn(
        lambda img: tf.numpy_function(_compute_brisque, [img], tf.float32),
        image_batch_uint8,
        fn_output_signature=tf.TensorSpec(shape=(1,), dtype=tf.float32)
    )
    return tf.reduce_mean(scores)

def build_nima_mobilenet_model(weights_path='weights/mobilenet_weights.h5'):
    # Define base MobileNet model
    base_model = MobileNet(input_shape=(None, None, 3), alpha=1.0, include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    output = Dense(10, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.load_weights(weights_path)

    return model


def earth_mover_distance(y_pred):
    """Computes aesthetic quality from NIMA probability output (higher = better)"""
    probs = tf.nn.softmax(y_pred, axis=-1)
    scores = tf.range(1, 11, dtype=tf.float32)
    return tf.reduce_sum(probs * scores, axis=-1) / 10.0