
import tensorflow as tf

# Strategy for multi-GPU
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    from model import FrameCorrectionModel
    from util import apply_transformations_tf, dummy_nima_score_model, supervised_loss_fn
    from dataset_pipeline import create_dataset
    import pandas as pd

    # Setup
    score_model = dummy_nima_score_model()
    model = FrameCorrectionModel()
    optimizer = tf.keras.optimizers.Adam(1e-4)
    weights = [1.0, 1.0, 1.0, 2.0, 0.5, 0.5]  # EV, WB_r, WB_b, Zoom, dx, dy

    # Dummy build to initialize
    dummy_rgb = tf.random.normal((1, 224, 224, 3))
    dummy_depth = tf.random.normal((1, 224, 224, 1))
    _ = model([dummy_rgb, dummy_depth])

    # Dataset
    batch_size = 6  # Use higher batch size for multi-GPU
    epochs = 5
    dataset = create_dataset().batch(batch_size).repeat()

    csv_path = "composition_one_hot_encoded.csv"
    num_samples = len(pd.read_csv(csv_path))
    steps_per_epoch = num_samples // batch_size

    results = []

    for epoch in range(epochs):
        for step, batch in enumerate(dataset):
            if step >= steps_per_epoch:
                break

            rgb_img, depth_map, y_reg, y_class = batch

            with tf.GradientTape() as tape:
                preds, classify = model([rgb_img, depth_map], training=True)
                transformed = apply_transformations_tf(rgb_img, preds)
                sup_loss = supervised_loss_fn(preds, y_reg, weights)
                aesth_loss = score_model(transformed)
                comp_loss = tf.keras.losses.categorical_crossentropy(y_class, classify)
                total_loss = 0.4 * sup_loss + 0.3 * (1 - aesth_loss) + 0.3 * comp_loss

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            results.append({
                "epoch": epoch,
                "step": step,
                "loss": float(tf.reduce_mean(total_loss)),
                "delta_ev": float(preds[0, 0]),
                "wb_r": float(preds[0, 1]),
                "wb_b": float(preds[0, 2]),
                "zoom": float(preds[0, 3]),
                "dx": float(preds[0, 4]),
                "dy": float(preds[0, 5])
            })

            print(f"Epoch {epoch}, Step {step}, Loss: {float(tf.reduce_mean(total_loss)):.4f}")

        # Save model after each epoch
        model.save_weights(f"checkpoints/model_checkpoint_epoch_{epoch+1}.weights.h5")
        print(f" Saved weights after epoch {epoch+1}")

    # Save results
    pd.DataFrame(results).to_csv("frame_correction_results.csv", index=False)
