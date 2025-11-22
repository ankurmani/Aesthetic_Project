import tensorflow as tf

class FrameCorrectionModel(tf.keras.Model):
    def __init__(self):
        super(FrameCorrectionModel, self).__init__()
        base = tf.keras.applications.ResNet50(include_top=False, input_shape=(224, 224, 3), pooling='avg')
        self.backbone = base
        self.depth_proj = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")
        self.flatten = tf.keras.layers.GlobalAveragePooling2D()
        self.concat = tf.keras.layers.Concatenate()
        self.dense = tf.keras.layers.Dense(128, activation="relu")
        self.out = tf.keras.layers.Dense(6, activation="linear")  # delta_ev, wb, zoom, dx, dy, comp
        self.classify = tf.keras.layers.Dense(39, activation='softmax')

    def call(self, inputs):
        rgb, depth = inputs
        x1 = self.backbone(rgb)
        x2 = self.flatten(self.depth_proj(depth))
        x = self.concat([x1, x2])
        x = self.dense(x)
        return self.out(x), self.classify(x)