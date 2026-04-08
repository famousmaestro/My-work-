import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train = y_train.squeeze().astype("int32")
y_test = y_test.squeeze().astype("int32")

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Train:", x_train.shape, y_train.shape)
print("Test:", x_test.shape, y_test.shape)

IMG_SIZE = 96

def resize_images(x):
    return tf.image.resize(x, (IMG_SIZE, IMG_SIZE))

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1),
])

def build_baseline_cnn(input_shape=(32, 32, 3), lr=1e-3):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model

def build_transfer_model(backbone_name="MobileNetV2", lr=1e-3, fine_tune=False, unfreeze_last=20):
    inputs = layers.Input(shape=(32, 32, 3))
    x = data_augmentation(inputs)
    x = layers.Lambda(resize_images)(x)

    if backbone_name == "MobileNetV2":
        backbone = keras.applications.MobileNetV2(
            include_top=False, weights="imagenet",
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        preprocess = keras.applications.mobilenet_v2.preprocess_input
    elif backbone_name == "ResNet50":
        backbone = keras.applications.ResNet50(
            include_top=False, weights="imagenet",
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        preprocess = keras.applications.resnet.preprocess_input
    else:
        backbone = keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet",
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        preprocess = keras.applications.efficientnet.preprocess_input

    x = layers.Lambda(preprocess)(x)
    backbone.trainable = False
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    if fine_tune:
        backbone.trainable = True
        for layer in backbone.layers[:-unfreeze_last]:
            layer.trainable = False
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    return model

def train_and_eval(model, x_train, y_train, x_test, y_test, epochs=10, batch_size=64):
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]
    t0 = time.time()
    history = model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2
    )
    t1 = time.time()
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss={loss:.4f}, Test acc={acc*100:.2f}% | time={t1-t0:.1f}s")
    return history, (loss, acc)

baseline = build_baseline_cnn()
baseline.summary()
_ = train_and_eval(baseline, x_train, y_train, x_test, y_test, epochs=10, batch_size=64)

tl_model = build_transfer_model("MobileNetV2", lr=1e-3, fine_tune=False)
tl_model.summary()
_ = train_and_eval(tl_model, x_train, y_train, x_test, y_test, epochs=10, batch_size=64)

ft_model = build_transfer_model("MobileNetV2", lr=1e-3, fine_tune=True, unfreeze_last=20)
ft_model.summary()
_ = train_and_eval(ft_model, x_train, y_train, x_test, y_test, epochs=10, batch_size=64)

cmp_model = build_transfer_model("ResNet50", lr=1e-3, fine_tune=False)
cmp_model.summary()
_ = train_and_eval(cmp_model, x_train, y_train, x_test, y_test, epochs=10, batch_size=64)
