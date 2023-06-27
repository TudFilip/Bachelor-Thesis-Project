import tensorflow as tf
import datetime
import Utility.custom_model_utility as CustomModelUtils
import tqdm
from tqdm import tqdm

from timeit import default_timer as timer

EPOCHS = CustomModelUtils.EPOCHS
BATCH_SIZE = CustomModelUtils.BATCH_SIZE

TRAIN_DATASET = CustomModelUtils.create_train_dataset()
VAL_DATASET = CustomModelUtils.create_val_dataset()


print("Creating My Custom Model...")
model = CustomModelUtils.create_model()


current_date = datetime.datetime.now().strftime("%Y-%m-%d")
current_time = datetime.datetime.now().strftime("%H-%M")


print("Training Model With Custom Training Loop...")
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
train_size = len(TRAIN_DATASET)
max_val_acc = 0.0
max_train_acc = 0.0

start = timer()
for epoch in range(EPOCHS):
    print(f"\nStart of epoch {epoch}")
    for step, (x_batch_train, y_batch_train) in tqdm(enumerate(TRAIN_DATASET)):
        with tf.GradientTape() as tape:
            y_pred = model(x_batch_train, training=True)
            loss_value = loss_object(y_batch_train, y_pred)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        acc_metric.update_state(y_batch_train, y_pred)

    train_acc = acc_metric.result()
    print(f"Training acc: {float(train_acc)}")
    acc_metric.reset_states()

    if train_acc > max_train_acc:
        model.save_weights('../../Tmp/CustomModel/custom_model_latest_10movies.hdf5')
        model.save_weights(f'../../Tmp/CustomModel/custom_model_10movies_{current_date}_{current_time}_trainSize_{train_size}.hdf5')
        max_train_acc = train_acc

    for step, (x_batch_val, y_batch_val) in tqdm(enumerate(VAL_DATASET)):
        val_pred = model(x_batch_val, training=False)
        acc_metric.update_state(y_batch_val, val_pred)
    val_acc = acc_metric.result()
    print(f"Validation acc: {float(val_acc)}")
    acc_metric.reset_states()

    if val_acc > max_val_acc:
        max_val_acc = val_acc

end = timer()
model.save('../../Tmp/CustomModel/custom_model_10movies_tf', save_format='tf', overwrite=True)

CustomModelUtils.save_training(
    end - start,
    current_date,
    current_time,
    max_train_acc,
    "unknown",
    max_val_acc,
    "unknown"
)
