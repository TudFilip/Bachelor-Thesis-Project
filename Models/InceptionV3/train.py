import tensorflow as tf
import datetime
import Utility.inceptionv3_utility as InceptionV3Utils

from timeit import default_timer as timer

EPOCHS = InceptionV3Utils.EPOCHS
BATCH_SIZE = InceptionV3Utils.BATCH_SIZE

TRAIN_DATASET = InceptionV3Utils.create_train_dataset()
VAL_DATASET = InceptionV3Utils.create_val_dataset()


print("Creating InceptionV3 Model...")
model = InceptionV3Utils.create_model()


current_date = datetime.datetime.now().strftime("%Y-%m-%d")
current_time = datetime.datetime.now().strftime("%H-%M")


print("Creating Checkpoint...")
latest_checkpoint = InceptionV3Utils.create_checkpoint_latest()
checkpoint = InceptionV3Utils.create_checkpoint(current_date, current_time)


print("Training Model...")
steps_per_epoch = tf.data.experimental.cardinality(TRAIN_DATASET).numpy() // BATCH_SIZE
validation_steps = tf.data.experimental.cardinality(VAL_DATASET).numpy() // BATCH_SIZE
start = timer()
history = model.fit(
    TRAIN_DATASET,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    callbacks=[checkpoint, latest_checkpoint],
    verbose=1,
)
end = timer()
model.save('../../Tmp/InceptionV3/inceptionv3_10movies_tf', save_format='tf', overwrite=True)


InceptionV3Utils.save_training(
    end - start,
    current_date,
    current_time,
    history.history['accuracy'],
    history.history['loss'],
    'unknown',
    'unknown',
)
