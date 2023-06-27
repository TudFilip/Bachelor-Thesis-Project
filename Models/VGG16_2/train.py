import tensorflow as tf
import datetime
import Utility.vgg16_2_utility as VGG16Utils

from timeit import default_timer as timer

EPOCHS = VGG16Utils.EPOCHS
BATCH_SIZE = VGG16Utils.BATCH_SIZE

TRAIN_DATASET = VGG16Utils.create_train_dataset()
VAL_DATASET = VGG16Utils.create_val_dataset()


print("Creating VGG16 Model 2...")
model = VGG16Utils.create_model()


current_date = datetime.datetime.now().strftime("%Y-%m-%d")
current_time = datetime.datetime.now().strftime("%H-%M")


print("Creating Checkpoint...")
latest_checkpoint = VGG16Utils.create_checkpoint_latest()
checkpoint = VGG16Utils.create_checkpoint(current_date, current_time)


print("Training Model...")
steps_per_epoch = tf.data.experimental.cardinality(TRAIN_DATASET).numpy() // BATCH_SIZE
start = timer()
history = model.fit(
    TRAIN_DATASET,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    verbose=1,
    callbacks=[checkpoint, latest_checkpoint],
)
end = timer()
model.save('../../Tmp/VGG16_2/vgg16_2_10movies_tf', save_format='tf')

# val_loss = 0
# val_acc = 0
# validation_steps = tf.data.experimental.cardinality(VAL_DATASET).numpy() // BATCH_SIZE
# with tf.device('/GPU:0'):
#     val_loss, val_acc = model.evaluate(VAL_DATASET, steps=validation_steps, verbose=1)

VGG16Utils.save_training(
    end - start,
    current_date,
    current_time,
    history.history['accuracy'],
    history.history['loss'],
    "unknown",
    "unknown"
)
