import tensorflow as tf
import datetime
import Utility.vgg16_utility as VGG16Utils

from timeit import default_timer as timer

EPOCHS = VGG16Utils.EPOCHS
BATCH_SIZE = VGG16Utils.BATCH_SIZE

TRAIN_DATASET = VGG16Utils.create_train_dataset()
VAL_DATASET = VGG16Utils.create_val_dataset()


print("Creating VGG16 Model 1...")
model = VGG16Utils.create_model()


current_date = datetime.datetime.now().strftime("%Y-%m-%d")
current_time = datetime.datetime.now().strftime("%H-%M")


print("Creating Checkpoint...")
latest_checkpoint = VGG16Utils.create_checkpoint_latest()
checkpoint = VGG16Utils.create_checkpoint(current_date, current_time)


print("Training Model...")
start = timer()
history = model.fit(
    TRAIN_DATASET,
    epochs=EPOCHS,
    callbacks=[checkpoint, latest_checkpoint],
    verbose=1
)
end = timer()
model.save('../../Tmp/VGG16_1/vgg16_1_10movies_tf', save_format='tf')

val_loss = 0
val_acc = 0
with tf.device('/GPU:0'):
    val_loss, val_acc = model.evaluate(VAL_DATASET)

VGG16Utils.save_training(
    end - start,
    current_date,
    current_time,
    history.history['accuracy'],
    history.history['loss'],
    val_acc,
    val_loss
)
