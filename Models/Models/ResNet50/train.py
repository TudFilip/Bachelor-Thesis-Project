import datetime
import Utility.resnet50_utility as ResNet50Utils

from timeit import default_timer as timer

EPOCHS = ResNet50Utils.EPOCHS
BATCH_SIZE = ResNet50Utils.BATCH_SIZE

TRAIN_DATASET = ResNet50Utils.create_train_dataset()
VAL_DATASET = ResNet50Utils.create_val_dataset()


print("Creating ResNet50 Model...")
model = ResNet50Utils.create_model()


current_date = datetime.datetime.now().strftime("%Y-%m-%d")
current_time = datetime.datetime.now().strftime("%H-%M")


print("Creating Checkpoint...")
latest_checkpoint = ResNet50Utils.create_checkpoint_latest()
checkpoint = ResNet50Utils.create_checkpoint(current_date, current_time)


print("Training Model...")
# steps_per_epoch = tf.data.experimental.cardinality(TRAIN_DATASET).numpy() // BATCH_SIZE
start = timer()
history = model.fit(
    TRAIN_DATASET,
    epochs=EPOCHS,
    # steps_per_epoch=steps_per_epoch,
    callbacks=[checkpoint, latest_checkpoint],
    verbose=1
)
end = timer()
model.save('../../Tmp/ResNet50/resnet50_10movies_tf', save_format='tf', overwrite=True)

ResNet50Utils.save_training(
    end - start,
    current_date,
    current_time,
    history.history['accuracy'],
    history.history['loss'],
    'unknown',
    'unknown'
)
