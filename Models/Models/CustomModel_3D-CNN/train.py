import os
import datetime
import Utility.custom_model_3d_cnn_utility as CustomModel3DCNNUtils

from timeit import default_timer as timer

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


EPOCHS = CustomModel3DCNNUtils.EPOCHS
BATCH_SIZE = CustomModel3DCNNUtils.BATCH_SIZE

TRAIN_DATASET = CustomModel3DCNNUtils.train_dataset_creation()
VAL_DATASET = CustomModel3DCNNUtils.validation_dataset_creation()

print("Creating Custom Model 3D CNN...")
model = CustomModel3DCNNUtils.create_model()


current_date = datetime.datetime.now().strftime("%Y-%m-%d")
current_time = datetime.datetime.now().strftime("%H-%M")


print("Creating Checkpoint...")
latest_checkpoint = CustomModel3DCNNUtils.create_checkpoint_latest()
checkpoint = CustomModel3DCNNUtils.create_checkpoint(current_date, current_time)


print("Training Model...")
start = timer()
history = model.fit(
    TRAIN_DATASET,
    epochs=EPOCHS,
    callbacks=[latest_checkpoint, checkpoint],
    verbose=1
)
end = timer()
model.save('../../Tmp/CustomModel_3D-CNN/custom_model_3d_cnn_10movies_tf', save_format='tf')

model.load_weights('../../Tmp/CustomModel_3D-CNN/custom_model_3d_cnn_latest_10movies.h5')
val_loss, val_acc = model.evaluate(VAL_DATASET)

CustomModel3DCNNUtils.save_training(
    end - start,
    current_date,
    current_time,
    history.history['accuracy'],
    history.history['loss'],
    val_acc,
    val_loss
)
