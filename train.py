import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.optimizers import Adadelta, Adam, SGD, RMSprop
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from model import get_model
from callback import f1_class
from util import *
from functools import partial

#  global variables

batch_size = 512
epochs = 40
size = 32
nb1 = len(glob.glob('train32x32/train/1/*'))
nb0 = len(glob.glob('train32x32/train/0/*'))
input_shape = (size, size, 3)
steps_per_epochs = (len(glob.glob('train32x32/train/1/*'))+
                    len(glob.glob('train32x32/train/0/*')))/batch_size

validation_steps = (len(glob.glob('train32x32/val/1/*'))+
                    len(glob.glob('train32x32/val/0/*')))/batch_size

# Data generators

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'train32x32/train',
    target_size=(size,size),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    'train32x32/val',
    target_size=(size,size),
    batch_size=batch_size,
    class_mode='binary')


# Defined 2 generators without suffle for testing purpose
train_gen_no_shuffle = test_datagen.flow_from_directory(
    'train32x32/train',
    target_size=(size,size),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

val_gen_no_shuffle = test_datagen.flow_from_directory(
    'train32x32/val',
    target_size=(size,size),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)

mc = ModelCheckpoint('model_weights/final04.{epoch:02d}-{loss:.2f}.hdf5', save_weights_only=True)
es = EarlyStopping(monitor='f1', patience=20, mode='max', restore_best_weights=True)
rl = ReduceLROnPlateau(monitor='loss', patience=6, factor=0.5, verbose=1)

model = get_model()

# Partials functions to pass to the model
w0, w1 = class_weight(nb1, nb0)
wb_part = partial(wbce, weight1=w1, weight0=w0)
f1_cb = f1_class(val_gen_no_shuffle)

model.compile(loss=wb_part,
              optimizer=Adam(),
              metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epochs,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[f1_cb,rl, mc],
    verbose=2)

y_val_true = val_gen_no_shuffle.labels
y_val_pred = model.predict_generator(val_gen_no_shuffle)
best = f1(y_val_true, y_val_pred.flatten())

print("seuil de prédiction optimal : ", best.y_pred)