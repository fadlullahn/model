import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Memuat dataset
train_dir = 'image/train'  # Ganti dengan path direktori training
val_dir = 'image/valid'  # Ganti dengan path direktori validasi
image_size = (320, 320)
batch_size = 32

# Augmentasi dan generator gambar
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Membangun model MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(320, 320, 3))

# Menambahkan layer klasifikasi baru
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Gabungkan model basis dengan lapisan klasifikasi yang baru
model = Model(inputs=base_model.input, outputs=predictions)

# Membekukan layer basis yang ada
for layer in base_model.layers:
    layer.trainable = False

# Kompilasi model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Pelatihan model
epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size
)

# Evaluasi model
loss, accuracy = model.evaluate(val_generator, steps=val_generator.samples // batch_size)
print(f'Validation loss: {loss:.4f}, Validation accuracy: {accuracy:.4f}')
