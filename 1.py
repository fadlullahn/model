import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

# Mengunduh model MobileNetV2 dengan bobot pra-latih (tanpa lapisan atas)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(320, 320, 3))

# Menyimpan bobot model ke file lokal
base_model.save_weights('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_320_no_top.h5')
