import tensorflow as tf
import tfjs_graph_converter.api as tfjs

tfjs.graph_model_to_saved_model(
    "posenet_resnet50_js/posenet_resnet50_quantized_2_1_default_1/model-stride16.json",
    "tflite_model"
)

# Code below taken from https://www.tensorflow.org/lite/convert/python_api
converter = tf.lite.TFLiteConverter.from_saved_model("realsavedmodel")
tflite_model = converter.convert()

# Save the TF Lite model.
with tf.io.gfile.GFile('model.tflite', 'wb') as f:
    f.write(tflite_model)
