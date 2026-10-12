# ONNX Runtime and LiteRT call into native code and look classes up by name: keep them whole.
-keep class ai.onnxruntime.** { *; }
-keep class com.google.ai.edge.litert.** { *; }
-dontwarn ai.onnxruntime.**
-dontwarn com.google.ai.edge.litert.**
-dontwarn org.tensorflow.lite.**
