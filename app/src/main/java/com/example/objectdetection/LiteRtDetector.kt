package com.example.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Runs a YOLOX TFLite model (NHWC float input `[1, size, size, 3]` in BGR 0..255, output `[1, anchors, 85]`) with LiteRT on the
 * given [backend]: the GPU (compiled program cached in the app's cache directory, so only the first start of a model pays the
 * compilation) or the CPU (XNNPACK). Creation runs the model once and throws if the backend cannot run it, so the caller can
 * move on to the next backend.
 *
 * The model must have static tensor shapes: the GPU delegate rejects dynamic ones (see TECHNICAL_DOCUMENTATION.md for how to convert).
 */
class LiteRtDetector(context: Context, modelAsset: String, override val backend: Backend) : ObjectDetector {
    private val env = Environment.create(context)
    // lateinit so that close() can release whatever was created when creation fails halfway.
    private lateinit var model: CompiledModel
    private lateinit var inputBuffers: List<TensorBuffer>
    private lateinit var outputBuffers: List<TensorBuffer>

    /** Square input side, derived from the size of the model's input tensor. */
    private var inputSize = 0
    private lateinit var renderer: LetterboxRenderer
    private lateinit var input: FloatArray

    init {
        try {
            val options = when (backend) {
                Backend.GPU -> {
                    // The cache key includes the model size, so replacing a model file never picks up a stale compiled program.
                    val modelBytes = context.assets.open(modelAsset).use { it.available() }
                    val gpu = CompiledModel.GpuOptions(
                        precision = CompiledModel.GpuOptions.Precision.FP16,
                        serializationDir = context.cacheDir.absolutePath,
                        modelCacheKey = "${modelAsset.removeSuffix(".tflite")}-$modelBytes",
                        serializeProgramCache = true,
                    )
                    CompiledModel.Options(Accelerator.GPU).apply { gpuOptions = gpu }
                }
                Backend.CPU -> CompiledModel.Options(Accelerator.CPU).apply {
                    cpuOptions = CompiledModel.CpuOptions(numThreads = Runtime.getRuntime().availableProcessors().coerceIn(2, 6))
                }
            }
            model = CompiledModel.create(context.assets, modelAsset, options, env)
            inputBuffers = model.createInputBuffers()
            outputBuffers = model.createOutputBuffers()
            inputSize = sqrt(inputBuffers[0].readFloat().size / 3.0).roundToInt()
            renderer = LetterboxRenderer(inputSize)
            input = FloatArray(inputSize * inputSize * 3)
            selfTest()
        } catch (e: Throwable) {
            runCatching { close() }
            throw e
        }
        Log.d(TAG, "$modelAsset ready on the ${backend.label} (input ${inputSize}x$inputSize)")
    }

    /** Runs the model once on a gray image; a backend that compiles but cannot run (or returns garbage) is rejected here. */
    private fun selfTest() {
        input.fill(114f)
        inputBuffers[0].writeFloat(input)
        model.run(inputBuffers, outputBuffers)
        val raw = outputBuffers[0].readFloat()
        check(raw.size == YoloxPostProcessor.rows(inputSize) * ROW_SIZE) { "Unexpected output size ${raw.size}" }
        check(raw.all { it.isFinite() }) { "The model returned NaN or infinite values" }
    }

    override fun detect(bitmap: Bitmap): DetectionResult {
        val t0 = System.nanoTime()
        val letterbox = Letterbox(bitmap.width, bitmap.height, inputSize)
        val pixels = renderer.render(bitmap, letterbox)
        // HWC float, BGR in 0..255 as YOLOX was trained (TFLite models from onnx2tf are NHWC).
        for (i in pixels.indices) {
            val p = pixels[i]
            input[3 * i] = (p and 0xFF).toFloat()
            input[3 * i + 1] = ((p shr 8) and 0xFF).toFloat()
            input[3 * i + 2] = ((p shr 16) and 0xFF).toFloat()
        }
        inputBuffers[0].writeFloat(input)
        val t1 = System.nanoTime()

        model.run(inputBuffers, outputBuffers)
        val raw = outputBuffers[0].readFloat()
        val t2 = System.nanoTime()

        val detections = YoloxPostProcessor.decode(raw, inputSize, letterbox)
        val t3 = System.nanoTime()
        return DetectionResult(
            detections,
            inferenceMs = (t3 - t0) / 1_000_000,
            preprocessMs = (t1 - t0) / 1_000_000,
            runMs = (t2 - t1) / 1_000_000,
            postMs = (t3 - t2) / 1_000_000,
        )
    }

    override fun close() {
        if (::inputBuffers.isInitialized) inputBuffers.forEach { it.close() }
        if (::outputBuffers.isInitialized) outputBuffers.forEach { it.close() }
        if (::model.isInitialized) model.close()
        env.close()
    }

    private companion object {
        const val TAG = "LiteRtDetector"
        const val ROW_SIZE = 5 + YoloPostProcessor.NUM_CLASSES
    }
}
