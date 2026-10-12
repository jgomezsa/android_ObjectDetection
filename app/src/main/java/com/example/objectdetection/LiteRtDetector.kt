package com.example.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer

/**
 * Runs a YOLOX TFLite model (NHWC float input `[1, size, size, 3]` in BGR 0..255, output `[1, anchors, 5 + classes]`) with LiteRT on the
 * given [backend]: the GPU (compiled program cached in the app's cache directory, so only the first start of a model pays the
 * compilation) or the CPU (XNNPACK). The model is a bundled asset (`<id>.tflite`) or a downloaded file. Creation runs the model once
 * and throws if the backend cannot run it or the model does not fit the YOLOX contract, so the caller can move on to the next backend.
 *
 * The model must have static tensor shapes: the GPU delegate rejects dynamic ones (see TECHNICAL_DOCUMENTATION.md for how to convert).
 */
class LiteRtDetector(context: Context, private val info: ModelInfo, override val backend: Backend) : ObjectDetector {
    private val env = Environment.create(context)

    // lateinit so that close() can release whatever was created when creation fails halfway.
    private lateinit var model: CompiledModel
    private lateinit var inputBuffers: List<TensorBuffer>
    private lateinit var outputBuffers: List<TensorBuffer>
    private lateinit var renderer: LetterboxRenderer
    private lateinit var input: FloatArray
    private lateinit var names: List<String>

    /** Square input side, derived from the size of the model's input tensor. */
    var inputSize = 0
        private set

    /** Number of classes, derived from the size of the model's output tensor. */
    var numClasses = 0
        private set

    private var anchors = 0

    init {
        try {
            val options = when (backend) {
                Backend.GPU -> {
                    // The cache key includes the model size, so replacing a model file never picks up a stale compiled program.
                    val gpu = CompiledModel.GpuOptions(
                        precision = CompiledModel.GpuOptions.Precision.FP16,
                        serializationDir = context.cacheDir.absolutePath,
                        modelCacheKey = "${info.id}-${info.sizeBytes}",
                        serializeProgramCache = true,
                    )
                    CompiledModel.Options(Accelerator.GPU).apply { gpuOptions = gpu }
                }
                Backend.CPU -> CompiledModel.Options(Accelerator.CPU).apply {
                    cpuOptions = CompiledModel.CpuOptions(numThreads = Runtime.getRuntime().availableProcessors().coerceIn(2, 6))
                }
            }
            val file = info.file
            model = if (file == null) {
                CompiledModel.create(context.assets, "${info.id}.tflite", options, env)
            } else {
                CompiledModel.create(file.absolutePath, options, env)
            }
            inputBuffers = model.createInputBuffers()
            outputBuffers = model.createOutputBuffers()
            val inputElements = inputBuffers[0].readFloat().size
            inputSize = ModelContract.fromSizes(inputElements, outputBuffers[0].readFloat().size).inputSize
            anchors = YoloxPostProcessor.rows(inputSize)
            renderer = LetterboxRenderer(inputSize)
            input = FloatArray(inputSize * inputSize * 3)
            selfTest()
        } catch (e: Throwable) {
            runCatching { close() }
            throw e
        }
        Log.d(TAG, "${info.id} ready on the ${backend.label} (input ${inputSize}x$inputSize, $numClasses classes)")
    }

    /** Runs the model once on a gray image; a backend that compiles but cannot run, or a model that is not YOLOX-style, is rejected here. */
    private fun selfTest() {
        input.fill(114f)
        inputBuffers[0].writeFloat(input)
        model.run(inputBuffers, outputBuffers)
        val raw = outputBuffers[0].readFloat()
        numClasses = ModelContract.classesFor(inputSize, raw.size)
        ModelContract.checkValues(raw, anchors, 5 + numClasses)
        names = LabelParser.namesFor(numClasses, info.labels)
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

        val detections = YoloxPostProcessor.decode(raw, inputSize, letterbox, numClasses = numClasses, names = names)
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
    }
}
