package com.example.objectdetection

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Runs a YOLOX ONNX model (NCHW float input `[1, 3, size, size]` in BGR 0..255, output `[1, anchors, 5 + classes]`) with ONNX Runtime on the
 * CPU. ONNX Runtime for Android has no GPU backend, so these models are slower than TFLite ones on the GPU, but a downloaded `.onnx` file
 * needs no conversion. Creation runs the model once and throws when the file does not fit the YOLOX contract.
 */
class OrtDetector(@Suppress("UNUSED_PARAMETER") context: Context, private val info: ModelInfo, override val backend: Backend = Backend.CPU) : ObjectDetector {
    private val env = OrtEnvironment.getEnvironment()
    private lateinit var session: OrtSession
    private lateinit var inputName: String
    private lateinit var renderer: LetterboxRenderer
    private lateinit var inputBuffer: FloatBuffer
    private lateinit var raw: FloatArray
    private lateinit var names: List<String>
    private var anchors = 0

    var inputSize = 0
        private set
    var numClasses = 0
        private set

    init {
        require(backend == Backend.CPU) { "ONNX models run on the CPU only" }
        try {
            val file = info.file ?: throw ModelRejectedException("ONNX models have to be downloaded files.")
            val options = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(2, 6))
            }
            session = env.createSession(file.absolutePath, options)
            inputName = session.inputNames.first()
            val inputShape = (session.inputInfo[inputName]!!.info as TensorInfo).shape
            val outputShape = (session.outputInfo.values.first().info as? TensorInfo)?.shape
                ?: throw ModelRejectedException("The model output is not a tensor.")
            val shape = ModelContract.fromOnnxShapes(inputShape, outputShape)
            inputSize = shape.inputSize
            anchors = YoloxPostProcessor.rows(inputSize)
            renderer = LetterboxRenderer(inputSize)
            inputBuffer = ByteBuffer.allocateDirect(4 * 3 * inputSize * inputSize).order(ByteOrder.nativeOrder()).asFloatBuffer()
            selfTest()
        } catch (e: Throwable) {
            runCatching { close() }
            throw e
        }
        Log.d(TAG, "${info.id} ready on the CPU (ONNX Runtime, input ${inputSize}x$inputSize, $numClasses classes)")
    }

    private fun selfTest() {
        for (i in 0 until inputBuffer.capacity()) inputBuffer.put(i, 114f)
        val values = run()
        numClasses = ModelContract.classesFor(inputSize, values.size)
        raw = values
        ModelContract.checkValues(values, anchors, 5 + numClasses)
        names = LabelParser.namesFor(numClasses, info.labels)
    }

    /** Runs the model on [inputBuffer] and returns the flattened output. */
    private fun run(): FloatArray {
        inputBuffer.rewind()
        OnnxTensor.createTensor(env, inputBuffer, longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { result ->
                val out = (result[0] as OnnxTensor).floatBuffer
                out.rewind()
                val values = if (::raw.isInitialized && raw.size == out.remaining()) raw else FloatArray(out.remaining())
                out.get(values)
                return values
            }
        }
    }

    override fun detect(bitmap: Bitmap): DetectionResult {
        val t0 = System.nanoTime()
        val letterbox = Letterbox(bitmap.width, bitmap.height, inputSize)
        val pixels = renderer.render(bitmap, letterbox)
        // CHW float, BGR in 0..255 as YOLOX was trained.
        val plane = inputSize * inputSize
        for (i in 0 until plane) {
            val p = pixels[i]
            inputBuffer.put(i, (p and 0xFF).toFloat())
            inputBuffer.put(plane + i, ((p shr 8) and 0xFF).toFloat())
            inputBuffer.put(2 * plane + i, ((p shr 16) and 0xFF).toFloat())
        }
        val t1 = System.nanoTime()
        val values = run()
        val t2 = System.nanoTime()
        val detections = YoloxPostProcessor.decode(values, inputSize, letterbox, numClasses = numClasses, names = names)
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
        if (::session.isInitialized) session.close()
    }

    private companion object {
        const val TAG = "OrtDetector"
    }
}
