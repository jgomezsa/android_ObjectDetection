package com.example.objectdetection

import kotlin.math.abs
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * What a model file must look like for the app to run it: a YOLOX-style detector with a square image input and an undecoded output
 * `[1, anchors, 5 + classes]` (box offsets, objectness and class scores that are already probabilities). The checks give the user a
 * reason when a file does not fit.
 */
object ModelContract {
    /** Input size assumed for ONNX models whose input height and width are dynamic. */
    const val DYNAMIC_INPUT_SIZE = 640

    /** Largest raw box value expected on an undecoded YOLOX output (offsets and log sizes are small; decoded pixels are not). */
    private const val MAX_RAW_BOX = 64f

    data class Shape(val inputSize: Int, val numClasses: Int)

    /** For TFLite files, where only the element counts of the input and output tensors are known. */
    fun fromSizes(inputElements: Int, outputElements: Int): Shape {
        val size = squareSize(inputElements)
            ?: throw ModelRejectedException("The model input is not a square 3-channel image ($inputElements values), so it is not a YOLOX-style detector.")
        return Shape(size, classesFor(size, outputElements))
    }

    /** For ONNX files. Dynamic dimensions are reported as -1 (or 0). */
    fun fromOnnxShapes(input: LongArray, output: LongArray): Shape {
        if (input.size != 4) throw ModelRejectedException("The model input has ${input.size} dimensions; a YOLOX image input has 4 ([1, 3, height, width]).")
        if (input[1] != 3L) {
            throw ModelRejectedException("The model input is not channels-first ([1, 3, height, width]), which is what ONNX YOLOX models use.")
        }
        val height = input[2]
        val width = input[3]
        if (height > 0 && width > 0 && height != width) {
            throw ModelRejectedException("The model input is ${width}x$height. Only square inputs are supported.")
        }
        val size = (if (height > 0) height else if (width > 0) width else DYNAMIC_INPUT_SIZE.toLong()).toInt()
        if (size % 32 != 0 || size < 64) throw ModelRejectedException("The model input size $size is not a multiple of 32.")
        val rows = YoloxPostProcessor.rows(size)
        if (output.size != 3) {
            throw ModelRejectedException("The model output has ${output.size} dimensions; a YOLOX output has 3 ([1, $rows, 5 + classes]). Models with built-in post-processing are not supported.")
        }
        val anchors = output[1]
        val columns = output[2]
        if (anchors > 0 && anchors != rows.toLong()) {
            if (columns == rows.toLong()) {
                throw ModelRejectedException("The output is channels-first ([1, ${anchors}, $rows]) like YOLOv5/v8/v11 models. Only YOLOX output ([1, $rows, 5 + classes]) is supported.")
            }
            throw ModelRejectedException("The model has $anchors output rows, but a YOLOX detector with a ${size}x$size input has $rows. It is a different kind of model.")
        }
        if (columns in 1..5) throw ModelRejectedException("The model output has only $columns columns; a YOLOX output has 5 + classes.")
        return Shape(size, if (columns > 5) (columns - 5).toInt() else 0)
    }

    /** Number of classes from the output element count for an input of [size]; rejects outputs that do not match YOLOX. */
    fun classesFor(size: Int, outputElements: Int): Int {
        val rows = YoloxPostProcessor.rows(size)
        if (outputElements % rows != 0) {
            throw ModelRejectedException(
                "The model output has $outputElements values, which does not fit $rows anchors (what a YOLOX detector with a ${size}x$size input has). " +
                    "It is a different kind of model (YOLOv5/v8/v11, SSD, EfficientDet, RT-DETR...).",
            )
        }
        val rowSize = outputElements / rows
        if (rowSize < 6) throw ModelRejectedException("The model output has no class scores.")
        return rowSize - 5
    }

    /** Checks the values of an output obtained on a gray test image; throws when they do not look like raw YOLOX output. */
    fun checkValues(raw: FloatArray, rows: Int, rowSize: Int) {
        var maxBox = 0f
        for (row in 0 until rows) {
            val base = row * rowSize
            for (c in 0 until rowSize) {
                val v = raw[base + c]
                if (!v.isFinite()) throw ModelRejectedException("The model returned NaN or infinite values.")
                if (c < 4) {
                    maxBox = maxOf(maxBox, abs(v))
                } else if (v < -0.001f || v > 1.001f) {
                    throw ModelRejectedException(
                        "The objectness/class scores are not probabilities between 0 and 1. The model must apply the sigmoid itself " +
                            "(as the official YOLOX exports do), or it is a different kind of model.",
                    )
                }
            }
        }
        if (maxBox > MAX_RAW_BOX) {
            throw ModelRejectedException("The boxes look already decoded (pixel coordinates). The app expects the raw YOLOX output, decoded on the phone.")
        }
    }

    /** `sqrt(elements / 3)` when the input is a square 3-channel image whose side is a multiple of 32 (and at least 64). */
    private fun squareSize(elements: Int): Int? {
        if (elements % 3 != 0) return null
        val size = sqrt((elements / 3).toDouble()).roundToInt()
        return if (size * size * 3 == elements && size % 32 == 0 && size >= 64) size else null
    }
}
