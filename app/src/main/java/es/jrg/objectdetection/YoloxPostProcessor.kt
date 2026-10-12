package es.jrg.objectdetection

import kotlin.math.exp

/**
 * Decodes the raw output of YOLOX (`[1, anchors, 5 + numClasses]`, anchors = 8400 for a 640 input) into detections.
 * Per anchor: raw x, y (offsets from the grid cell), raw w, h (log of the size in strides), objectness and class scores
 * (both already sigmoided). Anchors are the grid cells of the strides 8, 16 and 32, one stride after the other, row by row.
 */
object YoloxPostProcessor {
    private val STRIDES = intArrayOf(8, 16, 32)

    /** Number of anchors (rows of the output) for a square input of [inputSize]: 8400 for 640, 3549 for 416. */
    fun rows(inputSize: Int): Int = STRIDES.sumOf { (inputSize / it) * (inputSize / it) }

    fun decode(
        output: FloatArray,
        inputSize: Int,
        letterbox: Letterbox,
        confThreshold: Float = 0.25f,
        iouThreshold: Float = 0.45f,
        numClasses: Int = YoloPostProcessor.NUM_CLASSES,
        names: List<String> = CocoLabels.names,
    ): List<Detection> {
        val rowSize = 5 + numClasses
        val numAnchors = rows(inputSize)
        require(output.size >= rowSize * numAnchors) { "Unexpected output size ${output.size}" }
        val candidates = ArrayList<Detection>()
        var row = 0
        for (stride in STRIDES) {
            val cells = inputSize / stride
            for (gy in 0 until cells) {
                for (gx in 0 until cells) {
                    val base = row * rowSize
                    row++
                    val objectness = output[base + 4]
                    if (objectness <= confThreshold) continue
                    var best = -1
                    var bestScore = confThreshold
                    for (c in 0 until numClasses) {
                        val score = objectness * output[base + 5 + c]
                        if (score > bestScore) {
                            bestScore = score
                            best = c
                        }
                    }
                    if (best < 0) continue
                    val cx = (output[base] + gx) * stride
                    val cy = (output[base + 1] + gy) * stride
                    val w = exp(output[base + 2]) * stride
                    val h = exp(output[base + 3]) * stride
                    val label = names.getOrElse(best) { "class $best" }
                    YoloPostProcessor.toDetection(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, best, bestScore, letterbox, label)
                        ?.let(candidates::add)
                }
            }
        }
        return YoloPostProcessor.nms(candidates, iouThreshold)
    }
}
