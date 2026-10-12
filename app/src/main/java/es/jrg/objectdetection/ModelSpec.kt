package es.jrg.objectdetection

/** How the pixels are fed to a model: channel order and value range. */
enum class InputConvention {
    /** Blue, green, red in 0..255 (YOLOX). */
    BGR_255,

    /** Red, green, blue in 0..1 (YOLOv5/v8/v11, RT-DETR, D-FINE). */
    RGB_01,
}

/** The kinds of model output the app can decode. [family] is the name shown to the user. */
enum class OutputLayout(val family: String) {
    /** `[1, anchors, 5 + classes]`: raw offsets and log sizes, objectness, class probabilities (YOLOX). */
    YOLOX_ROWS("YOLOX"),

    /** `[1, 4 + classes, anchors]`: box centre and size in input pixels, class probabilities, no objectness (YOLOv8, YOLO11). */
    V8_CHANNELS_FIRST("YOLOv8/11"),

    /** The same as [V8_CHANNELS_FIRST] with the axes swapped: `[1, anchors, 4 + classes]`. */
    V8_ROWS("YOLOv8/11"),

    /** Two outputs: class logits `[1, queries, classes]` and boxes `[1, queries, 4]` (normalised centre and size): RT-DETR, D-FINE. */
    DETR_SPLIT("DETR"),

    /** One output `[1, queries, 4 + classes]`: normalised box centre and size, then class probabilities. */
    DETR_MERGED("DETR"),

    /** `[1, queries, 6]`: x1, y1, x2, y2, score, class, already filtered (YOLOv10 and other NMS-free exports). */
    END2END("NMS-free YOLO"),
}

/**
 * Everything the app needs to know to run and decode a model, found by running it once ([ModelContract.analyze]) and stored with the model.
 * For [OutputLayout.DETR_SPLIT], [logitsIndex] and [boxesIndex] say which of the two outputs is which.
 */
data class ModelSpec(
    val layout: OutputLayout,
    val inputSize: Int,
    val numClasses: Int,
    val queries: Int = 0,
    val logitsIndex: Int = 0,
    val boxesIndex: Int = 1,
    /** YOLOv8/11 only: the boxes are fractions of the input size (0..1, as in Ultralytics TFLite exports) instead of pixels. */
    val normalizedBoxes: Boolean = false,
) {
    val family: String get() = layout.family

    /** YOLOX wants BGR 0..255; every other supported family RGB 0..1. */
    val convention: InputConvention get() = if (layout == OutputLayout.YOLOX_ROWS) InputConvention.BGR_255 else InputConvention.RGB_01

    /** DETR-style models are trained on images resized to the full square; the others on letterboxed images. */
    val stretch: Boolean get() = layout == OutputLayout.DETR_SPLIT || layout == OutputLayout.DETR_MERGED

    /** Number of anchors of a YOLO-style output for this input size (8400 at 640). */
    val anchors: Int get() = YoloxPostProcessor.rows(inputSize)

    /** The number of values each output must have, in output order. */
    fun expectedOutputSizes(): List<Int> = when (layout) {
        OutputLayout.YOLOX_ROWS -> listOf(anchors * (5 + numClasses))
        OutputLayout.V8_CHANNELS_FIRST, OutputLayout.V8_ROWS -> listOf(anchors * (4 + numClasses))
        OutputLayout.DETR_MERGED -> listOf(queries * (4 + numClasses))
        OutputLayout.END2END -> listOf(queries * 6)
        OutputLayout.DETR_SPLIT -> {
            val sizes = IntArray(2)
            sizes[logitsIndex] = queries * numClasses
            sizes[boxesIndex] = queries * 4
            sizes.toList()
        }
    }

    fun toProperties(): Map<String, String> = mapOf(
        "layout" to layout.name,
        "input" to inputSize.toString(),
        "classes" to numClasses.toString(),
        "queries" to queries.toString(),
        "logits" to logitsIndex.toString(),
        "boxes" to boxesIndex.toString(),
        "normalized" to normalizedBoxes.toString(),
    )

    companion object {
        fun yolox(inputSize: Int, numClasses: Int) = ModelSpec(OutputLayout.YOLOX_ROWS, inputSize, numClasses)

        /** Reads a spec written by [toProperties]; a missing layout means YOLOX, which is what models added by earlier versions were. */
        fun fromProperties(get: (String) -> String?): ModelSpec = ModelSpec(
            layout = get("layout")?.let { OutputLayout.valueOf(it) } ?: OutputLayout.YOLOX_ROWS,
            inputSize = get("input")!!.toInt(),
            numClasses = get("classes")!!.toInt(),
            queries = get("queries")?.toInt() ?: 0,
            logitsIndex = get("logits")?.toInt() ?: 0,
            boxesIndex = get("boxes")?.toInt() ?: 1,
            normalizedBoxes = get("normalized")?.toBoolean() ?: false,
        )
    }
}
