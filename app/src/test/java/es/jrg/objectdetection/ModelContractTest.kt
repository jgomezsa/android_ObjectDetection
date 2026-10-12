package es.jrg.objectdetection

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelContractTest {
    private val size = 640
    private val anchors = 8400

    private fun rejected(block: () -> Unit): String {
        try {
            block()
        } catch (e: ModelRejectedException) {
            return e.message!!
        }
        throw AssertionError("expected the model to be rejected")
    }

    // Synthetic outputs on a gray image, shaped like the real ones (see TECHNICAL_DOCUMENTATION.md 6.2).
    private fun yoloxRows(classes: Int) = FloatArray(anchors * (5 + classes)) { if (it % (5 + classes) < 4) -1.5f + it % (5 + classes) else 0.01f }
    private fun v8ChannelsFirst(classes: Int) = FloatArray((4 + classes) * anchors) { if (it / anchors < 4) (it % anchors) / 10f else 0.02f }
    private fun v8Rows(classes: Int) = FloatArray(anchors * (4 + classes)) { if (it % (4 + classes) < 4) 200f else 0.02f }
    private fun end2end(queries: Int = 300) = FloatArray(queries * 6) {
        when (it % 6) {
            0 -> 10f
            1 -> 20f
            2 -> 110f
            3 -> 120f
            4 -> 0.9f
            else -> 17f
        }
    }
    private fun merged(queries: Int = 300, classes: Int = 80) = FloatArray(queries * (4 + classes)) { if (it % (4 + classes) < 4) 0.5f else 0.02f }

    @Test
    fun yoloxRawOutput() {
        val spec = ModelContract.analyze(size, listOf(yoloxRows(80)), listOf(longArrayOf(1, 8400, 85)))
        assertEquals(ModelSpec(OutputLayout.YOLOX_ROWS, 640, 80), spec)
        assertEquals(ModelSpec(OutputLayout.YOLOX_ROWS, 640, 1), ModelContract.analyze(size, listOf(yoloxRows(1))))
        assertEquals(InputConvention.BGR_255, spec.convention)
        assertEquals(false, spec.stretch)
    }

    @Test
    fun yolov8ChannelsFirstWithAndWithoutShape() {
        val expected = ModelSpec(OutputLayout.V8_CHANNELS_FIRST, 640, 80)
        assertEquals(expected, ModelContract.analyze(size, listOf(v8ChannelsFirst(80)), listOf(longArrayOf(1, 84, 8400))))
        // A TFLite model gives no shape: the values decide.
        assertEquals(expected, ModelContract.analyze(size, listOf(v8ChannelsFirst(80))))
        assertEquals(InputConvention.RGB_01, expected.convention)
    }

    @Test
    fun ultralyticsTfliteHasNormalisedBoxesAndIsNotMistakenForDetr() {
        // The real pothole model: [1, 5, 8400], one class, boxes as fractions of the input. 42000 values also divide into 300 queries x 140.
        val raw = FloatArray(5 * anchors) { if (it / anchors < 4) 0.4f else 0.01f }
        val spec = ModelContract.analyze(size, listOf(raw))
        assertEquals(ModelSpec(OutputLayout.V8_CHANNELS_FIRST, 640, 1, normalizedBoxes = true), spec)
        // Pixel boxes stay pixel boxes.
        assertEquals(false, ModelContract.analyze(size, listOf(v8ChannelsFirst(80))).normalizedBoxes)
        val stored = spec.toProperties()
        assertEquals(spec, ModelSpec.fromProperties { stored[it] })
    }

    @Test
    fun yolov8SingleClass() {
        assertEquals(ModelSpec(OutputLayout.V8_CHANNELS_FIRST, 640, 1), ModelContract.analyze(size, listOf(v8ChannelsFirst(1))))
    }

    @Test
    fun yolov8RowsLayout() {
        val expected = ModelSpec(OutputLayout.V8_ROWS, 640, 80)
        assertEquals(expected, ModelContract.analyze(size, listOf(v8Rows(80)), listOf(longArrayOf(1, 8400, 84))))
        assertEquals(expected, ModelContract.analyze(size, listOf(v8Rows(80))))
    }

    @Test
    fun nmsFreeYolo() {
        val expected = ModelSpec(OutputLayout.END2END, 640, ModelContract.END2END_CLASSES, 300)
        assertEquals(expected, ModelContract.analyze(size, listOf(end2end()), listOf(longArrayOf(1, 300, 6))))
        assertEquals(expected, ModelContract.analyze(size, listOf(end2end())))
    }

    @Test
    fun detrWithMergedOutput() {
        val expected = ModelSpec(OutputLayout.DETR_MERGED, 640, 80, 300)
        assertEquals(expected, ModelContract.analyze(size, listOf(merged()), listOf(longArrayOf(1, 300, 84))))
        assertEquals(expected, ModelContract.analyze(size, listOf(merged())))
        assertEquals(true, expected.stretch)
    }

    @Test
    fun detrWithSeparateLogitsAndBoxesInEitherOrder() {
        val logits = FloatArray(300 * 80) { -5f }
        val boxes = FloatArray(300 * 4) { 0.4f }
        val dynamicBatch = listOf(longArrayOf(-1, 300, 80), longArrayOf(-1, 300, 4))
        assertEquals(ModelSpec(OutputLayout.DETR_SPLIT, 640, 80, 300, 0, 1), ModelContract.analyze(size, listOf(logits, boxes), dynamicBatch))
        assertEquals(ModelSpec(OutputLayout.DETR_SPLIT, 640, 80, 300, 1, 0), ModelContract.analyze(size, listOf(boxes, logits), dynamicBatch.reversed()))
        assertEquals(ModelSpec(OutputLayout.DETR_SPLIT, 640, 80, 300, 0, 1), ModelContract.analyze(size, listOf(logits, boxes)))
    }

    @Test
    fun outputSizesPerLayout() {
        assertEquals(listOf(8400 * 85), ModelSpec(OutputLayout.YOLOX_ROWS, 640, 80).expectedOutputSizes())
        assertEquals(listOf(8400 * 84), ModelSpec(OutputLayout.V8_CHANNELS_FIRST, 640, 80).expectedOutputSizes())
        assertEquals(listOf(300 * 6), ModelSpec(OutputLayout.END2END, 640, 80, 300).expectedOutputSizes())
        assertEquals(listOf(300 * 84), ModelSpec(OutputLayout.DETR_MERGED, 640, 80, 300).expectedOutputSizes())
        assertEquals(listOf(300 * 80, 300 * 4), ModelSpec(OutputLayout.DETR_SPLIT, 640, 80, 300, 0, 1).expectedOutputSizes())
        assertEquals(listOf(300 * 4, 300 * 80), ModelSpec(OutputLayout.DETR_SPLIT, 640, 80, 300, 1, 0).expectedOutputSizes())
    }

    @Test
    fun specSurvivesBeingStoredAsProperties() {
        for (spec in listOf(
            ModelSpec(OutputLayout.YOLOX_ROWS, 416, 3),
            ModelSpec(OutputLayout.V8_CHANNELS_FIRST, 640, 1),
            ModelSpec(OutputLayout.DETR_SPLIT, 640, 80, 300, 1, 0),
            ModelSpec(OutputLayout.END2END, 640, 80, 300),
        )) {
            val stored = spec.toProperties()
            assertEquals(spec, ModelSpec.fromProperties { stored[it] })
        }
        // A model stored by an earlier version has no layout: it was a YOLOX model.
        val old = mapOf("input" to "640", "classes" to "1")
        assertEquals(ModelSpec(OutputLayout.YOLOX_ROWS, 640, 1), ModelSpec.fromProperties { old[it] })
    }

    @Test
    fun rejectsOutputsThatAreNoSupportedType() {
        assertTrue(rejected { ModelContract.analyze(size, listOf(FloatArray(1000) { 5f })) }.contains("Supported outputs"))
        assertTrue(rejected { ModelContract.analyze(size, listOf(FloatArray(1001))) }.contains("fits no supported type"))
        assertTrue(rejected { ModelContract.analyze(size, listOf(FloatArray(10), FloatArray(10), FloatArray(10))) }.contains("3 outputs"))
        assertTrue(rejected { ModelContract.analyze(size, listOf(FloatArray(300 * 80), FloatArray(300 * 4) { 300f })) }.contains("normalised"))
    }

    @Test
    fun rejectsScoresThatAreNotProbabilities() {
        val logits = FloatArray(anchors * 85) { if (it % 85 < 4) -1f else 3.2f }
        assertTrue(rejected { ModelContract.analyze(size, listOf(logits), listOf(longArrayOf(1, 8400, 85))) }.contains("probabilities"))
    }

    @Test
    fun rejectsNaN() {
        assertTrue(rejected { ModelContract.analyze(size, listOf(FloatArray(100) { Float.NaN })) }.contains("NaN"))
    }

    @Test
    fun rawYoloxBoxesFeedingAV8LookingShapeAreStillYolox() {
        // The same number of values fits YOLOX (85 columns) and a 4 + 81 class YOLOv8 read as rows; the small raw boxes decide.
        assertEquals(OutputLayout.YOLOX_ROWS, ModelContract.analyze(size, listOf(yoloxRows(80))).layout)
    }

    @Test
    fun tolerantOfSmallFp16Overshoot() {
        val spec = ModelContract.analyze(size, listOf(FloatArray(5 * anchors) { if (it / anchors < 4) (it % anchors) / 10f else 1.006f }))
        assertEquals(OutputLayout.V8_CHANNELS_FIRST, spec.layout)
    }

    @Test
    fun resolveTriesBothPixelScales() {
        val seen = ArrayList<Float>()
        // A model that only gives usable output on the 0..1 scale (the 0..255 gray saturates it into garbage).
        val spec = ModelContract.resolve(size, listOf(null)) { value ->
            seen += value
            listOf(if (value > 1f) FloatArray(1001) else v8ChannelsFirst(80))
        }
        assertEquals(OutputLayout.V8_CHANNELS_FIRST, spec.layout)
        assertEquals(listOf(114f, 114f / 255f), seen)
        // When both fail, the first reason is reported.
        assertTrue(rejected { ModelContract.resolve(size, listOf(null)) { listOf(FloatArray(1001)) } }.contains("fits no supported type"))
    }

    @Test
    fun verifyChecksSizesAndFiniteness() {
        val spec = ModelSpec(OutputLayout.V8_CHANNELS_FIRST, 640, 80)
        ModelContract.verify(spec, listOf(v8ChannelsFirst(80)))
        assertTrue(rejected { ModelContract.verify(spec, listOf(FloatArray(10))) }.contains("do not match"))
        assertTrue(rejected { ModelContract.verify(spec, listOf(FloatArray(84 * 8400) { Float.POSITIVE_INFINITY })) }.contains("NaN or infinite"))
        assertEquals(114f, ModelContract.grayFor(ModelSpec.yolox(640, 80)), 0f)
        assertEquals(114f / 255f, ModelContract.grayFor(spec), 0f)
    }

    @Test
    fun tfliteInputSizes() {
        assertEquals(640, ModelContract.tfliteInputSize(640 * 640 * 3))
        assertEquals(416, ModelContract.tfliteInputSize(416 * 416 * 3))
        assertTrue(rejected { ModelContract.tfliteInputSize(640 * 480 * 3) }.contains("square"))
        assertTrue(rejected { ModelContract.tfliteInputSize(100) }.contains("square"))
        assertTrue(rejected { ModelContract.tfliteInputSize(650 * 650 * 3) }.contains("square"))
    }

    @Test
    fun onnxInputSizes() {
        assertEquals(640, ModelContract.onnxInputSize(longArrayOf(1, 3, 640, 640)))
        assertEquals(960, ModelContract.onnxInputSize(longArrayOf(-1, 3, 960, 960)))
        // Dynamic height and width (the RT-DETR export from Hugging Face): 640 is assumed.
        assertEquals(640, ModelContract.onnxInputSize(longArrayOf(-1, 3, -1, -1)))
        assertTrue(rejected { ModelContract.onnxInputSize(longArrayOf(1, 3, 736, 1280)) }.contains("square"))
        assertTrue(rejected { ModelContract.onnxInputSize(longArrayOf(1, 640, 640, 3)) }.contains("channels-first"))
        assertTrue(rejected { ModelContract.onnxInputSize(longArrayOf(1, 3, 650, 650)) }.contains("multiple of 32"))
        assertTrue(rejected { ModelContract.onnxInputSize(longArrayOf(3, 640)) }.contains("4"))
    }
}
