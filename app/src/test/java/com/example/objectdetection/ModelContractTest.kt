package com.example.objectdetection

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelContractTest {
    private fun rejected(block: () -> Unit): String {
        try {
            block()
        } catch (e: ModelRejectedException) {
            return e.message!!
        }
        throw AssertionError("expected the model to be rejected")
    }

    @Test
    fun tfliteSizesGiveInputSizeAndClasses() {
        assertEquals(ModelContract.Shape(640, 80), ModelContract.fromSizes(640 * 640 * 3, 8400 * 85))
        assertEquals(ModelContract.Shape(416, 80), ModelContract.fromSizes(416 * 416 * 3, 3549 * 85))
        assertEquals(ModelContract.Shape(640, 1), ModelContract.fromSizes(640 * 640 * 3, 8400 * 6))
        assertEquals(ModelContract.Shape(960, 3), ModelContract.fromSizes(960 * 960 * 3, 18900 * 8))
    }

    @Test
    fun tfliteRejectsOtherInputsAndOutputs() {
        assertTrue(rejected { ModelContract.fromSizes(640 * 480 * 3, 8400 * 85) }.contains("square"))
        assertTrue(rejected { ModelContract.fromSizes(100, 8400 * 85) }.contains("square"))
        // YOLOv8-style [1, 84, 8400] has 84 * 8400 values, which fits "84 columns": the value check catches it, but SSD-style outputs fail here.
        assertTrue(rejected { ModelContract.fromSizes(640 * 640 * 3, 100 * 6) }.contains("different kind of model"))
        assertTrue(rejected { ModelContract.fromSizes(640 * 640 * 3, 8400 * 5) }.contains("no class"))
    }

    @Test
    fun onnxShapes() {
        assertEquals(ModelContract.Shape(640, 80), ModelContract.fromOnnxShapes(longArrayOf(1, 3, 640, 640), longArrayOf(1, 8400, 85)))
        assertEquals(ModelContract.Shape(416, 3), ModelContract.fromOnnxShapes(longArrayOf(1, 3, 416, 416), longArrayOf(1, 3549, 8)))
        // Dynamic height/width: 640 is assumed.
        assertEquals(ModelContract.Shape(640, 80), ModelContract.fromOnnxShapes(longArrayOf(1, 3, -1, -1), longArrayOf(1, 8400, 85)))
        assertEquals(ModelContract.Shape(640, 80), ModelContract.fromOnnxShapes(longArrayOf(-1, 3, 640, 640), longArrayOf(-1, -1, 85)))
    }

    @Test
    fun onnxRejectsOtherFamilies() {
        assertTrue(rejected { ModelContract.fromOnnxShapes(longArrayOf(1, 3, 640, 640), longArrayOf(1, 84, 8400)) }.contains("channels-first"))
        assertTrue(rejected { ModelContract.fromOnnxShapes(longArrayOf(1, 3, 640, 640), longArrayOf(1, 300, 6)) }.contains("different kind"))
        assertTrue(rejected { ModelContract.fromOnnxShapes(longArrayOf(1, 3, 640, 640), longArrayOf(300, 6)) }.contains("post-processing"))
        assertTrue(rejected { ModelContract.fromOnnxShapes(longArrayOf(1, 3, 736, 1280), longArrayOf(1, 8400, 85)) }.contains("square"))
        assertTrue(rejected { ModelContract.fromOnnxShapes(longArrayOf(1, 640, 640, 3), longArrayOf(1, 8400, 85)) }.contains("channels-first"))
        assertTrue(rejected { ModelContract.fromOnnxShapes(longArrayOf(1, 3, 650, 650), longArrayOf(1, 8400, 85)) }.contains("multiple of 32"))
        assertTrue(rejected { ModelContract.fromOnnxShapes(longArrayOf(3, 640), longArrayOf(1, 8400, 85)) }.contains("4"))
    }

    private fun output(rows: Int, rowSize: Int, fill: (row: Int, column: Int) -> Float) =
        FloatArray(rows * rowSize) { fill(it / rowSize, it % rowSize) }

    @Test
    fun rawYoloxValuesPass() {
        // Raw boxes are small offsets / log sizes; objectness and class scores are probabilities.
        val raw = output(100, 85) { _, c -> if (c < 4) -1.5f + c else 0.01f }
        ModelContract.checkValues(raw, 100, 85)
    }

    @Test
    fun valuesThatAreNotRawYoloxAreRejected() {
        assertTrue(rejected { ModelContract.checkValues(output(10, 85) { _, c -> if (c == 7) Float.NaN else 0f }, 10, 85) }.contains("NaN"))
        // Logits instead of probabilities.
        assertTrue(rejected { ModelContract.checkValues(output(10, 85) { _, c -> if (c == 4) 3.2f else 0f }, 10, 85) }.contains("probabilities"))
        // Already decoded boxes (pixel coordinates).
        assertTrue(rejected { ModelContract.checkValues(output(10, 85) { _, c -> if (c == 0) 320f else 0f }, 10, 85) }.contains("decoded"))
    }

    @Test
    fun channelsFirstOutputIsCaughtByTheValues() {
        // A YOLOv8-style [1, 84, 8400] output read as 8400 rows of 84: pixel box values end up in the score columns.
        val yolov8 = FloatArray(84 * 8400) { if (it < 4 * 8400) 300f else 0.2f }
        assertTrue(rejected { ModelContract.checkValues(yolov8, 8400, 84) }.isNotEmpty())
    }
}
