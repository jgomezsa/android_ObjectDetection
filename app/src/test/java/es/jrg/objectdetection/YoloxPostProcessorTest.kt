package es.jrg.objectdetection

import kotlin.math.ln
import org.junit.Assert.assertEquals
import org.junit.Test

class YoloxPostProcessorTest {
    private val size = 640
    private val rowSize = 85
    private val letterbox = Letterbox(640, 640, size)

    private fun emptyOutput() = FloatArray(8400 * rowSize)

    private fun FloatArray.setRow(row: Int, x: Float, y: Float, w: Float, h: Float, obj: Float, classId: Int, cls: Float) {
        val b = row * rowSize
        this[b] = x
        this[b + 1] = y
        this[b + 2] = w
        this[b + 3] = h
        this[b + 4] = obj
        this[b + 5 + classId] = cls
    }

    @Test
    fun decodesStride8Anchor() {
        val out = emptyOutput()
        // Cell (gx=10, gy=20) of the 80x80 grid: x = (0.5 + 10) * 8 = 84, y = (0.5 + 20) * 8 = 164, w = 4 * 8 = 32, h = 2 * 8 = 16.
        out.setRow(20 * 80 + 10, 0.5f, 0.5f, ln(4f), ln(2f), obj = 0.9f, classId = 3, cls = 0.8f)
        val d = YoloxPostProcessor.decode(out, size, letterbox).single()
        assertEquals(3, d.classId)
        assertEquals(0.72f, d.score, 1e-5f)
        assertEquals(68f / 640, d.left, 1e-5f)
        assertEquals(156f / 640, d.top, 1e-5f)
        assertEquals(100f / 640, d.right, 1e-5f)
        assertEquals(172f / 640, d.bottom, 1e-5f)
    }

    @Test
    fun decodesStride16And32Anchors() {
        val out = emptyOutput()
        // Stride 16 starts at row 6400: cell (gx=7, gy=5) of the 40x40 grid, centre x = (0 + 7) * 16 = 112, y = (0 + 5) * 16 = 80, size 16 x 16.
        out.setRow(6400 + 5 * 40 + 7, 0f, 0f, 0f, 0f, obj = 0.95f, classId = 0, cls = 0.9f)
        // Stride 32 starts at row 8000: cell (gx=3, gy=2) of the 20x20 grid, centre x = 96, y = 64, size 32 x 32.
        out.setRow(8000 + 2 * 20 + 3, 0f, 0f, 0f, 0f, obj = 0.95f, classId = 1, cls = 0.9f)
        val boxes = YoloxPostProcessor.decode(out, size, letterbox).sortedBy { it.classId }
        assertEquals(2, boxes.size)
        assertEquals(104f / 640, boxes[0].left, 1e-5f)
        assertEquals(72f / 640, boxes[0].top, 1e-5f)
        assertEquals(120f / 640, boxes[0].right, 1e-5f)
        assertEquals(88f / 640, boxes[0].bottom, 1e-5f)
        assertEquals(80f / 640, boxes[1].left, 1e-5f)
        assertEquals(48f / 640, boxes[1].top, 1e-5f)
        assertEquals(112f / 640, boxes[1].right, 1e-5f)
        assertEquals(80f / 640, boxes[1].bottom, 1e-5f)
    }

    @Test
    fun scoreIsObjectnessTimesClassAndFiltered() {
        val out = emptyOutput()
        // 0.6 * 0.4 = 0.24 is below the 0.25 threshold.
        out.setRow(0, 0f, 0f, 0f, 0f, obj = 0.6f, classId = 2, cls = 0.4f)
        assertEquals(0, YoloxPostProcessor.decode(out, size, letterbox).size)
    }

    @Test
    fun overlappingBoxesOfTheSameClassAreSuppressed() {
        val out = emptyOutput()
        out.setRow(20 * 80 + 10, 0.5f, 0.5f, ln(4f), ln(2f), obj = 0.9f, classId = 3, cls = 0.8f)
        out.setRow(20 * 80 + 11, 0.0f, 0.5f, ln(4f), ln(2f), obj = 0.9f, classId = 3, cls = 0.6f)
        assertEquals(1, YoloxPostProcessor.decode(out, size, letterbox).size)
    }

    @Test
    fun anchorCountMatchesTheInputSize() {
        assertEquals(8400, YoloxPostProcessor.rows(640))
        assertEquals(3549, YoloxPostProcessor.rows(416))
    }

    @Test
    fun decodesTheSmallerTinyInput() {
        val tiny = 416
        val out = FloatArray(YoloxPostProcessor.rows(tiny) * rowSize)
        // Stride 16 starts after the 52x52 stride-8 grid (2704 rows); cell (gx=4, gy=3) of the 26x26 grid: centre (64, 48), size 16 x 16.
        out.setRow(2704 + 3 * 26 + 4, 0f, 0f, 0f, 0f, obj = 0.9f, classId = 5, cls = 0.9f)
        val d = YoloxPostProcessor.decode(out, tiny, Letterbox(416, 416, tiny)).single()
        assertEquals(5, d.classId)
        assertEquals(56f / 416, d.left, 1e-5f)
        assertEquals(40f / 416, d.top, 1e-5f)
        assertEquals(72f / 416, d.right, 1e-5f)
        assertEquals(56f / 416, d.bottom, 1e-5f)
    }
}
