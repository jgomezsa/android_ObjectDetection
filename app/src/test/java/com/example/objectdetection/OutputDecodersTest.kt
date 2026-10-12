package com.example.objectdetection

import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class OutputDecodersTest {
    private val anchors = 8400
    private val names = CocoLabels.names
    private val square = Letterbox(640, 640, 640)

    private fun assertBox(d: Detection, left: Float, top: Float, right: Float, bottom: Float, classId: Int, score: Float) {
        assertEquals("left", left, d.left, 1e-4f)
        assertEquals("top", top, d.top, 1e-4f)
        assertEquals("right", right, d.right, 1e-4f)
        assertEquals("bottom", bottom, d.bottom, 1e-4f)
        assertEquals(classId, d.classId)
        assertEquals(score, d.score, 1e-5f)
    }

    private fun channelsFirst(vararg boxes: FloatArray): FloatArray {
        // Each box: anchor, cx, cy, w, h, class, score.
        val out = FloatArray(84 * anchors)
        for (b in boxes) {
            val i = b[0].toInt()
            out[i] = b[1]
            out[anchors + i] = b[2]
            out[2 * anchors + i] = b[3]
            out[3 * anchors + i] = b[4]
            out[(4 + b[5].toInt()) * anchors + i] = b[6]
        }
        return out
    }

    @Test
    fun yolov8ChannelsFirst() {
        val spec = ModelSpec(OutputLayout.V8_CHANNELS_FIRST, 640, 80)
        val raw = channelsFirst(floatArrayOf(100f, 320f, 320f, 100f, 200f, 3f, 0.9f))
        val d = OutputDecoders.decode(spec, listOf(raw), square, names).single()
        assertBox(d, 270f / 640, 220f / 640, 370f / 640, 420f / 640, 3, 0.9f)
        assertEquals("motorcycle", d.label)
    }

    @Test
    fun yolov8NormalisedBoxesAreScaledToTheInput() {
        val spec = ModelSpec(OutputLayout.V8_CHANNELS_FIRST, 640, 80, normalizedBoxes = true)
        val raw = channelsFirst(floatArrayOf(100f, 0.5f, 0.5f, 0.25f, 0.5f, 3f, 0.9f))
        assertBox(OutputDecoders.decode(spec, listOf(raw), square, names).single(), 0.375f, 0.25f, 0.625f, 0.75f, 3, 0.9f)
    }

    @Test
    fun yolov8RowsGiveTheSameBoxes() {
        val cf = channelsFirst(floatArrayOf(7f, 100f, 200f, 50f, 60f, 0f, 0.8f))
        val rows = FloatArray(84 * anchors)
        for (c in 0 until 84) for (i in 0 until anchors) rows[i * 84 + c] = cf[c * anchors + i]
        val a = OutputDecoders.decode(ModelSpec(OutputLayout.V8_CHANNELS_FIRST, 640, 80), listOf(cf), square, names)
        val b = OutputDecoders.decode(ModelSpec(OutputLayout.V8_ROWS, 640, 80), listOf(rows), square, names)
        assertEquals(a, b)
        assertBox(b.single(), 75f / 640, 170f / 640, 125f / 640, 230f / 640, 0, 0.8f)
    }

    @Test
    fun yolov8DropsLowScoresAndMergesOverlaps() {
        val spec = ModelSpec(OutputLayout.V8_CHANNELS_FIRST, 640, 80)
        val raw = channelsFirst(
            floatArrayOf(1f, 100f, 100f, 80f, 80f, 2f, 0.9f),
            floatArrayOf(2f, 102f, 101f, 80f, 80f, 2f, 0.7f), // same class, overlaps -> removed
            floatArrayOf(3f, 102f, 101f, 80f, 80f, 5f, 0.6f), // other class -> kept
            floatArrayOf(4f, 400f, 400f, 40f, 40f, 2f, 0.2f), // below the threshold
        )
        val out = OutputDecoders.decode(spec, listOf(raw), square, names)
        assertEquals(listOf(0.9f, 0.6f), out.map { it.score })
    }

    @Test
    fun yolov8BoxesAreMappedBackThroughTheLetterbox() {
        val lb = Letterbox(1280, 640, 640) // scale 0.5, padY 160
        val spec = ModelSpec(OutputLayout.V8_CHANNELS_FIRST, 640, 80)
        val raw = channelsFirst(floatArrayOf(5f, 480f, 320f, 320f, 320f, 2f, 0.9f)) // the right half of the source
        val d = OutputDecoders.decode(spec, listOf(raw), lb, names).single()
        assertBox(d, 0.5f, 0f, 1f, 1f, 2, 0.9f)
    }

    @Test
    fun detrSeparateOutputsUseSigmoidScoresAndStretchedCoordinates() {
        val spec = ModelSpec(OutputLayout.DETR_SPLIT, 640, 80, 300, 0, 1)
        val logits = FloatArray(300 * 80) { -10f }
        val boxes = FloatArray(300 * 4) { 0.5f }
        logits[5 * 80 + 7] = 2f // sigmoid = 0.8808
        boxes[5 * 4 + 0] = 0.5f
        boxes[5 * 4 + 1] = 0.5f
        boxes[5 * 4 + 2] = 0.2f
        boxes[5 * 4 + 3] = 0.4f
        logits[6 * 80 + 1] = -3f // sigmoid = 0.047: dropped
        // Stretched, so normalised coordinates are the same on the source image whatever its shape.
        val lb = Letterbox(1280, 960, 640, stretch = true)
        val d = OutputDecoders.decode(spec, listOf(logits, boxes), lb, names).single()
        assertBox(d, 0.4f, 0.3f, 0.6f, 0.7f, 7, 0.88079f)
        // The same model with the outputs in the other order.
        val swapped = ModelSpec(OutputLayout.DETR_SPLIT, 640, 80, 300, 1, 0)
        assertEquals(d, OutputDecoders.decode(swapped, listOf(boxes, logits), lb, names).single())
    }

    @Test
    fun detrMergedOutput() {
        val spec = ModelSpec(OutputLayout.DETR_MERGED, 640, 80, 300)
        val raw = FloatArray(300 * 84)
        raw[10 * 84 + 0] = 0.5f
        raw[10 * 84 + 1] = 0.5f
        raw[10 * 84 + 2] = 0.5f
        raw[10 * 84 + 3] = 0.5f
        raw[10 * 84 + 4 + 2] = 0.8f
        val d = OutputDecoders.decode(spec, listOf(raw), Letterbox(640, 640, 640, stretch = true), names).single()
        assertBox(d, 0.25f, 0.25f, 0.75f, 0.75f, 2, 0.8f)
    }

    @Test
    fun nmsFreeOutputInPixelsOrFractions() {
        val spec = ModelSpec(OutputLayout.END2END, 640, 80, 300)
        fun rows(scale: Float) = FloatArray(300 * 6).also {
            it[0] = 64f * scale
            it[1] = 128f * scale
            it[2] = 320f * scale
            it[3] = 256f * scale
            it[4] = 0.9f
            it[5] = 17f
            it[10] = 0.1f // a second row with a low score: dropped
        }
        val pixels = OutputDecoders.decode(spec, listOf(rows(1f)), square, names).single()
        assertBox(pixels, 0.1f, 0.2f, 0.5f, 0.4f, 17, 0.9f)
        assertEquals(pixels, OutputDecoders.decode(spec, listOf(rows(1f / 640)), square, names).single())
    }

    @Test
    fun thresholdControlsWhichDetectionsAreKept() {
        val spec = ModelSpec(OutputLayout.V8_CHANNELS_FIRST, 640, 80)
        val raw = channelsFirst(
            floatArrayOf(1f, 100f, 100f, 40f, 40f, 2f, 0.9f),
            floatArrayOf(2f, 300f, 300f, 40f, 40f, 3f, 0.4f),
        )
        assertEquals(2, OutputDecoders.decode(spec, listOf(raw), square, names).size)
        assertEquals(2, OutputDecoders.decode(spec, listOf(raw), square, names, 0.3f).size)
        assertEquals(listOf(0.9f), OutputDecoders.decode(spec, listOf(raw), square, names, 0.5f).map { it.score })
        assertEquals(0, OutputDecoders.decode(spec, listOf(raw), square, names, 0.95f).size)
    }

    @Test
    fun namesFallBackToClassNumbers() {
        val spec = ModelSpec(OutputLayout.V8_CHANNELS_FIRST, 640, 80)
        val raw = channelsFirst(floatArrayOf(1f, 100f, 100f, 40f, 40f, 4f, 0.9f))
        assertEquals("plate", OutputDecoders.decode(spec, listOf(raw), square, List(80) { "plate" }).single().label)
        assertEquals("class 4", OutputDecoders.decode(spec, listOf(raw), square, emptyList()).single().label)
    }

    @Test
    fun inputPackingFollowsTheConvention() {
        val pixel = 0xFF102030.toInt() // r = 16, g = 32, b = 48
        val nhwc = FloatArray(6)
        InputPacker.nhwc(intArrayOf(pixel, pixel), InputConvention.BGR_255, nhwc)
        assertArrayEquals(floatArrayOf(48f, 32f, 16f, 48f, 32f, 16f), nhwc, 0f)
        InputPacker.nhwc(intArrayOf(pixel, pixel), InputConvention.RGB_01, nhwc)
        assertArrayEquals(floatArrayOf(16 / 255f, 32 / 255f, 48 / 255f, 16 / 255f, 32 / 255f, 48 / 255f), nhwc, 1e-6f)

        val buffer = ByteBuffer.allocateDirect(4 * 6).order(ByteOrder.nativeOrder()).asFloatBuffer()
        InputPacker.nchw(intArrayOf(pixel, pixel), InputConvention.BGR_255, buffer)
        val planes = FloatArray(6).also { buffer.rewind(); buffer.get(it) }
        // Planar: the blue plane, then green, then red.
        assertArrayEquals(floatArrayOf(48f, 48f, 32f, 32f, 16f, 16f), planes, 0f)
    }
}
