package com.example.objectdetection

import org.junit.Assert.assertEquals
import org.junit.Test

class YoloPostProcessorTest {
    @Test
    fun letterbox_wideImage_padsVertically() {
        val lb = Letterbox(1280, 640, 640)
        assertEquals(0.5f, lb.scale, 1e-6f)
        assertEquals(0f, lb.padX, 1e-6f)
        assertEquals(160f, lb.padY, 1e-6f)
    }

    @Test
    fun stretch_resizesToTheWholeSquareWithoutPadding() {
        val lb = Letterbox(1280, 640, 640, stretch = true)
        assertEquals(0.5f, lb.scaleX, 1e-6f)
        assertEquals(1f, lb.scaleY, 1e-6f)
        assertEquals(0f, lb.padX, 1e-6f)
        assertEquals(0f, lb.padY, 1e-6f)
        // The right half of the source is model x 320..640 over the full height.
        val d = YoloPostProcessor.toDetection(320f, 0f, 640f, 640f, 1, 0.5f, lb)!!
        assertEquals(0.5f, d.left, 1e-5f)
        assertEquals(1f, d.right, 1e-5f)
        assertEquals(0f, d.top, 1e-5f)
        assertEquals(1f, d.bottom, 1e-5f)
    }

    @Test
    fun toDetection_mapsModelSpaceBackToNormalizedSourceCoordinates() {
        val lb = Letterbox(1280, 640, 640) // scale 0.5, padY 160
        // The right half of the source: model x 320..640, y 160..480.
        val d = YoloPostProcessor.toDetection(320f, 160f, 640f, 480f, 2, 0.9f, lb)!!
        assertEquals(2, d.classId)
        assertEquals("car", d.label)
        assertEquals(0.5f, d.left, 1e-5f)
        assertEquals(1f, d.right, 1e-5f)
        assertEquals(0f, d.top, 1e-5f)
        assertEquals(1f, d.bottom, 1e-5f)
    }

    @Test
    fun toDetection_dropsBoxesOutsideTheImage() {
        val lb = Letterbox(1280, 640, 640)
        // Entirely inside the top padding.
        assertEquals(null, YoloPostProcessor.toDetection(0f, 0f, 100f, 100f, 0, 0.9f, lb))
    }

    @Test
    fun nms_suppressesOverlappingSameClassOnly() {
        val kept = YoloPostProcessor.nms(
            listOf(
                Detection(0.10f, 0.10f, 0.30f, 0.30f, 0, 0.9f),
                Detection(0.11f, 0.11f, 0.31f, 0.31f, 0, 0.8f), // same class, overlaps -> removed
                Detection(0.11f, 0.11f, 0.31f, 0.31f, 1, 0.7f), // other class -> kept
                Detection(0.70f, 0.70f, 0.80f, 0.80f, 0, 0.6f), // far away -> kept
            ),
            0.45f,
        )
        assertEquals(listOf(0.9f, 0.7f, 0.6f), kept.map { it.score })
    }

    @Test
    fun iou_ofIdenticalBoxesIsOne_andDisjointIsZero() {
        val a = Detection(0f, 0f, 0.5f, 0.5f, 0, 1f)
        val b = Detection(0.6f, 0.6f, 1f, 1f, 0, 1f)
        assertEquals(1f, YoloPostProcessor.iou(a, a), 1e-6f)
        assertEquals(0f, YoloPostProcessor.iou(a, b), 1e-6f)
    }
}
