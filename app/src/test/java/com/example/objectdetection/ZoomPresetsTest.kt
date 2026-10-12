package com.example.objectdetection

import org.junit.Assert.assertEquals
import org.junit.Test

class ZoomPresetsTest {
    @Test
    fun ultrawideAndDigitalRange() {
        // The Poco X7 Pro's back camera reports 0.6..10.
        assertEquals(listOf(0.6f, 1f, 2f, 5f, 10f), ZoomPresets.forRange(0.60000002f, 10f))
    }

    @Test
    fun smallerMaximumDropsHigherStops() {
        assertEquals(listOf(1f, 2f), ZoomPresets.forRange(1f, 4f))
        assertEquals(listOf(0.5f, 1f, 2f, 5f), ZoomPresets.forRange(0.5f, 5f))
    }

    @Test
    fun ladderStopsAtTenEvenForHugeRanges() {
        assertEquals(listOf(1f, 2f, 5f, 10f), ZoomPresets.forRange(1f, 100f))
    }

    @Test
    fun noChoiceMeansNoStops() {
        assertEquals(emptyList<Float>(), ZoomPresets.forRange(1f, 1f))
        assertEquals(emptyList<Float>(), ZoomPresets.forRange(1f, 1.5f))
    }

    @Test
    fun positionInterpolatesBetweenEvenlySpacedStops() {
        val stops = listOf(0.6f, 1f, 2f, 5f, 10f)
        assertEquals(0f, ZoomPresets.position(stops, 0.6f), 1e-6f)
        assertEquals(0.25f, ZoomPresets.position(stops, 1f), 1e-6f)
        assertEquals(0.375f, ZoomPresets.position(stops, 1.5f), 1e-6f)
        assertEquals(0.5f, ZoomPresets.position(stops, 2f), 1e-6f)
        assertEquals(1f, ZoomPresets.position(stops, 10f), 1e-6f)
        assertEquals(1f, ZoomPresets.position(stops, 50f), 1e-6f)
        assertEquals(0f, ZoomPresets.position(stops, 0.3f), 1e-6f)
    }

    @Test
    fun ratioAtInvertsPosition() {
        val stops = listOf(0.6f, 1f, 2f, 5f, 10f)
        for (r in listOf(0.6f, 0.8f, 1f, 1.5f, 3f, 7.5f, 10f)) {
            assertEquals(r, ZoomPresets.ratioAt(stops, ZoomPresets.position(stops, r)), 1e-4f)
        }
    }

    @Test
    fun nearestStopSnapsToTheClosestPoint() {
        val stops = listOf(0.6f, 1f, 2f, 5f, 10f)
        assertEquals(1f, ZoomPresets.nearestStop(stops, 0.3f), 0f)
        assertEquals(5f, ZoomPresets.nearestStop(stops, 0.7f), 0f)
        assertEquals(10f, ZoomPresets.nearestStop(stops, 1f), 0f)
    }

    @Test
    fun labels() {
        assertEquals("0.6x", ZoomPresets.label(0.6f))
        assertEquals("1x", ZoomPresets.label(1f))
        assertEquals("10x", ZoomPresets.label(10f))
    }

    @Test
    fun modelNames() {
        assertEquals("YOLOX-S", modelDisplayName("yolox_s"))
        assertEquals("YOLOX-Tiny", modelDisplayName("yolox_tiny"))
        assertEquals("rtdetr", modelDisplayName("rtdetr"))
        assertEquals(listOf("yolox_s", "yolox_tiny"), MODELS)
    }
}
