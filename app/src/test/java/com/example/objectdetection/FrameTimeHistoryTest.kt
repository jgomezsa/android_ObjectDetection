package com.example.objectdetection

import org.junit.Assert.assertEquals
import org.junit.Test

class FrameTimeHistoryTest {
    @Test
    fun keepsOnlyTheLastNFramesOldestFirst() {
        val h = FrameTimeHistory(capacity = 3)
        (1L..5L).forEach(h::add)
        assertEquals(listOf(3L, 4L, 5L), h.snapshot())
    }

    @Test
    fun clearEmptiesTheHistory() {
        val h = FrameTimeHistory(capacity = 3)
        h.add(10)
        h.clear()
        assertEquals(emptyList<Long>(), h.snapshot())
    }

    @Test
    fun niceMaxRoundsUpToRoundValues() {
        assertEquals(1L, FrameTimeHistory.niceMax(0))
        assertEquals(80L, FrameTimeHistory.niceMax(63))
        assertEquals(100L, FrameTimeHistory.niceMax(97))
        assertEquals(100L, FrameTimeHistory.niceMax(100))
        assertEquals(200L, FrameTimeHistory.niceMax(101))
        assertEquals(600L, FrameTimeHistory.niceMax(450))
        assertEquals(1000L, FrameTimeHistory.niceMax(1000))
    }
}
