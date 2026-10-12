package es.jrg.objectdetection

/** Keeps the processing time of the last [capacity] frames, oldest first. */
class FrameTimeHistory(private val capacity: Int = 50) {
    private val times = ArrayDeque<Long>()

    fun add(ms: Long) {
        if (times.size == capacity) times.removeFirst()
        times.addLast(ms)
    }

    fun snapshot(): List<Long> = times.toList()

    fun clear() = times.clear()

    companion object {
        private val STEPS = longArrayOf(1, 2, 4, 6, 8, 10)

        /** Smallest "round" value (1, 2, 4, 6, 8 or 10 times a power of ten) that is at least [maxMs]; the plot's top tick. */
        fun niceMax(maxMs: Long): Long {
            var scale = 1L
            while (true) {
                for (step in STEPS) if (step * scale >= maxMs) return step * scale
                scale *= 10
            }
        }
    }
}
