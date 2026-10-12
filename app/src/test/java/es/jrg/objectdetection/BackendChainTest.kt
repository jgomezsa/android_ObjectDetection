package es.jrg.objectdetection

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class BackendChainTest {
    private val quiet: (Backend, Throwable) -> Unit = { _, _ -> }

    @Test
    fun onlyTheGpuIsReportedAsGpu() {
        assertTrue(Backend.GPU.isGpu)
        assertFalse(Backend.CPU.isGpu)
    }

    @Test
    fun gpuIsTriedFirst() {
        assertEquals(listOf(Backend.GPU, Backend.CPU), BackendChain.DEFAULT_ORDER)
        val tried = ArrayList<Backend>()
        val (backend, value) = BackendChain.firstWorking(BackendChain.DEFAULT_ORDER, quiet) { tried += it; "model on $it" }
        assertEquals(Backend.GPU, backend)
        assertEquals("model on GPU", value)
        assertEquals(listOf(Backend.GPU), tried)
    }

    @Test
    fun fallsBackToTheCpuWhenTheGpuFails() {
        val failed = ArrayList<Backend>()
        val (backend, value) = BackendChain.firstWorking(BackendChain.DEFAULT_ORDER, { b, _ -> failed += b }) {
            if (it == Backend.GPU) error("no GPU") else "cpu model"
        }
        assertEquals(Backend.CPU, backend)
        assertEquals("cpu model", value)
        assertEquals(listOf(Backend.GPU), failed)
    }

    @Test
    fun throwsWithEveryFailureWhenNothingWorks() {
        try {
            BackendChain.firstWorking<String>(BackendChain.DEFAULT_ORDER, quiet) { error("$it broke") }
            fail("expected an exception")
        } catch (e: IllegalStateException) {
            assertEquals(listOf("GPU broke", "CPU broke"), e.suppressed.map { it.message })
        }
    }
}
