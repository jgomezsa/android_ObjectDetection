package com.example.objectdetection

import android.util.Log

/** Where a model runs. The GPU is always tried first; the CPU is the fallback when the model does not run on the GPU. */
enum class Backend(val label: String) {
    GPU("GPU"),
    CPU("CPU");

    val isGpu: Boolean get() = this == GPU
}

/** Tries backends in order until one can create (and sanity-run) a detector. */
object BackendChain {
    /** Default order: the GPU first, then the CPU. A new backend only has to be added here. */
    val DEFAULT_ORDER = listOf(Backend.GPU, Backend.CPU)

    /**
     * Returns the first successfully created value with the backend that produced it. [create] should throw when the backend
     * cannot run the model. Throws [IllegalStateException] (with every failure attached as suppressed) when none works.
     */
    fun <T> firstWorking(
        backends: List<Backend>,
        onFailure: (Backend, Throwable) -> Unit = { backend, error -> Log.w("Inference", "$backend backend failed", error) },
        create: (Backend) -> T,
    ): Pair<Backend, T> {
        val failures = ArrayList<Throwable>()
        for (backend in backends) {
            try {
                return backend to create(backend)
            } catch (e: Throwable) {
                failures += e
                onFailure(backend, e)
            }
        }
        throw IllegalStateException("No backend could run the model (${backends.joinToString { it.label }})").also { error ->
            failures.forEach(error::addSuppressed)
        }
    }
}
