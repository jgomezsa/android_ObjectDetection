package es.jrg.objectdetection

import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/**
 * Downloads a model file (or a small text file) over https. Redirects are followed by [HttpURLConnection] (https to https only),
 * the size is capped, and the user can cancel. Failures are reported as [ModelRejectedException] with a message for the user.
 */
class ModelDownloader(
    private val maxModelBytes: Long = MAX_MODEL_BYTES,
    private val connectTimeoutMs: Int = 15_000,
    private val readTimeoutMs: Int = 30_000,
    private val allowHttp: Boolean = false,
) {
    /** Writes the response to [target]; [onProgress] gets the bytes so far and the total (-1 when unknown). */
    fun download(url: String, target: File, onProgress: (Long, Long) -> Unit, isCancelled: () -> Boolean) {
        val connection = open(url)
        try {
            val total = connection.contentLengthLong
            if (total > maxModelBytes) throw ModelRejectedException("The file is ${total / MB} MB; the limit is ${maxModelBytes / MB} MB.")
            var done = 0L
            connection.inputStream.use { input ->
                target.outputStream().use { output ->
                    val buffer = ByteArray(64 * 1024)
                    while (true) {
                        if (isCancelled()) throw ModelRejectedException("Cancelled.")
                        val n = input.read(buffer)
                        if (n < 0) break
                        done += n
                        if (done > maxModelBytes) throw ModelRejectedException("The file is larger than the limit of ${maxModelBytes / MB} MB.")
                        output.write(buffer, 0, n)
                        onProgress(done, total)
                    }
                }
            }
            if (done == 0L) throw ModelRejectedException("The download was empty.")
        } catch (e: ModelRejectedException) {
            target.delete()
            throw e
        } catch (e: IOException) {
            target.delete()
            throw ModelRejectedException("The download failed: ${e.message ?: e.javaClass.simpleName}")
        } finally {
            connection.disconnect()
        }
    }

    /** A small text file such as a list of class names. */
    fun downloadText(url: String): String {
        val connection = open(url)
        try {
            val bytes = connection.inputStream.use { input ->
                val out = java.io.ByteArrayOutputStream()
                val buffer = ByteArray(8 * 1024)
                while (true) {
                    val n = input.read(buffer)
                    if (n < 0) break
                    out.write(buffer, 0, n)
                    if (out.size() > MAX_TEXT_BYTES) throw ModelRejectedException("The labels file is larger than ${MAX_TEXT_BYTES / 1024} KB.")
                }
                out.toByteArray()
            }
            return String(bytes, Charsets.UTF_8)
        } catch (e: IOException) {
            throw ModelRejectedException("The labels download failed: ${e.message ?: e.javaClass.simpleName}")
        } finally {
            connection.disconnect()
        }
    }

    private fun open(url: String): HttpURLConnection {
        ModelUrl.parse(url, allowHttp)
        val connection = try {
            URL(url).openConnection() as HttpURLConnection
        } catch (e: IOException) {
            throw ModelRejectedException("Cannot connect to that address.")
        }
        connection.connectTimeout = connectTimeoutMs
        connection.readTimeout = readTimeoutMs
        connection.instanceFollowRedirects = true
        connection.setRequestProperty("User-Agent", "ObjectDetection-Android")
        try {
            val code = connection.responseCode
            if (code != HttpURLConnection.HTTP_OK) {
                connection.disconnect()
                throw ModelRejectedException(
                    when (code) {
                        401, 403 -> "The server refused the download (HTTP $code): the file may need a login."
                        404 -> "The file was not found (HTTP 404)."
                        else -> "The server answered HTTP $code."
                    },
                )
            }
        } catch (e: ModelRejectedException) {
            throw e
        } catch (e: IOException) {
            connection.disconnect()
            throw ModelRejectedException("Cannot reach that address: ${e.message ?: e.javaClass.simpleName}")
        }
        return connection
    }

    companion object {
        const val MB = 1024L * 1024L
        const val MAX_MODEL_BYTES = 400L * MB
        const val MAX_TEXT_BYTES = 1024 * 1024
    }
}
