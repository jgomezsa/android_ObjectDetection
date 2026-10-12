package com.example.objectdetection

import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStream
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import kotlin.concurrent.thread
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

/** Runs the downloader against a tiny local HTTP server (plain http is allowed only in these tests). */
class ModelDownloaderTest {
    @get:Rule
    val folder = TemporaryFolder()

    private lateinit var server: ServerSocket
    private val payload = ByteArray(200_000) { (it % 251).toByte() }
    private val base get() = "http://localhost:${server.localPort}"

    @Before
    fun startServer() {
        server = ServerSocket(0, 20, InetAddress.getLoopbackAddress())
        thread(isDaemon = true) {
            while (!server.isClosed) {
                val socket = try {
                    server.accept()
                } catch (e: Exception) {
                    break
                }
                thread(isDaemon = true) { handle(socket) }
            }
        }
    }

    @After
    fun stopServer() = server.close()

    private fun handle(socket: Socket) {
        socket.use {
            val reader = BufferedReader(InputStreamReader(socket.getInputStream()))
            val path = reader.readLine()?.split(" ")?.getOrNull(1) ?: return
            while (reader.readLine()?.isNotEmpty() == true) { /* skip the request headers */ }
            val out = socket.getOutputStream()
            when (path) {
                "/model.tflite" -> respond(out, 200, "OK", payload)
                "/redirect" -> respond(out, 302, "Found", ByteArray(0), "Location: $base/model.tflite\r\n")
                "/missing" -> respond(out, 404, "Not Found", ByteArray(0))
                "/private" -> respond(out, 403, "Forbidden", ByteArray(0))
                "/empty" -> respond(out, 200, "OK", ByteArray(0))
                "/labels.txt" -> respond(out, 200, "OK", "cat\ndog\n".toByteArray())
                else -> respond(out, 404, "Not Found", ByteArray(0))
            }
        }
    }

    private fun respond(out: OutputStream, code: Int, text: String, body: ByteArray, extraHeaders: String = "") {
        out.write("HTTP/1.1 $code $text\r\nContent-Length: ${body.size}\r\n${extraHeaders}Connection: close\r\n\r\n".toByteArray())
        out.write(body)
        out.flush()
    }

    private fun downloader(maxBytes: Long = 10_000_000) = ModelDownloader(maxModelBytes = maxBytes, allowHttp = true)

    private fun rejected(block: () -> Unit): String {
        try {
            block()
        } catch (e: ModelRejectedException) {
            return e.message!!
        }
        throw AssertionError("expected a rejection")
    }

    @Test
    fun downloadsTheFileAndReportsProgress() {
        val target = folder.newFile()
        val progress = ArrayList<Pair<Long, Long>>()
        downloader().download("$base/model.tflite", target, { done, total -> progress += done to total }, { false })
        assertArrayEquals(payload, target.readBytes())
        assertEquals(payload.size.toLong(), progress.last().first)
        assertEquals(payload.size.toLong(), progress.last().second)
    }

    @Test
    fun followsRedirects() {
        val target = folder.newFile()
        downloader().download("$base/redirect", target, { _, _ -> }, { false })
        assertEquals(payload.size.toLong(), target.length())
    }

    @Test
    fun reportsHttpErrorsInPlainWords() {
        val target = folder.newFile()
        assertTrue(rejected { downloader().download("$base/missing", target, { _, _ -> }, { false }) }.contains("not found"))
        assertTrue(rejected { downloader().download("$base/private", target, { _, _ -> }, { false }) }.contains("login"))
    }

    @Test
    fun refusesFilesOverTheLimitAndRemovesPartialFiles() {
        val target = folder.newFile()
        assertTrue(rejected { downloader(maxBytes = 1000).download("$base/model.tflite", target, { _, _ -> }, { false }) }.contains("limit"))
        assertFalse(target.exists())
    }

    @Test
    fun canBeCancelled() {
        val target = folder.newFile()
        assertTrue(rejected { downloader().download("$base/model.tflite", target, { _, _ -> }, { true }) }.contains("Cancelled"))
        assertFalse(target.exists())
    }

    @Test
    fun emptyDownloadsAreRejected() {
        val target = folder.newFile()
        assertTrue(rejected { downloader().download("$base/empty", target, { _, _ -> }, { false }) }.contains("empty"))
    }

    @Test
    fun plainHttpIsRefusedByDefault() {
        val target = folder.newFile()
        assertTrue(rejected { ModelDownloader().download("$base/model.tflite", target, { _, _ -> }, { false }) }.contains("https"))
    }

    @Test
    fun downloadsSmallTextFiles() {
        assertEquals(listOf("cat", "dog"), LabelParser.parse(downloader().downloadText("$base/labels.txt")))
    }
}
