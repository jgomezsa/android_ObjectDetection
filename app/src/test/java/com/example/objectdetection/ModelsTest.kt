package com.example.objectdetection

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelsTest {
    private fun bytes(vararg values: Int) = ByteArray(values.size) { values[it].toByte() }

    private fun padded(prefix: ByteArray, size: Int = 512) = prefix + ByteArray(size - prefix.size)

    @Test
    fun builtInModels() {
        assertEquals(listOf("yolox_s", "yolox_tiny"), BuiltInModels.all.map { it.id })
        assertEquals(listOf("YOLOX-S", "YOLOX-Tiny"), BuiltInModels.all.map { it.displayName })
        assertTrue(BuiltInModels.all.all { it.builtIn && it.numClasses == 80 && it.labels.size == 80 })
        assertEquals("yolox_s", DEFAULT_MODEL)
    }

    @Test
    fun detectsTfliteByItsMagicBytes() {
        // TFLite is a FlatBuffer: a 4-byte offset, then the file identifier "TFL3".
        val header = padded(bytes(0x1C, 0, 0, 0, 'T'.code, 'F'.code, 'L'.code, '3'.code))
        assertEquals(ModelFormat.TFLITE, ModelFormat.detect(header, "whatever"))
    }

    @Test
    fun detectsOnnxByTheProtobufStartOrTheExtension() {
        assertEquals(ModelFormat.ONNX, ModelFormat.detect(padded(bytes(0x08, 0x08, 0x12, 0x07)), "model"))
        assertEquals(ModelFormat.ONNX, ModelFormat.detect(padded(bytes(0x12, 0x07)), "model.onnx"))
        assertNull(ModelFormat.detect(padded(bytes(0x12, 0x07)), "model"))
    }

    @Test
    fun rejectsArchivesWebPagesAndUnknownFiles() {
        val zip = padded(bytes('P'.code, 'K'.code, 3, 4))
        val gzip = padded(bytes(0x1F, 0x8B, 8))
        val html = padded("<!DOCTYPE html><html>".toByteArray())
        val json = padded("{\"error\": 1}".toByteArray())
        for (header in listOf(zip, gzip, html, json)) assertNull(ModelFormat.detect(header, "model.onnx"))
        assertTrue(ModelFormat.whyNot(zip).contains("archive"))
        assertTrue(ModelFormat.whyNot(gzip).contains("archive"))
        assertTrue(ModelFormat.whyNot(html).contains("web page"))
        assertTrue(ModelFormat.whyNot(padded(bytes(1, 2, 3))).contains("not a TFLite or ONNX"))
    }

    @Test
    fun labelsAreParsedOnePerLine() {
        assertEquals(listOf("cat", "dog", "bird"), LabelParser.parse("﻿cat\r\n dog \n\nbird\n"))
        assertEquals(listOf("a", "b"), LabelParser.namesFor(2, listOf("a", "b")))
        assertEquals(CocoLabels.names, LabelParser.namesFor(80, emptyList()))
        assertEquals(listOf("class 0", "class 1", "class 2"), LabelParser.namesFor(3, emptyList()))
        // Names that do not match the class count are ignored.
        assertEquals(listOf("class 0", "class 1", "class 2"), LabelParser.namesFor(3, listOf("only", "two")))
    }

    @Test
    fun urlsMustBeHttps() {
        val parsed = ModelUrl.parse("  https://huggingface.co/autolane/yolox-s-alpr/resolve/main/yolox_plates_s.onnx?download=true ")
        assertEquals("yolox_plates_s.onnx", parsed.fileName)
        assertEquals("yolox_plates_s", ModelUrl.displayName(parsed.fileName))
        assertEquals("http", ModelUrl.parse("http://localhost:8080/a.tflite", allowHttp = true).url.substringBefore(':'))
        for (bad in listOf("", "   ", "http://example.com/a.onnx", "ftp://example.com/a.onnx", "not a url", "https:///a.onnx", "file:///sdcard/a.onnx")) {
            try {
                ModelUrl.parse(bad)
                throw AssertionError("should have been rejected: '$bad'")
            } catch (expected: ModelRejectedException) {
                assertTrue(expected.message!!.isNotEmpty())
            }
        }
    }

    @Test
    fun urlWithoutAFileNameGetsADefault() {
        val parsed = ModelUrl.parse("https://example.com/")
        assertEquals("model", parsed.fileName)
        assertEquals("model", ModelUrl.displayName("model"))
    }

    @Test
    fun chosenNameWinsOverTheFileName() {
        assertEquals("Potholes", ModelUrl.chooseName("  Potholes ", "best.tflite"))
        assertEquals("best", ModelUrl.chooseName("", "best.tflite"))
        assertEquals("best", ModelUrl.chooseName("   ", "best.tflite"))
        assertEquals(40, ModelUrl.chooseName("x".repeat(100), "best.tflite").length)
    }
}
