package com.example.objectdetection

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class ModelStoreTest {
    @get:Rule
    val folder = TemporaryFolder()

    private fun download(content: String = "model bytes") = folder.newFile().also { it.writeText(content) }

    @Test
    fun startsWithOnlyTheBuiltInModels() {
        val store = ModelStore(File(folder.root, "models"))
        assertEquals(emptyList<ModelInfo>(), store.userModels())
        assertEquals(listOf("yolox_s", "yolox_tiny"), store.all().map { it.id })
    }

    @Test
    fun installedModelsAreListedWithTheirDetails() {
        val store = ModelStore(File(folder.root, "models"))
        val info = store.install(download("12345"), ModelFormat.ONNX, "Plates S", 640, 1, listOf("license_plate"), "https://example.com/p.onnx")
        assertEquals("plates_s", info.id)
        assertEquals("Plates S", info.displayName)
        assertEquals(ModelFormat.ONNX, info.format)
        assertFalse(info.builtIn)
        assertEquals(5L, info.sizeBytes)
        assertEquals(640, info.inputSize)
        assertEquals(1, info.numClasses)
        assertEquals(listOf("license_plate"), info.labels)
        assertEquals("https://example.com/p.onnx", info.sourceUrl)
        assertTrue(info.file!!.isFile)

        // A new store instance (a new app start) sees it too.
        val again = ModelStore(File(folder.root, "models"))
        assertEquals(listOf("yolox_s", "yolox_tiny", "plates_s"), again.all().map { it.id })
        assertEquals(info, again.find("plates_s"))
    }

    @Test
    fun modelsWithoutLabelsGetGenericOrCocoNames() {
        val store = ModelStore(File(folder.root, "models"))
        val three = store.install(download(), ModelFormat.TFLITE, "three", 416, 3, emptyList(), null)
        assertEquals(listOf("class 0", "class 1", "class 2"), three.labels)
        val coco = store.install(download(), ModelFormat.TFLITE, "coco like", 640, 80, emptyList(), null)
        assertEquals(CocoLabels.names, coco.labels)
    }

    @Test
    fun idsAndNamesStayUnique() {
        val store = ModelStore(File(folder.root, "models"))
        val first = store.install(download(), ModelFormat.TFLITE, "My Model", 640, 80, emptyList(), null)
        val second = store.install(download(), ModelFormat.TFLITE, "My Model", 640, 80, emptyList(), null)
        assertEquals("my_model", first.id)
        assertEquals("my_model_2", second.id)
        assertEquals("My Model", first.displayName)
        assertEquals("My Model (2)", second.displayName)
        // A name that collides with a built-in one is made unique as well.
        val clash = store.install(download(), ModelFormat.TFLITE, "YOLOX-S", 640, 80, emptyList(), null)
        assertEquals("yolox_s_2", clash.id)
        assertEquals("YOLOX-S (2)", clash.displayName)
        assertEquals("YOLOX-S (2)", ModelStore(File(folder.root, "models")).find("yolox_s_2")!!.displayName)
    }

    @Test
    fun deleteRemovesTheFolderButNeverBuiltInModels() {
        val store = ModelStore(File(folder.root, "models"))
        val info = store.install(download(), ModelFormat.TFLITE, "temp", 640, 80, emptyList(), null)
        val dir = info.file!!.parentFile!!
        assertTrue(store.delete("temp"))
        assertFalse(dir.exists())
        assertNull(store.find("temp"))
        assertFalse(store.delete("temp"))
        assertFalse(store.delete("yolox_s"))
        assertNotNull(store.find("yolox_s"))
        // Ids that try to leave the store folder are refused.
        val outside = folder.newFolder("outside")
        assertFalse(store.delete("../outside"))
        assertTrue(outside.exists())
    }

    @Test
    fun damagedFoldersAreIgnored() {
        val root = File(folder.root, "models")
        File(root, "broken").mkdirs()
        File(root, "half").mkdirs()
        File(root, "half/model.properties").writeText("format=TFLITE\nclasses=80\ninput=640\n")
        assertEquals(listOf("yolox_s", "yolox_tiny"), ModelStore(root).all().map { it.id })
    }

    @Test
    fun slugsAreSafeFileNames() {
        assertEquals("yolox_plates_s", ModelStore.slug("yolox_plates_s"))
        assertEquals("my_model_v2", ModelStore.slug("My Model (v2)"))
        assertEquals("model", ModelStore.slug("***"))
        assertEquals(40, ModelStore.slug("a".repeat(100)).length)
    }
}
