package es.jrg.objectdetection

import java.io.File
import java.util.Properties

/** Class names: one per line in a `.txt` file. */
object LabelParser {
    fun parse(text: String): List<String> = text.lineSequence().map { it.trim().trimStart('﻿') }.filter { it.isNotEmpty() }.toList()

    /** The names to use for a model with [numClasses] classes: the given [labels] when they match, COCO for 80 classes, otherwise "class N". */
    fun namesFor(numClasses: Int, labels: List<String>): List<String> = when {
        labels.size == numClasses -> labels
        numClasses == YoloPostProcessor.NUM_CLASSES -> CocoLabels.names
        else -> List(numClasses) { "class $it" }
    }

    /** As above for a model [spec]; an NMS-free output does not say how many classes there are, so any list of names is used as it is. */
    fun namesFor(spec: ModelSpec, labels: List<String>): List<String> =
        if (spec.layout == OutputLayout.END2END && labels.isNotEmpty()) labels else namesFor(spec.numClasses, labels)
}

/** An address a model can be downloaded from. */
object ModelUrl {
    data class Parsed(val url: String, val fileName: String)

    fun parse(text: String, allowHttp: Boolean = false): Parsed {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) throw ModelRejectedException("Enter the address of a model file.")
        val uri = try {
            java.net.URI(trimmed)
        } catch (e: Exception) {
            throw ModelRejectedException("That is not a valid web address.")
        }
        val scheme = uri.scheme?.lowercase()
        if (scheme != "https" && !(allowHttp && scheme == "http")) {
            throw ModelRejectedException("Only https:// addresses are accepted.")
        }
        if (uri.host.isNullOrEmpty()) throw ModelRejectedException("That address has no host name.")
        val lastSegment = uri.path.orEmpty().substringAfterLast('/').ifEmpty { "model" }
        return Parsed(trimmed, lastSegment)
    }

    /** The name to show for a model: what the user typed (trimmed, at most 40 characters), or the file name without its extension when that is blank. */
    fun chooseName(given: String, fileName: String): String = given.trim().take(40).trim().ifEmpty { displayName(fileName) }

    /** A readable model name from a file name: the file name without its extension. */
    fun displayName(fileName: String): String {
        val withoutExtension = fileName.removeSuffix(".tflite").removeSuffix(".onnx").removeSuffix(".TFLITE").removeSuffix(".ONNX")
        return withoutExtension.ifBlank { "model" }
    }
}

/**
 * Downloaded models, stored as `<root>/<id>/model.<ext>` with `model.properties` (and `labels.txt` when there are class names).
 * Built-in models are listed first and cannot be deleted.
 */
class ModelStore(private val root: File) {
    fun userModels(): List<ModelInfo> =
        (root.listFiles { file -> file.isDirectory }?.sortedBy { it.name } ?: emptyList()).mapNotNull { read(it) }

    fun all(): List<ModelInfo> = BuiltInModels.all + userModels()

    fun find(id: String): ModelInfo? = all().firstOrNull { it.id == id }

    /** Moves [downloaded] into the store and returns the new model. The display name and id are made unique. */
    fun install(
        downloaded: File,
        format: ModelFormat,
        displayName: String,
        spec: ModelSpec,
        labels: List<String>,
        sourceUrl: String?,
    ): ModelInfo {
        val existing = all()
        val id = uniqueId(slug(displayName), existing.map { it.id }.toSet())
        val name = uniqueName(displayName, existing.map { it.displayName }.toSet())
        val dir = File(root, id)
        check(dir.mkdirs() || dir.isDirectory) { "Cannot create $dir" }
        val target = File(dir, "model.${format.extension}")
        if (!downloaded.renameTo(target)) {
            downloaded.copyTo(target, overwrite = true)
            downloaded.delete()
        }
        if (labels.isNotEmpty()) File(dir, "labels.txt").writeText(labels.joinToString("\n"), Charsets.UTF_8)
        val props = Properties().apply {
            setProperty("name", name)
            setProperty("format", format.name)
            for ((key, value) in spec.toProperties()) setProperty(key, value)
            sourceUrl?.let { setProperty("url", it) }
        }
        File(dir, "model.properties").outputStream().use { props.store(it, null) }
        return read(dir) ?: error("The installed model could not be read back")
    }

    /** Deletes a downloaded model; returns false for built-in or unknown ids. */
    fun delete(id: String): Boolean {
        if (BuiltInModels.all.any { it.id == id }) return false
        val dir = File(root, id)
        return dir.isDirectory && dir.canonicalFile.parentFile == root.canonicalFile && dir.deleteRecursively()
    }

    private fun read(dir: File): ModelInfo? = try {
        val props = Properties().apply { File(dir, "model.properties").inputStream().use { load(it) } }
        val format = ModelFormat.valueOf(props.getProperty("format"))
        val file = File(dir, "model.${format.extension}")
        val spec = ModelSpec.fromProperties { props.getProperty(it) }
        val labelsFile = File(dir, "labels.txt")
        val labels = if (labelsFile.isFile) LabelParser.parse(labelsFile.readText(Charsets.UTF_8)) else emptyList()
        if (!file.isFile) {
            null
        } else {
            ModelInfo(
                id = dir.name,
                displayName = props.getProperty("name") ?: dir.name,
                format = format,
                builtIn = false,
                sizeBytes = file.length(),
                labels = LabelParser.namesFor(spec, labels),
                sourceUrl = props.getProperty("url"),
                file = file,
                spec = spec,
            )
        }
    } catch (e: Exception) {
        null // an incomplete or damaged folder is ignored
    }

    companion object {
        fun slug(name: String): String {
            val s = name.lowercase().map { if (it in 'a'..'z' || it in '0'..'9') it else '_' }.joinToString("").replace(Regex("_+"), "_").trim('_')
            return s.ifEmpty { "model" }.take(40)
        }

        fun uniqueId(base: String, taken: Set<String>): String {
            if (base !in taken) return base
            var n = 2
            while ("${base}_$n" in taken) n++
            return "${base}_$n"
        }

        fun uniqueName(base: String, taken: Set<String>): String {
            if (base !in taken) return base
            var n = 2
            while ("$base ($n)" in taken) n++
            return "$base ($n)"
        }
    }
}
