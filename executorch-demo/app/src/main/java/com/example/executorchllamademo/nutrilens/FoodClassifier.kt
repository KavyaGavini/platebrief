package com.example.executorchllamademo.nutrilens

import android.content.Context
import android.graphics.Bitmap
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Locale
import kotlin.math.exp

data class FoodPrediction(
    val label: String,
    val probability: Double
)

class FoodClassifier(private val context: Context) {

    private var model: Module? = null
    private var labels: List<String> = emptyList()
    private var inputSize = 128

    @Synchronized
    fun load() {
        if (model != null) return

        val hasPlateBrief = context.assets.list("platebrief")?.contains("platebrief_food.pte") == true
        val assetName = if (hasPlateBrief) "platebrief/platebrief_food.pte" else "food_cnn.pte"
        val modelFile = File(context.filesDir, assetName.substringAfterLast('/'))
        val temporary = File.createTempFile("food-classifier", ".pte", context.filesDir)
        try {
            context.assets.open(assetName).use { input ->
                temporary.outputStream().use { output -> input.copyTo(output) }
            }
            check(temporary.renameTo(modelFile)) { "Could not stage the food classifier" }
        } finally {
            temporary.delete()
        }

        val loadedLabels = if (hasPlateBrief) {
            val labelsJson = context.assets.open("platebrief/food_labels.json")
                .bufferedReader().use { JSONArray(it.readText()) }
            val metadata = context.assets.open("platebrief/metadata.json")
                .bufferedReader().use { JSONObject(it.readText()) }
            val shape = metadata.getJSONArray("input_shape")
            require(shape.length() == 4 && shape.getInt(0) == 1 && shape.getInt(1) == 3)
            inputSize = shape.getInt(2)
            require(inputSize in 32..512 && shape.getInt(3) == inputSize)
            List(labelsJson.length()) { labelsJson.getString(it) }
        } else {
            context.assets.open("food_labels.txt")
                .bufferedReader().use { it.readLines().filter(String::isNotBlank) }
        }
        require(loadedLabels.size >= 5 && loadedLabels.distinct().size == loadedLabels.size)

        val loadedModel = Module.load(modelFile.absolutePath)

        labels = loadedLabels
        model = loadedModel
    }

    @Synchronized
    fun classify(bitmap: Bitmap): List<FoodPrediction> {
        load()

        val loadedModel = requireNotNull(model)

        val resized = Bitmap.createScaledBitmap(
            bitmap, inputSize, inputSize, true
        )

        val pixels = IntArray(inputSize * inputSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)

        val input = FloatArray(3 * inputSize * inputSize)

        for (i in pixels.indices) {
            val pixel = pixels[i]

            val rgb = floatArrayOf(
                ((pixel shr 16) and 255) / 255f,
                ((pixel shr 8) and 255) / 255f,
                (pixel and 255) / 255f
            )

            for (c in 0..2) {
                input[c * inputSize * inputSize + i] =
                    (rgb[c] - mean[c]) / std[c]
            }
        }

        val tensor = Tensor.fromBlob(
            input, longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        )

        val startedAt = android.os.SystemClock.elapsedRealtimeNanos()
        val scores = loadedModel.forward(
            EValue.from(tensor)
        )[0].toTensor().dataAsFloatArray

        val inferenceMs = (android.os.SystemClock.elapsedRealtimeNanos() - startedAt) / 1_000_000.0
        android.util.Log.d("PlateBriefVision", "forward_ms=$inferenceMs")

        require(scores.size == labels.size) {
            "Model returned ${scores.size} scores"
        }

        val maxScore = scores.maxOrNull() ?: error("Empty model output")
        val exps = scores.map {
            exp((it - maxScore).toDouble())
        }
        val total = exps.sum()

        require(total > 0 && total.isFinite()) {
            "Invalid prediction probabilities"
        }

        return scores.indices
            .sortedByDescending { scores[it] }
            .take(5)
            .map { index ->
                FoodPrediction(
                    label = labels[index],
                    probability = exps[index] / total * 100.0
                )
            }
    }

    fun format(predictions: List<FoodPrediction>): String {
        return buildString {
            appendLine("Top 5 Food Predictions")
            appendLine()

            predictions.forEachIndexed { index, prediction ->
                val name = prediction.label
                    .replace(Regex("[_-]+"), " ")
                    .replaceFirstChar { it.titlecase(Locale.US) }

                appendLine(
                    "${index + 1}. $name: " +
                    String.format(
                        Locale.US,
                        "%.2f%%",
                        prediction.probability
                    )
                )
            }

            appendLine()
            append("Please confirm which food is correct.")
        }
    }
}
