package com.example.executorchllamademo.nutrilens

import android.content.Context
import org.json.JSONObject
import java.util.Locale

data class FoodNutrition(
    val foodName: String,
    val grams: Double,
    val calories: Double,
    val protein: Double,
    val carbs: Double,
    val fat: Double,
    val source: String
)

class NutritionCalculator(context: Context) {

    private val database: JSONObject = context.assets
        .open("food_nutrition.json")
        .bufferedReader()
        .use { JSONObject(it.readText()) }

    fun calculate(foodName: String, grams: Double): FoodNutrition? {
        if (!grams.isFinite() || grams <= 0.0 || grams > 5000.0) {
            return null
        }

        val key = foodName.trim()
            .lowercase(Locale.ROOT)
            .replace(Regex("[\\s-]+"), "_")

        val catalogKey = when (key) {
            "chips_french_fries" -> "french_fries"
            "dairy_ice_cream" -> "ice_cream"
            else -> key
        }
        val entry = database.optJSONObject(catalogKey) ?: return null

        if (entry.optString("unit") != "per_100g") return null

        val nutrients = listOf("calories", "protein", "carbs", "fat")

        if (nutrients.any { entry.opt(it) !is Number }) {
            return null
        }

        val factor = grams / 100.0

        return FoodNutrition(
            foodName = key.replace("_", " "),
            grams = grams,
            calories = entry.getDouble("calories") * factor,
            protein = entry.getDouble("protein") * factor,
            carbs = entry.getDouble("carbs") * factor,
            fat = entry.getDouble("fat") * factor,
            source = entry.optString("source", "Unknown")
        )
    }

    companion object {
        private val caloriesQuestion = Regex(
            """^how many (?:kcal|calories) (?:are )?in (\d+(?:\.\d+)?)\s*(?:g|grams?) if (\d+(?:\.\d+)?)\s*(?:g|grams?) (?:has|have|contains?) (\d+(?:\.\d+)?)\s*(?:kcal|calories)[?.]?$""",
            RegexOption.IGNORE_CASE
        )

        fun scaleStatedCalories(question: String): String? {
            val match = caloriesQuestion.matchEntire(question.trim()) ?: return null
            val (targetText, referenceText, energyText) = match.destructured
            val target = targetText.toDouble()
            val reference = referenceText.toDouble()
            val energy = energyText.toDouble()
            if (!target.isFinite() || !reference.isFinite() || !energy.isFinite() || reference <= 0) {
                return "Enter a positive reference weight and finite calorie values."
            }
            val result = target / reference * energy
            if (!result.isFinite()) return "The given values are too large to scale."
            return String.format(Locale.US, "%s g has %.1f kcal using your stated ratio.", targetText, result)
        }
    }

    fun format(result: FoodNutrition): String {
        fun number(value: Double) =
            String.format(Locale.US, "%.1f", value)

        return """
            Food: ${result.foodName}
            Serving: ${number(result.grams)} g
            Calories: ${number(result.calories)} kcal
            Protein: ${number(result.protein)} g
            Carbs: ${number(result.carbs)} g
            Fat: ${number(result.fat)} g
            Source: ${result.source}
        """.trimIndent()
    }
}
