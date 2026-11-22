package com.example.executorchllamademo.nutrilens

import android.content.Context
import org.json.JSONObject
import java.util.Locale

class NutritionRepository(context: Context) {

    private val database: JSONObject =
        context.assets.open("food_nutrition.json")
            .bufferedReader()
            .use { JSONObject(it.readText()) }

    data class NutritionResult(
        val food: String,
        val grams: Double,
        val calories: Double,
        val protein: Double,
        val carbs: Double,
        val fat: Double
    )

    fun findNutrition(question: String): NutritionResult? {
        val servingMatch = Regex(
            """(?i)\b(\d+(?:\.\d+)?)\s*(?:g|grams?|grammes?)\b"""
        ).find(question) ?: return null

        val grams = servingMatch.groupValues[1].toDoubleOrNull()
            ?: return null

        if (!grams.isFinite() || grams <= 0 || grams > 5000) {
            return null
        }

        val lowerQuestion = question.lowercase(Locale.ROOT)

        val matchedFood = database.keys().asSequence()
            .filter { food ->
                val name = food.replace('_', ' ')
                    .lowercase(Locale.ROOT)

                val pattern = Regex(
                    """(?<![\p{L}\p{N}])""" +
                        Regex.escape(name).replace(
                            "\\ ",
                            """[\s_-]+"""
                        ) +
                        """(?![\p{L}\p{N}])"""
                )

                pattern.containsMatchIn(lowerQuestion)
            }
            .maxByOrNull { it.length }
            ?: return null

        val nutrition = database.optJSONObject(matchedFood)
            ?: return null

        val fields = listOf(
            "calories", "protein", "carbs", "fat"
        )

        if (fields.any {
                !nutrition.has(it) ||
                    nutrition.isNull(it) ||
                    nutrition.opt(it) !is Number
            }) {
            return null
        }

        val factor = grams / 100.0

        return NutritionResult(
            food = matchedFood.replace('_', ' '),
            grams = grams,
            calories = nutrition.getDouble("calories") * factor,
            protein = nutrition.getDouble("protein") * factor,
            carbs = nutrition.getDouble("carbs") * factor,
            fat = nutrition.getDouble("fat") * factor
        )
    }

    fun buildGroundedPrompt(
        question: String,
        result: NutritionResult
    ): String {
        fun fmt(value: Double): String =
            String.format(Locale.US, "%.1f", value)

        return """
            You are NutriLens V5, a nutrition awareness assistant.

            Answer the user's nutrition question using only
            the verified nutrition estimates supplied below.

            Food: ${result.food}
            Serving: ${fmt(result.grams)} g
            Calories: ${fmt(result.calories)} kcal
            Protein: ${fmt(result.protein)} g
            Carbohydrates: ${fmt(result.carbs)} g
            Fat: ${fmt(result.fat)} g

            These values are estimates from the NutriLens
            nutrition database, scaled by serving weight.

            Do not invent or change the supplied numbers.
            Keep the answer clear and concise.
            Do not provide medical diagnoses.

            User question:
            $question
        """.trimIndent()
    }
}
