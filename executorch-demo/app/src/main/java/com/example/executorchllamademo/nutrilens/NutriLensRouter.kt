package com.example.executorchllamademo.nutrilens

import android.content.Context
import org.json.JSONObject
import java.util.Locale

class NutriLensRouter(context: Context) {

    private val calculator = NutritionCalculator(context)

    private val foodNames: List<String> = context.assets
        .open("food_nutrition.json")
        .bufferedReader()
        .use { reader ->
            val database = JSONObject(reader.readText())
            database.keys().asSequence().toList()
        }

    fun respond(question: String): String? {
        val gramsMatch = Regex(
            """(?i)\b(\d+(?:\.\d+)?)\s*(?:g|grams?|grammes?)\b"""
        ).find(question) ?: return null

        val grams = gramsMatch.groupValues[1].toDoubleOrNull()
            ?: return null

        val normalized = question.lowercase(Locale.ROOT)
            .replace('_', ' ')
            .replace('-', ' ')

        val matchedFood = foodNames
            .filter { food ->
                val name = food.replace('_', ' ')
                    .lowercase(Locale.ROOT)

                Regex(
                    """(?<![\p{L}\p{N}])""" +
                        Regex.escape(name) +
                        """(?![\p{L}\p{N}])"""
                ).containsMatchIn(normalized)
            }
            .maxByOrNull { it.length }
            ?: return null

        val result = calculator.calculate(matchedFood, grams)
            ?: return "I couldn't calculate that serving. Please use a valid weight in grams."

        return calculator.format(result) +
            "\n\nThese are estimated nutrition values for the specified serving."
    }
}
