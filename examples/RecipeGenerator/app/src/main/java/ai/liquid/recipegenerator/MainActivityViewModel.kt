package ai.liquid.recipegenerator

import android.app.Application
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import ai.liquid.android.runtime.Models
import ai.liquid.android.runtime.TextTaskViewModel
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
data class Recipe(
    val name: String,

    val ingredients: List<String>,

    val cookingTime: Int,

    val isVegetarian: Boolean,

    val steps: List<String>,

)

// JSON schema for Recipe. llama.cpp turns it into a grammar, so the model can only generate matching JSON.
private const val RECIPE_SCHEMA = """
{
  "type": "object",
  "additionalProperties": false,
  "description": "A recipe for cooking great dishes",
  "properties": {
    "name": { "type": "string" },
    "ingredients": { "type": "array", "items": { "type": "string" }, "description": "Ingredients for making the dishes" },
    "cookingTime": { "type": "integer", "description": "Cooking time in minutes" },
    "isVegetarian": { "type": "boolean", "description": "Whether the meal is vegetarian" },
    "steps": { "type": "array", "items": { "type": "string" }, "description": "Steps of cooking" }
  },
  "required": ["name", "ingredients", "cookingTime", "isVegetarian", "steps"]
}
"""

class MainActivityViewModel(application: Application) : TextTaskViewModel(
    application, Models.text,
    "You are a recipe generator. Generate JSON that follows this schema: $RECIPE_SCHEMA",
    RECIPE_SCHEMA,
) {
    var recipeState: Recipe? by mutableStateOf(null)
        private set
    fun generateRecipe() { generate("A recipe for a dinner dish") }
    override fun completed(output: String) { recipeState = Json.decodeFromString<Recipe>(output) }
}
