package ai.liquid.recipegenerator

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.Button
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

class MainActivity : ComponentActivity() {
    val viewModel: MainActivityViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        if (viewModel.generation.value.status == "Ready") viewModel.generateRecipe()

        setContent {
            Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                val state by viewModel.generation.collectAsState()
                val recipe = viewModel.recipeState
                if (recipe == null) {
                    Column(
                        modifier = Modifier
                            .padding(innerPadding)
                            .fillMaxSize(),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Center
                    ) {
                        Text(
                            text = state.error ?: state.status,
                            fontStyle = FontStyle.Italic,
                            fontSize = 24.sp,
                            textAlign = TextAlign.Center
                        )
                        Button(onClick = { viewModel.generateRecipe() }, enabled = !state.busy) { Text("Retry") }
                    }
                } else {
                    Column(Modifier.padding(innerPadding)) {
                        RecipeView(recipe, modifier = Modifier.weight(1f).padding(8.dp))
                        Text(state.error ?: state.status)
                        Button(onClick = { viewModel.generateRecipe() }, enabled = !state.busy) { Text("Generate another recipe") }
                    }
                }
            }
        }
    }

    private fun startGeneration() {
        viewModel.generateRecipe()
    }
}

@Composable
fun RecipeView(recipe: Recipe, modifier: Modifier = Modifier) {
    Column(modifier.verticalScroll(rememberScrollState())) {
        Text(recipe.name, fontSize = 24.sp)
        Text(
            "Cooking time: ${recipe.cookingTime} min",
            fontSize = 18.sp,
            modifier = Modifier.padding(top = 10.dp)
        )
        Text(
            "Vegetarian: ${
                if (recipe.isVegetarian) {
                    "Yes"
                } else {
                    "No"
                }
            }", fontSize = 18.sp,
            modifier = Modifier.padding(top = 10.dp)
        )
        Text("Ingredients", fontSize = 18.sp, modifier = Modifier.padding(top = 10.dp))
        Column {
            recipe.ingredients.map { item ->
                Text(" $item")
            }
        }
        Text("Steps", fontSize = 18.sp, modifier = Modifier.padding(top = 10.dp))
        Column {
            recipe.steps.map { item ->
                Text(" $item")
            }
        }
    }
}