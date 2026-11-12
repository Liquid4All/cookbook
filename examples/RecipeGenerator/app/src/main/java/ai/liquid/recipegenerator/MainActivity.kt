package ai.liquid.recipegenerator

import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    val viewModel: MainActivityViewModel by viewModels()

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            startGeneration()
        } else {
            // Permission denied, proceed without notifications
            startGeneration()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // Check and request notification permission for Android 13+
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            when {
                ContextCompat.checkSelfPermission(
                    this,
                    android.Manifest.permission.POST_NOTIFICATIONS
                ) == PackageManager.PERMISSION_GRANTED -> {
                    startGeneration()
                }
                else -> {
                    requestPermissionLauncher.launch(android.Manifest.permission.POST_NOTIFICATIONS)
                }
            }
        } else {
            startGeneration()
        }

        setContent {
            Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
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
                            text = viewModel.status,
                            fontStyle = FontStyle.Italic,
                            fontSize = 24.sp,
                            textAlign = TextAlign.Center
                        )
                    }
                } else {
                    Column(Modifier.padding(innerPadding)) {
                        RecipeView(recipe, modifier = Modifier.padding(8.dp))
                    }
                }
            }
        }
    }

    private fun startGeneration() {
        viewModel.generateRecipe(this)
    }
}

@Composable
fun RecipeView(recipe: Recipe, modifier: Modifier = Modifier) {
    Column(modifier) {
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