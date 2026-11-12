package ai.liquid.koogleapsdk.ui.screen

import androidx.activity.compose.BackHandler
import androidx.compose.animation.AnimatedContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHostState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.liquid.koogleapsdk.ui.navigation.Destination
import ai.liquid.koogleapsdk.ui.navigation.NavigationService
import ai.liquid.koogleapsdk.ui.screen.calculatorTool.CalculatorToolRoute
import ai.liquid.koogleapsdk.ui.screen.toolsList.ToolsListRoute
import ai.liquid.koogleapsdk.ui.screen.weatherTool.WeatherToolRoute
import ai.liquid.koogleapsdk.ui.util.SnackbarUtil
import androidx.compose.material3.SnackbarHost

@Composable
fun MainScreen() {
    val navigationService = remember { NavigationService.Companion.Instance }
    val destination by navigationService.destinationFlow.collectAsStateWithLifecycle()
    val snackbarHostState = remember { SnackbarHostState() }

    LaunchedEffect(snackbarHostState) {
        SnackbarUtil.snackbarHostState = snackbarHostState
    }

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { innerPadding ->
        AnimatedContent(
            targetState = destination,
            modifier = Modifier.padding(innerPadding)
        ) { destination ->
            when (destination) {
                Destination.CalculatorTool -> CalculatorToolRoute()
                Destination.ToolsList -> ToolsListRoute()
                Destination.WeatherTool -> WeatherToolRoute()
            }
        }
    }

    BackHandler {
        navigationService.navigateBack()
    }
}