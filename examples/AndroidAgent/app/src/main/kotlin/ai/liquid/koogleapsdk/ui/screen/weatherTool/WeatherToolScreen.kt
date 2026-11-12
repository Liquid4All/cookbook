package ai.liquid.koogleapsdk.ui.screen.weatherTool

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import ai.liquid.koogleapsdk.R

@Composable
fun WeatherToolScreen(
    state: WeatherToolState,
    onEvent: (WeatherToolEvent) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            text = stringResource(R.string.weather_tool),
            style = MaterialTheme.typography.headlineMedium,
            modifier = Modifier.fillMaxWidth()
        )

        var city by remember { mutableStateOf("New York, 2025-10-22, DAILY") }
        OutlinedTextField(
            value = city,
            onValueChange = {
                city = it
            },
            placeholder = {
                Text(stringResource(R.string.enter_city_name))
            },
            modifier = Modifier.fillMaxWidth()
        )

        if (state.weatherInfo?.isNotEmpty() == true && !state.isLoading) {
            Column(
                modifier = Modifier
                    .verticalScroll(rememberScrollState())
                    .padding(vertical = 8.dp)
                    .weight(1f)
            ) {
                Text(
                    text = state.weatherInfo,
                )
            }
        } else if (state.isLoading) {
            Text(
                text = stringResource(R.string.loading),
                modifier = Modifier
                    .padding(vertical = 8.dp)
            )
        }

        Button(
            onClick = {
                onEvent(WeatherToolEvent.OnSearchClick(city))
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(text = stringResource(R.string.get_weather))
        }
    }
}