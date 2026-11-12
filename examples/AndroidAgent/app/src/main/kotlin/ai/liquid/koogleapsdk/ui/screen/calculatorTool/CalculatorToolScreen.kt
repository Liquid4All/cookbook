package ai.liquid.koogleapsdk.ui.screen.calculatorTool

import ai.liquid.koogleapsdk.R
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
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

@Composable
fun CalculatorToolScreen(
    state: CalculatorToolState,
    onEvent: (CalculatorToolEvent) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            text = stringResource(R.string.calculator_tool),
            style = MaterialTheme.typography.headlineMedium,
            modifier = Modifier.fillMaxWidth()
        )

        var expression by remember { mutableStateOf("1234 + 4321") }
        OutlinedTextField(
            value = expression,
            onValueChange = {
                expression = it
            },
            placeholder = {
                Text(stringResource(R.string.enter_expression))
            },
            modifier = Modifier.fillMaxWidth()
        )

        if (state.toolCalls.isEmpty() || state.isCalculating) {
            Spacer(modifier = Modifier.weight(1f))
        } else {
            LazyColumn(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .padding(vertical = 8.dp)
            ) {
                items(state.toolCalls) {
                    Text("Tool call: $it")
                }

                item {
                    Spacer(modifier = Modifier.padding(4.dp))
                    Text(
                        text = "Answer: ${state.answer}",
                        style = MaterialTheme.typography.bodyLarge,
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            }
        }

        Button(
            onClick = {
                onEvent(CalculatorToolEvent.Calculate(expression))
            },
            modifier = Modifier
                .fillMaxWidth()
                .padding(8.dp),
            enabled = !state.isCalculating && expression.isNotBlank(),
        ) {
            Text(text = "Calculate")
        }
    }
}