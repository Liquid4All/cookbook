package ai.liquid.androidkoogagent.ui.screen.toolsList

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.Button
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import ai.liquid.androidkoogagent.R

@Composable
fun ToolsListScreen(
    state: ToolsListState,
    onEvent: (ToolsListEvent) -> Unit,
    modelStatus: String,
    onImportModel: () -> Unit,
) {
    Column {
        Text(
            text = stringResource(R.string.available_tools),
            style = MaterialTheme.typography.headlineMedium,
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        )

        Button(onClick = onImportModel, modifier = Modifier.padding(horizontal = 16.dp)) {
            Text("Import LFM3-1.7B GGUF")
        }
        Text(modelStatus, modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp))

        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(8.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(state.tools, key = { it.id }) {
                ToolListItem(
                    item = it,
                    onClick = {
                        onEvent(
                            ToolsListEvent.OnToolClick(
                                it.id
                            )
                        )
                    }
                )
            }
        }
    }
}

@Composable
private fun ToolListItem(
    item: ToolsListState.ToolItem,
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(MaterialTheme.colorScheme.surfaceContainer)
            .clickable(onClick = onClick)
            .padding(16.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(
            modifier = Modifier.weight(1f),
            horizontalAlignment = Alignment.Start,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = item.name,
                modifier = Modifier.padding(bottom = 4.dp),
                style = MaterialTheme.typography.bodyLarge
            )

            Text(
                text = item.description,
                style = MaterialTheme.typography.bodyMedium
            )
        }
    }
}
