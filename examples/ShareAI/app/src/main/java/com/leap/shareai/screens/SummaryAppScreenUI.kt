package com.leap.shareai.screens

import android.util.Log
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.systemBars
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.halilibo.richtext.commonmark.CommonmarkAstNodeParser
import com.halilibo.richtext.markdown.BasicMarkdown
import com.halilibo.richtext.ui.material3.RichText
import com.leap.shareai.viewmodels.AIChatViewModel
import com.leap.shareai.viewmodels.WebScrapingViewModel
import com.leap.shareai.webscraping.WebPageState

enum class ProcessingStatus {
    LOADING_MODEL,
    SCRAPING_WEBPAGE,
    THINKING,
    GENERATING_SUMMARY,
    COMPLETED,
    ERROR
}

data class StatusStep(
    val status: ProcessingStatus,
    val title: String,
    val description: String
)

@Composable
fun SummaryAppScreen(
    modifier: Modifier = Modifier,
    linkUrl: String,
    webScrapingViewModel: WebScrapingViewModel = viewModel(),
    aiChatViewModel: AIChatViewModel = viewModel()
) {
    val generation by aiChatViewModel.generation.collectAsState()
    val webState by webScrapingViewModel.state
    LaunchedEffect(linkUrl) { webScrapingViewModel.extractTextFromUrl(linkUrl) }
    LaunchedEffect(webState, linkUrl) {
        val content = (webState as? WebPageState.Success)?.content
        if (content != null && content.url == linkUrl) aiChatViewModel.summarize(linkUrl, content.text)
    }
    val error = (webState as? WebPageState.Error)?.message ?: generation.error.orEmpty()
    val scraping = webState is WebPageState.Loading || webState is WebPageState.Idle
    val status = when {
        error.isNotEmpty() -> ProcessingStatus.ERROR
        scraping -> ProcessingStatus.SCRAPING_WEBPAGE
        generation.busy && generation.output.isEmpty() -> ProcessingStatus.LOADING_MODEL
        generation.busy -> ProcessingStatus.GENERATING_SUMMARY
        else -> ProcessingStatus.COMPLETED
    }
    Column(modifier.fillMaxSize()) {
        SummaryAppScreenUI(
            modifier = Modifier.weight(1f), currentStatus = status,
            isProcessing = scraping || generation.busy, isThinking = false,
            summaryText = if (scraping) "" else generation.output, errorText = error,
        )
        if (error.isNotEmpty()) {
            Button(onClick = {
                val content = (webState as? WebPageState.Success)?.content
                if (content != null && content.url == linkUrl) aiChatViewModel.summarize(linkUrl, content.text, retry = true)
                else webScrapingViewModel.extractTextFromUrl(linkUrl)
            }, enabled = !generation.busy) { Text("Retry") }
        }
    }
}

@Composable
fun SummaryAppScreenUI(
    modifier: Modifier = Modifier,
    currentStatus: ProcessingStatus,
    isProcessing: Boolean,
    summaryText: String,
    isThinking: Boolean,
    errorText: String = ""
) {
    val gradient = Brush.verticalGradient(listOf(Color(0xFF6366F1), Color(0xFF8B5CF6), Color(0xFFA855F7)))
    Box(modifier.fillMaxSize().background(gradient).windowInsetsPadding(WindowInsets.systemBars)) {
        Column(Modifier.fillMaxSize().padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Text("Summary Generator", fontSize = 28.sp, fontWeight = FontWeight.Bold, color = Color.White)
            Spacer(Modifier.height(8.dp))
            Text(
                when {
                    isProcessing -> "Processing your content..."
                    errorText.isNotEmpty() -> "Error occurred"
                    else -> "Summary Complete"
                }, color = Color.White, fontSize = 16.sp,
            )
            Spacer(Modifier.height(16.dp))
            when {
                errorText.isNotEmpty() -> ErrorCard(errorText, Modifier.weight(1f))
                summaryText.isNotEmpty() -> SummaryCard(summaryText, isThinking, Modifier.weight(1f))
                else -> Spacer(Modifier.weight(1f))
            }
            if (isProcessing) StatusIndicators(currentStatus, true, Modifier.fillMaxWidth())
        }
    }
}

@Composable
fun StatusIndicators(
    currentStatus: ProcessingStatus,
    isProcessing: Boolean,
    modifier: Modifier = Modifier
) {
    val statusSteps = listOf(
        StatusStep(
            ProcessingStatus.LOADING_MODEL,
            "Loading Model",
            "Initializing AI model"
        ),
        StatusStep(
            ProcessingStatus.SCRAPING_WEBPAGE,
            "Scraping Webpage",
            "Extracting content"
        ),
        StatusStep(
            ProcessingStatus.THINKING,
            "Thinking",
            "Analyzing content"
        ),
        StatusStep(
            ProcessingStatus.GENERATING_SUMMARY,
            "Generating Summary",
            "Creating final summary"
        )
    )

    val currentStep = statusSteps.find { it.status == currentStatus }

    val animatedOffset by animateFloatAsState(
        targetValue = if (isProcessing) 0f else -100f,
        animationSpec = tween(durationMillis = 500, easing = FastOutSlowInEasing),
        label = "statusOffset"
    )

    if (currentStep != null && isProcessing) {
        Column(
            modifier = modifier
                .fillMaxWidth()
                .padding(24.dp)
                .clip(CircleShape)
                .background(Color.White)
                .padding(16.dp)
                .offset(y = animatedOffset.dp),
        ) {
            StatusItem(
                step = currentStep,
                isActive = true,
                isCompleted = false,
            )
        }
    }
}

@Composable
fun StatusItem(
    step: StatusStep,
    isActive: Boolean,
    isCompleted: Boolean,
) {
    val infiniteTransition = rememberInfiniteTransition(label = "pulse")
    val pulseAlpha by infiniteTransition.animateFloat(
        initialValue = 0.4f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulse"
    )

    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Status circle
        Box(
            modifier = Modifier
                .size(40.dp)
                .clip(CircleShape)
                .background(
                    when {
                        isCompleted && !isActive -> Color(0xFF10B981) // Green
                        isActive -> Color(0xFF6366F1).copy(alpha = pulseAlpha) // Blue with pulse
                        else -> Color(0xFFE5E7EB) // Gray
                    }
                ),
            contentAlignment = Alignment.Center
        ) {
            when {
                isCompleted && !isActive -> {
                    Icon(
                        imageVector = Icons.Default.Check,
                        contentDescription = "Completed",
                        tint = Color.White,
                        modifier = Modifier.size(20.dp)
                    )
                }

                isActive -> {
                    CircularProgressIndicator(
                        modifier = Modifier.size(20.dp),
                        strokeWidth = 2.dp,
                        color = Color.White
                    )
                }

                else -> {
                    Box(
                        modifier = Modifier
                            .size(12.dp)
                            .clip(CircleShape)
                            .background(Color(0xFF9CA3AF))
                    )
                }
            }
        }

        Spacer(modifier = Modifier.width(16.dp))

        // Status text
        Column(
            modifier = Modifier.weight(1f)
        ) {
            Text(
                text = step.title,
                fontSize = 16.sp,
                fontWeight = FontWeight.SemiBold,
                color = when {
                    isActive -> Color(0xFF6366F1)
                    isCompleted -> Color(0xFF10B981)
                    else -> Color(0xFF6B7280)
                }
            )

            Text(
                text = step.description,
                fontSize = 14.sp,
                color = Color(0xFF9CA3AF),
                modifier = Modifier.alpha(if (isActive || isCompleted) 1f else 0.6f)
            )
        }
    }
}

@Composable
fun ErrorCard(errorText: String, modifier: Modifier = Modifier){
    Card(
        modifier = modifier
            .fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = Color.White.copy(alpha = 0.9f)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 12.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(20.dp)
        ) {
            Text(
                text = "Error",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = Color(0xFFB91C1C)
            )
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                text = errorText,
                fontSize = 16.sp,
                color = Color(0xFFB91C1C),
                lineHeight = 24.sp
            )
        }
    }
}


@Composable
fun SummaryCard(summaryText: String, isThinking: Boolean, modifier: Modifier) {
    val textScrollState = rememberScrollState()
    LaunchedEffect(summaryText) {
        textScrollState.scrollTo(textScrollState.maxValue)
    }
    Card(
        modifier = modifier
            .fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = Color.White.copy(alpha = 0.9f)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 12.dp)
    ) {
        Column(
            modifier = Modifier
                .verticalScroll(textScrollState)
                .fillMaxSize()
                .padding(20.dp)
        ) {
            Text(
                text = if (isThinking) "\uD83E\uDD14 Thinking" else "\uD83D\uDCDD Summary",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF1F2937)
            )
            Spacer(modifier = Modifier.height(12.dp))
            RichText(
                modifier = Modifier
            ) {
                val parser = remember { CommonmarkAstNodeParser() }
                val astNode = remember(summaryText, parser) {
                    parser.parse(summaryText)
                }
                BasicMarkdown(astNode)
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun SummaryAppScreenPreview() {
    MaterialTheme {
        SummaryAppScreenUI(
            currentStatus = ProcessingStatus.LOADING_MODEL,
            isProcessing = true,
            summaryText = "dna ld a",
            isThinking = false,
            errorText = ""
        )
    }
}