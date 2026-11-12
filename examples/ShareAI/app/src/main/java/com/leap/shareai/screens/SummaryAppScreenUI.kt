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
import com.leap.shareai.viewmodels.LeapState
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
    var currentStatus by remember { mutableStateOf(ProcessingStatus.LOADING_MODEL) }
    var isProcessing by remember { mutableStateOf(true) }
    var summaryText by remember { mutableStateOf("") }
    var errorText by remember { mutableStateOf("") }

    val aiChatState by aiChatViewModel.state
    val webScrapingState by webScrapingViewModel.state

    val isLoading by aiChatViewModel.isLoading.collectAsState(false)
    val context = LocalContext.current

    LaunchedEffect(Unit) { aiChatViewModel.loadModel(context) }
    LaunchedEffect(isLoading) {
        when {
            isLoading -> {
                currentStatus = ProcessingStatus.LOADING_MODEL
                isProcessing = true
            }

            else -> {
                Log.d("SummaryAppScreenUI", "Request to load url")
                webScrapingViewModel.extractTextFromUrl(
                    url = linkUrl
                )
            }
        }
    }

    val message by aiChatViewModel.responseChunks.collectAsState("")
    val messages = remember { mutableStateOf("") }

    val reasoningChunks by aiChatViewModel.reasoningChunks.collectAsState("")
    val reasoningMessages = remember { mutableStateOf("") }

    LaunchedEffect(reasoningChunks) {
        if (reasoningChunks.isNotEmpty()) {
            reasoningMessages.value += reasoningChunks
            summaryText = reasoningMessages.value
        }
    }

    LaunchedEffect(message) {
        if (message.isNotEmpty()) {
            messages.value += message
            summaryText = messages.value
        }
    }

    LaunchedEffect(webScrapingState) {
        when (webScrapingState) {

            is WebPageState.Error -> {
                currentStatus = ProcessingStatus.ERROR
                errorText = (webScrapingState as WebPageState.Error).message
                isProcessing = false
            }

            WebPageState.Idle -> {}

            WebPageState.Loading -> {
                currentStatus = ProcessingStatus.SCRAPING_WEBPAGE
                isProcessing = true
            }

            is WebPageState.Success -> {
                currentStatus = ProcessingStatus.SCRAPING_WEBPAGE
                isProcessing = true

                aiChatViewModel.sendMessage(
                    userInput = (webScrapingState as WebPageState.Success).content.text
                )
            }
        }
    }

    LaunchedEffect(aiChatState) {
        when (aiChatState) {
            LeapState.Idle -> {}
            LeapState.Loading -> {
                currentStatus = ProcessingStatus.LOADING_MODEL
                isProcessing = true
            }

            is LeapState.Error -> {
                currentStatus = ProcessingStatus.ERROR
                errorText = (aiChatState as LeapState.Error).message
                isProcessing = false
            }

            LeapState.Thinking -> {
                currentStatus = ProcessingStatus.THINKING
                isProcessing = true
            }

            LeapState.Generating -> {
                currentStatus = ProcessingStatus.GENERATING_SUMMARY
                isProcessing = true
            }

            is LeapState.Success -> {
                currentStatus = ProcessingStatus.COMPLETED
                isProcessing = false
                summaryText = (aiChatState as LeapState.Success).response
                messages.value = summaryText // Update the messages state with the final summary
            }
        }
    }
    SummaryAppScreenUI(
        modifier = modifier,
        currentStatus = currentStatus,
        isProcessing = isProcessing,
        isThinking = aiChatState == LeapState.Thinking,
        summaryText = summaryText,
        errorText = errorText
    )
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
    val gradientBrush = Brush.verticalGradient(
        colors = listOf(
            Color(0xFF6366F1), // Indigo
            Color(0xFF8B5CF6), // Violet
            Color(0xFFA855F7)  // Purple
        )
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(gradientBrush)
            .windowInsetsPadding(WindowInsets.systemBars)
    ) {
        Column(
            modifier = modifier
                .fillMaxSize()
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(modifier = Modifier.height(60.dp))

            // Header
            Text(
                text = "Summary Generator",
                fontSize = 28.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White,
                textAlign = TextAlign.Center
            )

            Spacer(modifier = Modifier.height(12.dp))

            Text(
                text = when {
                    isProcessing -> "Processing your content..."
                    errorText.isNotEmpty() -> "Error occurred"
                    else -> "Summary Complete"
                },
                fontSize = 16.sp,
                color = Color.White.copy(alpha = 0.8f),
                textAlign = TextAlign.Center
            )

            Spacer(modifier = Modifier.height(40.dp))

            // Main content area
            when {
                errorText.isNotEmpty() -> {
                    ErrorCard(errorText, Modifier.weight(1f))
                }

                summaryText.isNotEmpty() -> {
                    SummaryCard(summaryText, isThinking, modifier.weight(1f))
                }

                else -> {
                    Spacer(modifier = Modifier.weight(1f))
                }
            }

            Spacer(modifier = Modifier.height(40.dp))
        }

        // Status indicators at the bottom
        StatusIndicators(
            currentStatus = currentStatus,
            isProcessing = isProcessing,
            modifier = Modifier.align(Alignment.BottomCenter)
        )
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
            .fillMaxWidth()
            .padding(bottom = 90.dp),
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