package com.leap.shareai.viewmodels

import android.util.Log
import androidx.compose.runtime.State
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.leap.shareai.webscraping.WebPageContent
import com.leap.shareai.webscraping.WebPageState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.jsoup.Jsoup
import org.jsoup.nodes.Document
import java.io.IOException

class WebScrapingViewModel : ViewModel() {
    private val _state = mutableStateOf<WebPageState>(WebPageState.Idle)
    val state: State<WebPageState> = _state

    fun extractTextFromUrl(url: String) {
        viewModelScope.launch {
            _state.value = WebPageState.Loading
            Log.d("WebScrapingViewModel", "loading data")
            try {
                val content = withContext(Dispatchers.IO) {
                    scrapeWebPage(url)
                }
                Log.d("WebScrapingViewModel", "data loaded")
                _state.value = WebPageState.Success(content)
            } catch (e: Exception) {
                Log.d("WebScrapingViewModel", "data loading error: $e")
                _state.value = WebPageState.Error(e.message ?: "Unknown error occurred")
            }
        }
    }

    private suspend fun scrapeWebPage(url: String): WebPageContent {
        return withContext(Dispatchers.IO) {
            try {
                // Connect and parse the document
                val document: Document = Jsoup.connect(url)
                    .userAgent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
                    .timeout(10000)
                    .get()

                // Extract title
                val title = document.title()

                // Remove script and style elements
                document.select("script, style, nav, header, footer, aside").remove()

                // Extract text content
                val text = document.body().text().trim()

                WebPageContent(
                    title = title,
                    text = text,
                    url = url
                )
            } catch (e: IOException) {
                throw Exception("Failed to connect to URL: ${e.message}")
            } catch (e: Exception) {
                throw Exception("Error parsing webpage: ${e.message}")
            }
        }
    }
}