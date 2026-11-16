package com.leap.shareai.viewmodels

import android.app.Application
import androidx.lifecycle.viewModelScope
import ai.liquid.android.runtime.Models
import ai.liquid.android.runtime.TextTaskViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

class AIChatViewModel(application: Application) : TextTaskViewModel(
    application, Models.small,
    "You are a helpful assistant.",
) {
    private var lastUrl: String? = null
    private var request: Job? = null
    fun summarize(url: String, text: String, retry: Boolean = false) {
        if (!retry && url == lastUrl && generation.value.error == null) return
        lastUrl = url
        request?.cancel()
        request = viewModelScope.launch {
            stopAndJoin()
            generate("Summarize in one sentence: $text")
        }
    }
}
