package com.leap.shareai.webscraping

sealed class WebPageState {
    object Idle : WebPageState()
    object Loading : WebPageState()
    data class Success(val content: WebPageContent) : WebPageState()
    data class Error(val message: String) : WebPageState()
}