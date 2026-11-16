package com.leap.shareai

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.leap.shareai.screens.SummaryAppScreen
import com.leap.shareai.ui.theme.ThemeShareAi

class MainActivity : ComponentActivity() {
    private var link by mutableStateOf("https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        (savedInstanceState?.getString("link") ?: sharedLink(intent))?.let { link = it }
        setContent { ThemeShareAi { SummaryAppScreen(linkUrl = link) } }
    }
    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        sharedLink(intent)?.let { link = it }
    }
    override fun onSaveInstanceState(outState: Bundle) {
        outState.putString("link", link)
        super.onSaveInstanceState(outState)
    }
    private fun sharedLink(intent: Intent): String? = intent.takeIf {
        it.action == Intent.ACTION_SEND && it.type == "text/plain"
    }?.getStringExtra(Intent.EXTRA_TEXT)
}
