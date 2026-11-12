package com.leap.shareai.model

import ai.liquid.leap.message.ChatMessage

data class ChatMessageDisplayItem(val role: ChatMessage.Role, val text: String, val reasoning: String? = null)