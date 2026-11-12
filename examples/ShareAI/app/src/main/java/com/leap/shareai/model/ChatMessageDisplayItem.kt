package com.leap.shareai.model

enum class Role { USER, ASSISTANT }

data class ChatMessageDisplayItem(val role: Role, val text: String, val reasoning: String? = null)