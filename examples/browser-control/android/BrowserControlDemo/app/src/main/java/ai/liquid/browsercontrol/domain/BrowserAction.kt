package ai.liquid.browsercontrol.domain

sealed class BrowserAction {
    object Noop : BrowserAction()
    data class Click(val bid: String) : BrowserAction()
    data class Fill(val bid: String, val text: String) : BrowserAction()
    data class SendKeys(val text: String) : BrowserAction()
    data class Scroll(val direction: String) : BrowserAction()

    override fun toString(): String = when (this) {
        is Noop -> "noop()"
        is Click -> "click('$bid')"
        is Fill -> "fill('$bid', '$text')"
        is SendKeys -> "send_keys('$text')"
        is Scroll -> "scroll('$direction')"
    }
}

fun parseAction(response: String): BrowserAction {
    // 提取第一行包含括号的语句
    val actionLine = response.lines()
        .firstOrNull { it.contains("(") && it.contains(")") }
        ?.trim()
        ?: return BrowserAction.Noop

    return when {
        actionLine.startsWith("click(") -> {
            val bid = actionLine.substringAfter("('").substringBefore("')")
            BrowserAction.Click(bid)
        }
        actionLine.startsWith("fill(") -> {
            val content = actionLine.substringAfter("(").substringBefore(")")
            val parts = content.split(",").map { it.trim().trim('\'', '"') }
            if (parts.size >= 2) {
                BrowserAction.Fill(parts[0], parts[1])
            } else {
                BrowserAction.Noop
            }
        }
        actionLine.startsWith("send_keys(") -> {
            val text = actionLine.substringAfter("('").substringBefore("')")
            BrowserAction.SendKeys(text)
        }
        actionLine.startsWith("scroll(") -> {
            val direction = actionLine.substringAfter("('").substringBefore("')")
            BrowserAction.Scroll(direction)
        }
        else -> BrowserAction.Noop
    }
}
