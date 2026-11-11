package ai.liquid.androidkoogagent.ui.screen.calculatorTool

sealed class CalculatorToolEvent {
    data class Calculate(val expression: String) : CalculatorToolEvent()
}