package ai.liquid.koogleapsdk.ui.screen.calculatorTool

sealed class CalculatorToolEvent {
    data class Calculate(val expression: String) : CalculatorToolEvent()
}