package ai.liquid.koogleapsdk.agents.calculator

import ai.liquid.koogleapsdk.agents.common.Tool
import ai.liquid.koogleapsdk.agents.common.parametersSchema
import ai.liquid.koogleapsdk.agents.common.property
import kotlinx.serialization.Serializable

object CalculatorTools {
    abstract class CalculatorTool(
        override val name: String,
        override val description: String,
    ) : Tool<CalculatorTool.Args, CalculatorTool.Result>() {
        @Serializable
        data class Args(
            val a: Float,
            val b: Float
        )

        @Serializable
        class Result(val result: Float)

        final override val argsSerializer = Args.serializer()
        final override val parameters = parametersSchema(
            "a" to property("number", "First number"),
            "b" to property("number", "Second number"),
        )

        final override fun resultText(result: Result) = result.result.toString()
    }

    /**
     * 2. Implement the tool (tools).
     */

    object PlusTool : CalculatorTool(
        name = "plus",
        description = "Adds a and b",
    ) {
        override suspend fun execute(args: Args): Result {
            return Result(args.a + args.b)
        }
    }

    object MinusTool : CalculatorTool(
        name = "minus",
        description = "Subtracts b from a",
    ) {
        override suspend fun execute(args: Args): Result {
            return Result(args.a - args.b)
        }
    }

    object DivideTool : CalculatorTool(
        name = "divide",
        description = "Divides a and b",
    ) {
        override suspend fun execute(args: Args): Result {
            return Result(args.a / args.b)
        }
    }

    object MultiplyTool : CalculatorTool(
        name = "multiply",
        description = "Multiplies a and b",
    ) {
        override suspend fun execute(args: Args): Result {
            return Result(args.a * args.b)
        }
    }

}