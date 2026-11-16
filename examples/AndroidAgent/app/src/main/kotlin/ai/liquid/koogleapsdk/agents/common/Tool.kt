package ai.liquid.koogleapsdk.agents.common

import kotlinx.serialization.KSerializer
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject

/**
 * A function the agent can call. The model sees its [name], [description] and
 * [parameters] (a JSON schema), and calls it with JSON arguments that decode into [Args].
 */
abstract class Tool<Args, Result> {
    abstract val name: String
    abstract val description: String
    abstract val parameters: JsonObject
    abstract val argsSerializer: KSerializer<Args>

    abstract suspend fun execute(args: Args): Result

    /** The text the model sees as the tool's result */
    open fun resultText(result: Result): String = result.toString()
}

/** JSON schema for a tool's arguments */
fun parametersSchema(
    vararg properties: Pair<String, JsonObject>,
    required: List<String> = properties.map { it.first },
): JsonObject = buildJsonObject {
    put("type", "object")
    putJsonObject("properties") { properties.forEach { (name, schema) -> put(name, schema) } }
    putJsonArray("required") { required.forEach { add(it) } }
}

/** JSON schema for one argument */
fun property(type: String, description: String): JsonObject = buildJsonObject {
    put("type", type)
    put("description", description)
}
