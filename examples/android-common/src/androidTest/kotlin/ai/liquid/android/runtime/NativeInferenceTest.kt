package ai.liquid.android.runtime

import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.ByteArrayOutputStream
import java.io.File
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

/** These tests load real GGUF checkpoints. Seed targetContext.externalFilesDir with Models' files. */
@RunWith(AndroidJUnit4::class)
class NativeInferenceTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val directory = checkNotNull(context.getExternalFilesDir(null))
    private fun file(spec: ModelSpec) = File(directory, spec.name).also { check(it.length() == spec.size) { "Seed ${it.absolutePath}" } }
    private fun text() = NativeTextModel(context.applicationInfo.nativeLibraryDir)
    private suspend fun output(model: NativeTextModel, prompt: String, schema: String = "", maxTokens: Int = 256): String {
        val result = StringBuilder()
        model.generate("You are a helpful assistant.", prompt, schema, maxTokens).collect { result.append(it) }
        return result.toString().also { Log.i("LiquidRuntimeTest", "Text response: $it") }
    }

    @Test fun textUnicodeSchemasConcurrentCallsAndRecovery() = runBlocking {
        val model = text()
        try {
            model.load(file(Models.small))
            val values = listOf("😀", "München", "東京")
            val replies = values.map { value -> async {
                val schema = JSONObject().put("type", "object").put("properties",
                    JSONObject().put("value", JSONObject().put("type", "string").put("const", value)))
                    .put("required", JSONArray().put("value")).put("additionalProperties", false).toString()
                JSONObject(output(model, "Return the required JSON value $value.", schema)).getString("value")
            } }.awaitAll()
            assertEquals(values, replies)
            try { output(model, "Hello", "{invalid"); fail("Invalid schema accepted") } catch (_: IllegalStateException) {}
            try {
                output(model, "Return a long value.",
                    """{"type":"object","properties":{"value":{"const":"a deliberately long required value"}},"required":["value"]}""", 1)
                fail("Incomplete JSON accepted as a complete response")
            } catch (error: IllegalStateException) { assertTrue(error.message.orEmpty().contains("limit")) }
            try {
                output(model, "word ".repeat(9000))
                fail("Oversized prompt accepted")
            } catch (error: IllegalStateException) { assertTrue(error.message.orEmpty().contains("context")) }
            val recovery = output(model, "Return the recovery marker.",
                """{"type":"object","properties":{"status":{"const":"recovered"}},"required":["status"],"additionalProperties":false}""")
            assertEquals("recovered", JSONObject(recovery).getString("status"))
            val summary = output(model, "Summarize in one sentence: The workshop is on Tuesday in London. It begins at 10 AM and ends at noon. Bring a laptop.")
            for (fact in listOf("Tuesday", "London", "10", "noon", "laptop")) {
                assertTrue(summary, summary.contains(fact, ignoreCase = true))
            }
        } finally { model.close().await() }
    }

    @Test fun cancellationCloseAndReplacementDoNotRace() = runBlocking {
        val model = text()
        model.load(file(Models.small))
        val started = CompletableDeferred<Unit>()
        val job = launch {
            try {
                model.generate("You are a helpful assistant.", "Write a very long detailed story of at least 5000 words.", maxTokens = 2048)
                    .collect { started.complete(Unit) }
            } catch (_: IllegalStateException) { /* Closing explicitly interrupts generation. */ }
        }
        withTimeout(120_000) { started.await() }
        val replacement = text()
        try {
            replacement.load(file(Models.small))
            withTimeout(30_000) { model.close().await(); job.join() }
            val result = output(replacement, "Return the replacement marker.",
                """{"type":"object","properties":{"status":{"const":"replacement"}},"required":["status"],"additionalProperties":false}""")
            assertEquals("replacement", JSONObject(result).getString("status"))
        } finally { model.close().await(); replacement.close().await() }
    }

    @Test fun agentParsesUnicodeAndCompletesToolRoundTrip() = runBlocking {
        val model = NativeAgentModel(context.applicationInfo.nativeLibraryDir)
        try {
            model.load(file(Models.text))
            val hello = JSONObject(model.chat("""[{"role":"user","content":"Say hello to 😀 in München."}]""", "[]"))
            Log.i("LiquidRuntimeTest", "Agent Unicode response: $hello")
            assertFalse(hello.has("error"))
            assertTrue(hello.getString("content").isNotBlank())
            val tools = """[{"type":"function","function":{"name":"plus","description":"Add two numbers","parameters":{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}}}]"""
            val messages = JSONArray("""[{"role":"system","content":"Use the plus tool for arithmetic, then report the result."},{"role":"user","content":"Use plus to add 4 and 5."}]""")
            val incomplete = JSONObject(model.chat(messages.toString(), tools, maxTokens = 1))
            assertTrue(incomplete.getString("error").contains("limit"))
            val reply = JSONObject(model.chat(messages.toString(), tools))
            Log.i("LiquidRuntimeTest", "Agent tool call: $reply")
            assertFalse(reply.has("error"))
            val calls = reply.getJSONArray("tool_calls")
            assertTrue(calls.length() > 0)
            messages.put(reply)
            for (index in 0 until calls.length()) {
                val call = calls.getJSONObject(index)
                val function = call.getJSONObject("function")
                assertEquals("plus", function.getString("name"))
                val args = JSONObject(function.getString("arguments"))
                val sum = args.getDouble("a") + args.getDouble("b")
                assertEquals(9.0, sum, 0.0)
                messages.put(JSONObject().put("role", "tool").put("tool_call_id", call.getString("id")).put("content", sum.toString()))
            }
            val answer = JSONObject(model.chat(messages.toString(), tools))
            Log.i("LiquidRuntimeTest", "Agent final response: $answer")
            assertFalse(answer.has("error"))
            assertTrue(answer.getString("content").contains("9"))
        } finally { model.close().await() }
    }

    @Test fun visionDescribesActualImageAndReleasesItsProjector() = runBlocking {
        val model = NativeVisionModel(context.applicationInfo.nativeLibraryDir)
        val bitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
        bitmap.eraseColor(Color.RED)
        val image = ByteArrayOutputStream().also { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }.toByteArray()
        bitmap.recycle()
        try {
            model.load(file(Models.vision), file(Models.projector))
            model.load(file(Models.vision), file(Models.projector))
            try {
                model.load(file(Models.vision), File(directory, "another-projector.gguf"))
                fail("Changing the loaded projector was silently accepted")
            } catch (error: IllegalStateException) { assertTrue(error.message.orEmpty().contains("projector")) }
            try {
                model.generate("You are a helpful assistant.", "Describe this image.", byteArrayOf(0)).collect {}
                fail("Invalid image accepted")
            } catch (error: IllegalStateException) { assertTrue(error.message.orEmpty().contains("decode")) }
            val result = StringBuilder()
            model.generate("You are a helpful assistant.", "What is the background color? Answer with one word.", image, 64)
                .collect { result.append(it) }
            assertTrue(result.toString(), result.toString().contains("red", ignoreCase = true))
        } finally { model.close().await() }
    }
}
