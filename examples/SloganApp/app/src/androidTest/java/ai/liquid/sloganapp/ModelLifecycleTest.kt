package ai.liquid.sloganapp

import androidx.lifecycle.ViewModelProvider
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

/** Seed the app's external files directory with the verified Models.text GGUF before running. */
@RunWith(AndroidJUnit4::class)
class ModelLifecycleTest {
    @Test fun rotationRetainsOwnerAndFinishingDuringGenerationDoesNotBreakReplacement() = runBlocking {
        var first: SloganViewModel? = null
        val scenario = ActivityScenario.launch(MainActivity::class.java)
        try {
            scenario.onActivity { activity ->
                first = ViewModelProvider(activity)[SloganViewModel::class.java]
                first!!.generate("Write a detailed paragraph of slogans for a local bakery.")
            }
            val original = checkNotNull(first)
            withTimeout(120_000) { original.generation.first { it.busy && it.output.isNotEmpty() } }
            scenario.recreate()
            scenario.onActivity { assertSame(original, ViewModelProvider(it)[SloganViewModel::class.java]) }
            val result = withTimeout(120_000) { original.generation.first { !it.busy } }
            assertNull(result.error)
            assertTrue(result.output.isNotBlank())

            scenario.onActivity { original.generate("Write a very long story of at least 5000 words about a bakery.") }
            withTimeout(120_000) { original.generation.first { it.busy && it.output.isNotEmpty() } }
            scenario.close()
            ActivityScenario.launch(MainActivity::class.java).use { replacement ->
                var second: SloganViewModel? = null
                replacement.onActivity {
                    second = ViewModelProvider(it)[SloganViewModel::class.java]
                    assertNotSame(original, second)
                    second!!.generate("Write one short slogan for a bakery.")
                }
                val newResult = withTimeout(120_000) { checkNotNull(second).generation.first { !it.busy } }
                assertNull(newResult.error)
                assertTrue(newResult.output.isNotBlank())
            }
        } finally { scenario.close() }
    }
}
