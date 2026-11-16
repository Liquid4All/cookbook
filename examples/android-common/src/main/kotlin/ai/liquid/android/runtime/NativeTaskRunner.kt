package ai.liquid.android.runtime

import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** One execution queue owns every native operation, including final release. */
class NativeTaskRunner(private val release: () -> Unit) {
    private val dispatcher = Executors.newSingleThreadExecutor { task ->
        Thread(task, "liquid-inference").apply { isDaemon = true }
    }.asCoroutineDispatcher()
    private val scope = CoroutineScope(SupervisorJob() + dispatcher)
    private val closing = AtomicBoolean(false)
    private val closed = CompletableDeferred<Unit>()

    suspend fun <T> run(block: (shouldContinue: () -> Boolean) -> T): T {
        check(!closing.get()) { "The model has been closed" }
        val caller = currentCoroutineContext()
        return withContext(dispatcher) {
            caller.ensureActive()
            check(!closing.get()) { "The model has been closed" }
            block { caller.isActive && !closing.get() }
        }
    }

    /** Cancellation is observed by the native token loop before release can run. */
    fun close(): Deferred<Unit> {
        if (closing.compareAndSet(false, true)) {
            scope.launch {
                try {
                    release()
                    closed.complete(Unit)
                } catch (error: Exception) {
                    closed.completeExceptionally(error)
                } finally {
                    scope.cancel()
                    dispatcher.close()
                }
            }
        }
        return closed
    }
}
