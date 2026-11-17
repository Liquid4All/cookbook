package ai.liquid.android.runtime

import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test

class NativeTaskRunnerTest {
    @Test fun requestsAreSerializedAndCloseRunsExactlyOnceAfterGeneration() = runBlocking {
        val active = AtomicInteger()
        val maximum = AtomicInteger()
        val released = AtomicInteger()
        val runner = NativeTaskRunner {
            assertEquals(0, active.get())
            released.incrementAndGet()
        }
        (1..12).map {
            async(Dispatchers.Default) {
                runner.run {
                    maximum.updateAndGet { old -> maxOf(old, active.incrementAndGet()) }
                    Thread.sleep(5)
                    active.decrementAndGet()
                }
            }
        }.awaitAll()
        assertEquals(1, maximum.get())
        runner.close().await()
        runner.close().await()
        assertEquals(1, released.get())
    }

    @Test fun closingCancelsTheBlockingOperationBeforeRelease() = runBlocking {
        val entered = CountDownLatch(1)
        val finished = CountDownLatch(1)
        val runner = NativeTaskRunner { assertEquals(0L, finished.count) }
        val job = launch(Dispatchers.Default) {
            runner.run { active ->
                entered.countDown()
                while (active()) Thread.sleep(1)
                finished.countDown()
            }
        }
        assertTrue(entered.await(3, TimeUnit.SECONDS))
        runner.close().await()
        job.join()
        assertEquals(0L, finished.count)
    }

    @Test fun callerCancellationStopsNativeWorkBeforeNextRequest() = runBlocking {
        val entered = CountDownLatch(1)
        val finished = CountDownLatch(1)
        val runner = NativeTaskRunner {}
        val job = launch(Dispatchers.Default) {
            runner.run { active ->
                entered.countDown()
                while (active()) Thread.sleep(1)
                finished.countDown()
            }
        }
        assertTrue(entered.await(3, TimeUnit.SECONDS))
        job.cancelAndJoin()
        runner.run { assertEquals(0L, finished.count) }
        runner.close().await()
    }
}
