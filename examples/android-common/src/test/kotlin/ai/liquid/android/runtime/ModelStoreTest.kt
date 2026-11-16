package ai.liquid.android.runtime

import java.nio.file.Files
import java.security.MessageDigest
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.mockwebserver.SocketPolicy
import okio.Buffer
import org.junit.Assert.*
import org.junit.Test

class ModelStoreTest {
    private val payload = "GGUF-fixture-😀".toByteArray()
    private fun digest(bytes: ByteArray) = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it.toInt() and 255) }
    private fun response(bytes: ByteArray = payload) = MockResponse().setBody(Buffer().write(bytes))
    private fun spec(server: MockWebServer, bytes: ByteArray = payload) = ModelSpec("model.gguf", server.url("/model").toString(), bytes.size.toLong(), digest(bytes))

    @Test fun partialInvalidAndConcurrentDownloadsRecoverAndCachedModelsWorkOffline() = runBlocking {
        val directory = Files.createTempDirectory("model-store").toFile()
        val server = MockWebServer()
        server.enqueue(response())
        server.start()
        val spec = spec(server)
        val store = ModelStore(directory)
        try {
            directory.resolve("model.gguf").writeBytes(ByteArray(payload.size))
            directory.resolve("model.gguf.part").writeText("interrupted transfer")
            val files = (1..8).map { async { store.ensure(spec) } }.awaitAll()
            assertEquals(1, server.requestCount)
            files.forEach { assertArrayEquals(payload, it.readBytes()) }
            assertFalse(directory.resolve("model.gguf.part").exists())
            server.shutdown()
            assertArrayEquals(payload, store.ensure(spec).readBytes())
        } finally { server.shutdown(); directory.deleteRecursively() }
    }

    @Test fun corruptDownloadIsNeverPublishedAndRetrySucceeds() = runBlocking {
        val directory = Files.createTempDirectory("model-store-failure").toFile()
        val server = MockWebServer()
        server.enqueue(response(ByteArray(payload.size)))
        server.enqueue(response())
        server.start()
        val spec = spec(server)
        try {
            try { ModelStore(directory).ensure(spec); fail("Bad checksum accepted") } catch (_: IllegalStateException) {}
            assertFalse(directory.resolve(spec.name).exists())
            assertFalse(directory.resolve("${spec.name}.part").exists())
            assertArrayEquals(payload, ModelStore(directory).ensure(spec).readBytes())
        } finally { server.shutdown(); directory.deleteRecursively() }
    }

    @Test fun httpFailureAndTruncatedBodyNeverPublishModel() = runBlocking {
        val directory = Files.createTempDirectory("model-store-http").toFile()
        val server = MockWebServer()
        server.enqueue(MockResponse().setResponseCode(503))
        server.enqueue(response(ByteArray(64 * 1024)).setSocketPolicy(SocketPolicy.DISCONNECT_DURING_RESPONSE_BODY))
        server.start()
        try {
            for (attempt in 1..2) {
                var failed = false
                try { ModelStore(directory).ensure(spec(server, ByteArray(64 * 1024))) } catch (_: Exception) { failed = true }
                assertTrue("Failed response was accepted", failed)
                assertFalse(directory.resolve("model.gguf").exists())
                assertFalse(directory.resolve("model.gguf.part").exists())
            }
        } finally { server.shutdown(); directory.deleteRecursively() }
    }

    @Test fun cancellationRemovesPartialBeforeAnotherCallerCanRetry() = runBlocking {
        val directory = Files.createTempDirectory("model-store-cancel").toFile()
        val server = MockWebServer()
        val bytes = ByteArray(256 * 1024) { 42 }
        server.enqueue(response(bytes).throttleBody(1024, 10, TimeUnit.MILLISECONDS))
        server.enqueue(response(bytes))
        server.start()
        try {
            val started = CompletableDeferred<Unit>()
            val store = ModelStore(directory)
            val spec = spec(server, bytes)
            val job = launch(Dispatchers.Default) { store.ensure(spec) { started.complete(Unit) } }
            withTimeout(5_000) { started.await() }
            withTimeout(5_000) { job.cancelAndJoin() }
            assertFalse(directory.resolve("model.gguf").exists())
            assertFalse(directory.resolve("model.gguf.part").exists())
            assertArrayEquals(bytes, store.ensure(spec).readBytes())
        } finally { server.shutdown(); directory.deleteRecursively() }
    }
}
