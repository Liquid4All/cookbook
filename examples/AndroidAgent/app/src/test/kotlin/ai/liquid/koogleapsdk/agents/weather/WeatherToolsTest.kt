package ai.liquid.koogleapsdk.agents.weather

import io.ktor.client.HttpClient
import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.respond
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.http.HttpHeaders
import io.ktor.http.headersOf
import io.ktor.serialization.kotlinx.json.json
import kotlinx.coroutines.runBlocking
import kotlinx.datetime.Instant
import kotlinx.serialization.json.Json
import org.junit.Assert.*
import org.junit.Test

class WeatherToolsTest {
    private val now = Instant.parse("2026-11-16T12:00:00Z")

    @Test fun tomorrowIsResolvedInTheLocationsTimezoneNearUtcMidnight() = runBlocking {
        val engine = MockEngine { request ->
            val body = if (request.url.host.startsWith("geocoding")) {
                """{"results":[{"id":1,"name":"New York","latitude":40.7,"longitude":-74.0,"timezone":"America/New_York"}]}"""
            } else {
                assertEquals("2026-11-17", request.url.parameters["start_date"])
                assertEquals("2026-11-17", request.url.parameters["end_date"])
                """{"latitude":40.7,"longitude":-74.0,"daily":{"time":["2026-11-17"]}}"""
            }
            respond(body, headers = headersOf(HttpHeaders.ContentType, "application/json"))
        }
        val http = HttpClient(engine) { install(ContentNegotiation) { json() } }
        try {
            val result = WeatherTools.WeatherForecastTool.execute(
                WeatherTools.WeatherForecastTool.Args("New York", "tomorrow"), OpenMeteoClient(http),
                Instant.parse("2026-11-17T00:30:00Z"))
            assertEquals("2026-11-17", result.date)
        } finally { http.close() }
    }

    @Test fun tomorrowRequestsTheActualIntervalAndReturnsEveryRequestedDay() = runBlocking {
        val engine = MockEngine { request ->
            val json = if (request.url.host.startsWith("geocoding")) {
                """{"results":[{"id":1,"name":"London","latitude":51.5,"longitude":-0.1,"timezone":"Europe/London"}]}"""
            } else {
                assertEquals("2026-11-17", request.url.parameters["start_date"])
                assertEquals("2026-11-18", request.url.parameters["end_date"])
                assertNull(request.url.parameters["forecast_days"])
                """{"latitude":51.5,"longitude":-0.1,"daily":{"time":["2026-11-17","2026-11-18"],"temperature_2m_max":[15,16],"temperature_2m_min":[5,6]}}"""
            }
            respond(json, headers = headersOf(HttpHeaders.ContentType, "application/json"))
        }
        val http = HttpClient(engine) { install(ContentNegotiation) { json(Json { ignoreUnknownKeys = true }) } }
        try {
            val result = WeatherTools.WeatherForecastTool.execute(
                WeatherTools.WeatherForecastTool.Args("London", "2026-11-17", 2), OpenMeteoClient(http), now)
            assertEquals("2026-11-17", result.date)
            assertTrue(result.forecast.contains("2026-11-17"))
            assertTrue(result.forecast.contains("2026-11-18"))
            assertFalse(result.forecast.contains("2026-11-16"))
        } finally { http.close() }
    }

    @Test fun unavailableDateNeverFallsBackToToday() = runBlocking {
        val engine = MockEngine { request ->
            val json = if (request.url.host.startsWith("geocoding"))
                """{"results":[{"id":1,"name":"London","latitude":51.5,"longitude":-0.1}]}"""
            else """{"latitude":51.5,"longitude":-0.1,"daily":{"time":["2026-11-16"]}}"""
            respond(json, headers = headersOf(HttpHeaders.ContentType, "application/json"))
        }
        val http = HttpClient(engine) { install(ContentNegotiation) { json() } }
        try {
            val result = WeatherTools.WeatherForecastTool.execute(
                WeatherTools.WeatherForecastTool.Args("London", "2026-11-17"), OpenMeteoClient(http), now)
            assertEquals("No forecast data available for 2026-11-17", result.forecast)
            try {
                WeatherTools.WeatherForecastTool.execute(WeatherTools.WeatherForecastTool.Args("London", "2026-12-02"), OpenMeteoClient(http), now)
                fail("Out-of-range date accepted")
            } catch (_: IllegalArgumentException) {}
        } finally { http.close() }
    }
}
