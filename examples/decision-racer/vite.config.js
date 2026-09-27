import fs from "node:fs";
import path from "node:path";
import { defineConfig } from "vite";

const DECISION_ENV_PATH = "/Users/leonie/Documents/code/decision/.env";

function readEnvFile(filePath) {
  if (!fs.existsSync(filePath)) return {};
  return Object.fromEntries(
    fs
      .readFileSync(filePath, "utf8")
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith("#"))
      .map((line) => {
        const index = line.indexOf("=");
        if (index === -1) return [line, ""];
        return [line.slice(0, index), line.slice(index + 1).replace(/^["']|["']$/g, "")];
      }),
  );
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
    });
    req.on("end", () => resolve(body));
    req.on("error", reject);
  });
}

function sendJson(res, status, data) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(data));
}

export default defineConfig({
  server: {
    port: 5173,
  },
  build: {
    target: "esnext",
  },
  plugins: [
    {
      name: "decision-racer-api-proxy",
      configureServer(server) {
        server.middlewares.use(async (req, res, next) => {
          if (!req.url?.startsWith("/api/")) {
            next();
            return;
          }

          const env = { ...readEnvFile(DECISION_ENV_PATH), ...process.env };

          if (req.url === "/api/config" && req.method === "GET") {
            sendJson(res, 200, {
              hasLiquidKey: Boolean(env.LIQUID_API_KEY),
              hasOpenRouterKey: Boolean(env.OPENROUTER_API_KEY),
              liquidModel: env.LIQUID_MODEL || "LiquidAI/LFM2.5-S1",
              jevModel: env.JEV_MODEL || "typesafe/jev-1.13",
            });
            return;
          }

          if (req.url === "/api/decision/liquid" && req.method === "POST") {
            await proxyDecision(req, res, {
              apiUrl: "https://api.liquid.ai/v1/systemone",
              apiKey: env.LIQUID_API_KEY,
              missingMessage: `Missing LIQUID_API_KEY. Set LIQUID_API_KEY in your shell environment or add it to ${path.basename(DECISION_ENV_PATH)} before running the dev server.`,
            });
            return;
          }

          if (req.url === "/api/decision/jev" && req.method === "POST") {
            await proxyDecision(req, res, {
              apiUrl: "https://openrouter.ai/api/alpha/decisions",
              apiKey: env.OPENROUTER_API_KEY,
              missingMessage: "Missing OPENROUTER_API_KEY. Set OPENROUTER_API_KEY in your shell environment before running the dev server.",
            });
            return;
          }

          sendJson(res, 404, { error: "Unknown Road Decider API route." });
        });
      },
    },
  ],
});

async function proxyDecision(req, res, { apiUrl, apiKey, missingMessage }) {
  if (!apiKey) {
    sendJson(res, 401, { error: missingMessage });
    return;
  }

  try {
    const body = await readBody(req);
    const upstream = await fetch(apiUrl, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body,
    });
    const text = await upstream.text();
    res.statusCode = upstream.status;
    res.setHeader("Content-Type", upstream.headers.get("content-type") || "application/json");
    res.end(text);
  } catch (error) {
    sendJson(res, 502, { error: error instanceof Error ? error.message : "Decision proxy failed." });
  }
}
