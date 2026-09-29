import { defineConfig, loadEnv } from "vite";
import { DECISION_PATHS, RACERS } from "./config.js";

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

// Resolve which model, endpoint, and key each racer uses.
function resolveRacer(name, env) {
  const { envPrefix, model, baseUrl } = RACERS[name];
  const keyName = `${envPrefix}_API_KEY`;
  const base = env[`${envPrefix}_BASE_URL`] || baseUrl;

  let url;
  try {
    const { host } = new URL(base);
    const path = DECISION_PATHS[host] || DECISION_PATHS.default;
    url = new URL(path, base).toString();
  } catch {
    return { error: `Invalid ${envPrefix}_BASE_URL "${base}".` };
  }

  return {
    model: env[`${envPrefix}_MODEL_NAME`] || model,
    url,
    apiKey: env[keyName],
    keyName,
  };
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
      name: "road-decider-api-proxy",
      configureServer(server) {
        server.middlewares.use(async (req, res, next) => {
          if (!req.url?.startsWith("/api/")) {
            next();
            return;
          }

          // Re-read .env on every request so key changes apply without a restart.
          const env = loadEnv(server.config.mode, server.config.root, "");

          if (req.url === "/api/config" && req.method === "GET") {
            const racers = Object.fromEntries(
              Object.keys(RACERS).map((name) => {
                const racer = resolveRacer(name, env);
                const error = racer.error || (racer.apiKey ? null : `Missing ${racer.keyName} in .env`);
                return [name, { ready: !error, error }];
              }),
            );
            sendJson(res, 200, racers);
            return;
          }

          const match = req.url.match(/^\/api\/decision\/(\w+)$/);
          if (match && RACERS[match[1]] && req.method === "POST") {
            await proxyDecision(req, res, resolveRacer(match[1], env));
            return;
          }

          sendJson(res, 404, { error: "Unknown Road Decider API route." });
        });
      },
    },
  ],
});

async function proxyDecision(req, res, racer) {
  if (racer.error) {
    sendJson(res, 400, { error: racer.error });
    return;
  }
  if (!racer.apiKey) {
    sendJson(res, 401, {
      error: `Missing ${racer.keyName}. Copy .env.example to .env and add your key.`,
    });
    return;
  }

  let body;
  try {
    body = JSON.parse(await readBody(req));
  } catch {
    sendJson(res, 400, { error: "Request body must be valid JSON." });
    return;
  }

  try {
    // The model is set server-side so the key and model always match.
    const upstream = await fetch(racer.url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${racer.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ...body, model: racer.model }),
    });
    const text = await upstream.text();
    res.statusCode = upstream.status;
    res.setHeader("Content-Type", upstream.headers.get("content-type") || "application/json");
    res.end(text);
  } catch (error) {
    sendJson(res, 502, { error: error instanceof Error ? error.message : "Decision proxy failed." });
  }
}
