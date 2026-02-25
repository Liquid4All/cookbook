/**
 * Guidance for users who selected a local GGUF model file.
 *
 * Shows how to serve the model via llama.cpp or import it into Ollama,
 * with the actual selected file path interpolated into the commands.
 */

interface LocalModelGuideProps {
  readonly modelPath: string;
}

/** Collapsible guide for serving a local GGUF model. */
export function LocalModelGuide({
  modelPath,
}: LocalModelGuideProps): React.JSX.Element {
  const isGguf = modelPath.toLowerCase().endsWith(".gguf");

  return (
    <div className="local-model-guide">
      <p className="local-model-guide-title">How to serve this model</p>
      <p className="local-model-guide-hint">
        LocalCowork connects to a local model via an OpenAI-compatible API.
        You need to start a model server before using the app.
      </p>

      {isGguf ? (
        <>
          <div className="install-step">
            <span className="install-step-label">
              Option A: llama.cpp (recommended for GGUF)
            </span>
            <code className="install-code">
              llama-server --model &quot;{modelPath}&quot; --port 8080 --ctx-size 32768
            </code>
          </div>
          <div className="install-step">
            <span className="install-step-label">
              Option B: Import into Ollama
            </span>
            <code className="install-code">
              {`echo 'FROM "${modelPath}"' > Modelfile && ollama create my-model -f Modelfile`}
            </code>
          </div>
        </>
      ) : (
        <div className="install-step">
          <span className="install-step-label">
            Start a compatible model server pointing to your file, then
            configure the endpoint in Settings after setup.
          </span>
        </div>
      )}

      <p className="local-model-guide-hint">
        The server must be running at{" "}
        <code className="install-code-inline">localhost:8080</code> (llama.cpp)
        or <code className="install-code-inline">localhost:11434</code> (Ollama).
      </p>
    </div>
  );
}
