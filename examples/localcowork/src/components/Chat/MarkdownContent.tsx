/**
 * MarkdownContent — renders markdown text as formatted React elements.
 *
 * Uses react-markdown with remark-gfm for GitHub Flavored Markdown
 * (tables, strikethrough, task lists, autolinks). Wrapped in a
 * `.md-content` class that resets white-space from the parent's
 * `pre-wrap` to `normal` so paragraphs flow naturally.
 *
 * Used for assistant messages only — user messages remain plain text.
 */

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface MarkdownContentProps {
  /** The raw markdown string to render. */
  readonly content: string;
}

export function MarkdownContent({
  content,
}: MarkdownContentProps): React.JSX.Element {
  // Small LLMs (LFM2-24B-A2B) sometimes emit literal "\n" (two characters:
  // backslash + n) instead of real newline characters. Normalize these so
  // react-markdown can parse them as line breaks.
  const normalized = content.replace(/\\n/g, "\n");

  return (
    <div className="md-content">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{normalized}</ReactMarkdown>
    </div>
  );
}
