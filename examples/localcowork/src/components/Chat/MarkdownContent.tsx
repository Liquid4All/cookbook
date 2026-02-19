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
  return (
    <div className="md-content">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
    </div>
  );
}
