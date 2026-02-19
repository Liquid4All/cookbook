/**
 * FileIcon — returns an appropriate icon character for a file entry.
 *
 * Uses the file extension or entry type to select a visual icon.
 */

import type { FileEntryType } from "../../types";

interface FileIconProps {
  readonly name: string;
  readonly entryType: FileEntryType;
}

/** Map file extensions to icon characters. */
function iconForExtension(ext: string): string {
  switch (ext.toLowerCase()) {
    case "pdf":
      return "\uD83D\uDCC4"; // page facing up
    case "doc":
    case "docx":
      return "\uD83D\uDCD8"; // blue book
    case "xls":
    case "xlsx":
    case "csv":
      return "\uD83D\uDCCA"; // bar chart
    case "ppt":
    case "pptx":
      return "\uD83D\uDCCA"; // bar chart
    case "jpg":
    case "jpeg":
    case "png":
    case "gif":
    case "svg":
    case "webp":
      return "\uD83D\uDDBC\uFE0F"; // framed picture
    case "mp3":
    case "wav":
    case "flac":
    case "m4a":
      return "\uD83C\uDFB5"; // musical note
    case "mp4":
    case "mov":
    case "avi":
    case "mkv":
      return "\uD83C\uDFAC"; // clapper board
    case "zip":
    case "tar":
    case "gz":
    case "rar":
    case "7z":
      return "\uD83D\uDCE6"; // package
    case "js":
    case "ts":
    case "tsx":
    case "jsx":
    case "py":
    case "rs":
    case "go":
    case "java":
    case "c":
    case "cpp":
    case "h":
      return "\uD83D\uDCDD"; // memo
    case "json":
    case "yaml":
    case "yml":
    case "toml":
    case "xml":
      return "\u2699\uFE0F"; // gear
    case "md":
    case "txt":
    case "rtf":
      return "\uD83D\uDCC3"; // page with curl
    case "html":
    case "css":
      return "\uD83C\uDF10"; // globe
    default:
      return "\uD83D\uDCC4"; // page facing up (generic file)
  }
}

/** Get icon for a file entry. */
function getIcon(name: string, entryType: FileEntryType): string {
  if (entryType === "dir") {
    return "\uD83D\uDCC1"; // open file folder
  }
  if (entryType === "symlink") {
    return "\uD83D\uDD17"; // link
  }
  const dotIndex = name.lastIndexOf(".");
  if (dotIndex < 0) {
    return "\uD83D\uDCC4"; // generic file
  }
  const ext = name.slice(dotIndex + 1);
  return iconForExtension(ext);
}

export function FileIcon({ name, entryType }: FileIconProps): React.JSX.Element {
  return <span className="file-icon">{getIcon(name, entryType)}</span>;
}
