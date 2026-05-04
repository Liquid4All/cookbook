import Foundation
import os

/// Structured-logging facade for the POC. Backed by `os.Logger` so logs
/// surface in Console.app with subsystem/category filtering and honor
/// iOS privacy annotations.
///
/// Per CLAUDE.local.md "Observability" principle: no `print()`, no
/// `NSLog(...)` scattered through the codebase. Every component logs
/// via a topic-scoped `AppLog` instance so failures carry structured
/// context (layer, operation, error).
public enum AppLog {
    private static let subsystem = "ai.liquid.demos.telcotriage"

    public static let chat         = Logger(subsystem: subsystem, category: "chat")
    public static let intelligence = Logger(subsystem: subsystem, category: "intelligence")
    public static let tools        = Logger(subsystem: subsystem, category: "tools")
    public static let vision       = Logger(subsystem: subsystem, category: "vision")
    public static let lfm          = Logger(subsystem: subsystem, category: "lfm")
}
