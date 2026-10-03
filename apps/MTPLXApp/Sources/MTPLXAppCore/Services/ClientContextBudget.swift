import Foundation

/// The window MTPLX writes into Pi and OpenCode.
///
/// On 2026-09-29 the app wrote 262,144 into Pi's `contextWindow` and
/// `maxTokens` straight from Settings while the engine could not serve a
/// conversation that long. Once the daemon is up it publishes the window it
/// executes (`/health` `execution_window`), and clients are configured from
/// that. Before the daemon answers (the first write of a launch) the setting
/// is used.
///
/// The window is also Pi's answer ceiling (`maxTokens`): Pi clamps each
/// request to the room its prompt leaves, and the server caps each answer to
/// the memory actually free, so no fixed share of the window is held back
/// from an answer (mtplx/server/served_window.py). OpenCode's `limit.output`
/// is OpenCode's own reply reserve (`OpenCodeIntegration.outputLimit`).
public enum ClientContextBudget {
    /// The served window when the daemon published one, else the setting
    /// (`configuration.effectiveContextWindow(default:)`).
    public static func window(
        configuration: MTPLXAppConfiguration,
        served: ServedExecutionWindow?,
        defaultWindow: Int
    ) -> Int {
        if let served, served.tokens > 0 {
            return served.tokens
        }
        return configuration.effectiveContextWindow(default: defaultWindow)
    }
}
