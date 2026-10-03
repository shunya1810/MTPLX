import Foundation

/// What the header badge says about the engine (issue #528).
///
/// The engine and the live-stats stream are separate facts. A running
/// daemon whose stats stream dropped or was refused is still running and
/// says so ("Running · reconnecting live stats"); only the engine's own
/// state reads Degraded or Crashed. Before this split a running engine with
/// a refused stream read "Offline", and the header could not tell a user
/// whether the model server or only the dashboard feed was in trouble.
public struct DaemonStatusBadge: Equatable, Sendable {
    public enum Tone: Equatable, Sendable {
        /// The engine runs and the live stats are open.
        case healthy
        /// The engine is starting, warming or stopping, or it runs while
        /// the live stats connect, reconnect or stay unavailable.
        case pending
        /// The engine is degraded or crashed.
        case failed
        /// The engine is stopped.
        case idle
    }

    public let label: String
    public let help: String
    public let tone: Tone

    /// Longest degraded reason shown inline; the full text stays in `help`.
    static let inlineReasonLimit = 44

    public init(daemonState: DaemonState, connectionState: MetricsConnectionState) {
        switch daemonState {
        case .running:
            switch connectionState {
            case .open:
                label = tr("Running")
                help = tr("Running and ready.")
                tone = .healthy
            case .connecting, .idle:
                label = tr("Running · connecting live stats")
                help = tr("MTPLX is running. Connecting to live stats…")
                tone = .pending
            case .reconnecting(let attempt):
                label = tr("Running · reconnecting live stats")
                help = tr("MTPLX is running. Live stats are reconnecting (attempt %lld).", attempt)
                tone = .pending
            case .failed(let reason):
                label = tr("Running · live stats unavailable")
                help = tr("MTPLX is running. Live stats are unavailable: %@", reason)
                tone = .pending
            }
        case .degraded(let reason):
            // The reason used to live only in a hover tooltip; screenshots
            // showed a bare "Degraded". Show a capped reason inline.
            let trimmed = reason.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty {
                label = tr("Degraded")
            } else {
                let capped = trimmed.count > Self.inlineReasonLimit
                    ? String(trimmed.prefix(Self.inlineReasonLimit))
                        .trimmingCharacters(in: .whitespaces) + "…"
                    : trimmed
                label = tr("Degraded — %@", capped)
            }
            help = tr("Degraded: %@", reason)
            tone = .failed
        case .starting:
            label = tr("Starting")
            help = tr("Starting up…")
            tone = .pending
        case .warming:
            label = tr("Warming")
            help = tr("Loading the model…")
            tone = .pending
        case .stopping:
            label = tr("Stopping")
            help = tr("Stopping…")
            tone = .pending
        case .crashed(let status):
            label = tr("Crashed")
            help = status.map { tr("Crashed (exit code %@).", String($0)) } ?? tr("Crashed.")
            tone = .failed
        case .stopped:
            label = tr("Stopped")
            help = tr("Not running.")
            tone = .idle
        }
    }
}
