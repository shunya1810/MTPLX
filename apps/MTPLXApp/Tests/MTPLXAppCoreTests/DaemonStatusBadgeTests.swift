import XCTest

@testable import MTPLXAppCore

/// Issue #528: the header badge said "Degraded" or "Offline" while the
/// engine answered /health and served requests. The engine and the
/// live-stats stream are separate facts, and the badge now says which one is
/// in trouble.
final class DaemonStatusBadgeTests: XCTestCase {
    override func setUp() {
        super.setUp()
        L10n.activate(.english)
    }

    private func badge(
        _ daemon: DaemonState,
        _ connection: MetricsConnectionState
    ) -> DaemonStatusBadge {
        DaemonStatusBadge(daemonState: daemon, connectionState: connection)
    }

    func testRunningEngineWithOpenStatsIsTheOnlyHealthyBadge() {
        let healthy = badge(.running, .open)
        XCTAssertEqual(healthy.label, "Running")
        XCTAssertEqual(healthy.help, "Running and ready.")
        XCTAssertEqual(healthy.tone, .healthy)

        let others: [(DaemonState, MetricsConnectionState)] = [
            (.running, .connecting), (.running, .idle), (.running, .reconnecting(1)),
            (.running, .failed("refused")), (.starting, .open), (.stopping, .open),
            (.degraded("x"), .open), (.crashed(1), .idle), (.stopped, .idle),
        ]
        for (daemon, connection) in others {
            XCTAssertNotEqual(badge(daemon, connection).tone, .healthy, "\(daemon) / \(connection)")
        }
    }

    /// The reported case: the stats stream drops while the engine keeps
    /// serving. The badge names the stream and keeps "Running" in front.
    func testDroppedStatsStreamOnARunningEngineSaysRunningNotDegradedOrOffline() {
        let reconnecting = badge(.running, .reconnecting(3))
        XCTAssertEqual(reconnecting.label, "Running · reconnecting live stats")
        XCTAssertEqual(reconnecting.help, "MTPLX is running. Live stats are reconnecting (attempt 3).")
        XCTAssertEqual(reconnecting.tone, .pending)

        let refused = badge(.running, .failed("Metrics stream rejected the app API key."))
        XCTAssertEqual(refused.label, "Running · live stats unavailable")
        XCTAssertEqual(
            refused.help,
            "MTPLX is running. Live stats are unavailable: Metrics stream rejected the app API key."
        )
        XCTAssertEqual(refused.tone, .pending, "a refused stats stream is not an engine failure")

        for connection in [MetricsConnectionState.connecting, .idle] {
            let connecting = badge(.running, connection)
            XCTAssertEqual(connecting.label, "Running · connecting live stats")
            XCTAssertEqual(connecting.help, "MTPLX is running. Connecting to live stats…")
            XCTAssertEqual(connecting.tone, .pending)
        }

        for connection in [
            MetricsConnectionState.connecting, .idle, .reconnecting(7), .failed("gone"),
        ] {
            let label = badge(.running, connection).label
            XCTAssertTrue(label.hasPrefix("Running"), label)
            XCTAssertFalse(label.contains("Degraded"), label)
            XCTAssertFalse(label.contains("Offline"), label)
        }
    }

    /// A real engine failure still reads Degraded, whatever the stream does.
    func testEngineDegradationWinsOverTheStatsStream() {
        let reason = "MTPLX lost contact with the model server. Start it again."
        for connection in [MetricsConnectionState.open, .reconnecting(2), .failed(reason), .idle] {
            let degraded = badge(.degraded(reason), connection)
            XCTAssertEqual(degraded.tone, .failed)
            XCTAssertTrue(degraded.label.hasPrefix("Degraded — "), degraded.label)
            XCTAssertEqual(degraded.help, "Degraded: \(reason)")
        }
    }

    func testDegradedReasonIsCappedInlineAndKeptWholeInTheTooltip() {
        let long = String(repeating: "port busy ", count: 10)
        let degraded = badge(.degraded(long), .idle)
        XCTAssertTrue(degraded.label.hasSuffix("…"), degraded.label)
        XCTAssertLessThanOrEqual(
            degraded.label.count,
            "Degraded — ".count + DaemonStatusBadge.inlineReasonLimit + 1
        )
        XCTAssertEqual(degraded.help, "Degraded: \(long)")

        XCTAssertEqual(badge(.degraded("  "), .idle).label, "Degraded")
        XCTAssertEqual(badge(.degraded("Port 8000 busy"), .idle).label, "Degraded — Port 8000 busy")
    }

    func testLifecycleStatesKeepTheirWords() {
        XCTAssertEqual(badge(.starting, .idle).label, "Starting")
        XCTAssertEqual(badge(.starting, .idle).help, "Starting up…")
        XCTAssertEqual(badge(.warming, .idle).label, "Warming")
        XCTAssertEqual(badge(.stopping, .open).label, "Stopping")
        XCTAssertEqual(badge(.stopped, .idle).label, "Stopped")
        XCTAssertEqual(badge(.stopped, .idle).tone, .idle)
        XCTAssertEqual(badge(.crashed(9), .idle).label, "Crashed")
        XCTAssertEqual(badge(.crashed(9), .idle).help, "Crashed (exit code 9).")
        XCTAssertEqual(badge(.crashed(nil), .idle).help, "Crashed.")
        XCTAssertEqual(badge(.crashed(nil), .idle).tone, .failed)
        for state in [DaemonState.starting, .warming, .stopping] {
            XCTAssertEqual(badge(state, .idle).tone, .pending, "\(state)")
        }
    }

    /// The new words ship in every language, not only in English.
    func testLiveStatsWordsAreTranslated() {
        let english = badge(.running, .reconnecting(2)).label
        for language in AppLanguage.allCases where language != .english {
            L10n.activate(language)
            let translated = badge(.running, .reconnecting(2))
            XCTAssertNotEqual(translated.label, english, language.code)
            XCTAssertTrue(translated.help.contains("2"), "\(language.code): \(translated.help)")
            XCTAssertFalse(translated.help.contains("%lld"), "\(language.code): \(translated.help)")
            XCTAssertFalse(badge(.running, .failed("x")).help.contains("%@"), language.code)
        }
        L10n.activate(.english)
    }
}
