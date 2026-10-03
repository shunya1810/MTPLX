import Darwin
import Foundation
import XCTest

@testable import MTPLXAppCore

/// `DaemonSupervisor.currentHold()` (#528) tells a start request whether the
/// supervisor already runs a daemon, so the store can reconnect to it
/// instead of calling `start` and getting `alreadyRunning`.
final class DaemonHoldTests: XCTestCase {
    private let baseURL = URL(string: "http://127.0.0.1:9")!

    private actor Gate {
        private var entered = false
        private var enteredWaiters: [CheckedContinuation<Void, Never>] = []
        private var release: CheckedContinuation<Void, Never>?

        func pass() async {
            entered = true
            enteredWaiters.forEach { $0.resume() }
            enteredWaiters.removeAll()
            await withCheckedContinuation { release = $0 }
        }

        func waitUntilEntered() async {
            if entered { return }
            await withCheckedContinuation { enteredWaiters.append($0) }
        }

        func open() {
            release?.resume()
            release = nil
        }
    }

    private func idleDaemonCommand(launchID: String?) -> DaemonCommand {
        DaemonCommand(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "trap 'exit 0' TERM; while :; do sleep 1; done"],
            environment: launchID.map { ["MTPLX_APP_LAUNCH_ID": $0] } ?? [:]
        )
    }

    private func healthPayload(launchID: String, pid: pid_t, modelPath: String) throws -> HealthPayload {
        let json = """
        {
          "ok": true,
          "model": "fixture",
          "model_path": "\(modelPath)",
          "generation_mode": "mtp",
          "load_mtp": true,
          "mtp_enabled": true,
          "depth": 1,
          "profile": {},
          "context_window": 1024,
          "active_requests": 0,
          "reasoning_parser": "none",
          "startup": {"launch_id": "\(launchID)", "pid": \(pid)}
        }
        """
        return try JSONDecoder().decode(HealthPayload.self, from: Data(json.utf8))
    }

    func testOwnedLaunchIsSettlingUntilItRunsThenHeldUntilStopped() async throws {
        let gate = Gate()
        let supervisor = DaemonSupervisor(beforeProcessRun: { await gate.pass() })
        XCTAssertEqual(supervisor.currentHold(), .none)

        let command = idleDaemonCommand(launchID: "owned-launch")
        let url = baseURL
        let start = Task {
            try await supervisor.start(command: command, healthBaseURL: url, probeHealth: false)
        }
        await gate.waitUntilEntered()
        XCTAssertEqual(supervisor.currentHold(), .settling, "a launch between reservation and run is not a daemon yet")
        await gate.open()
        _ = try await start.value

        let epoch = supervisor.supervisionSnapshot().lifecycleEpoch
        XCTAssertEqual(
            supervisor.currentHold(),
            .held(HeldDaemon(lifecycleEpoch: epoch, launchID: "owned-launch", baseURL: baseURL, adopted: false))
        )

        await supervisor.stop(graceSeconds: 1)
        XCTAssertEqual(supervisor.currentHold(), .none)
    }

    func testAdoptedDaemonIsHeldUnderTheLaunchIDItReported() async throws {
        // A real process stands in for the adopted daemon, so Stop signals
        // something this test owns. It carries the launch id in its
        // environment, as a daemon the app launched does; Stop signals only
        // a process that does.
        let adopted = Process()
        adopted.executableURL = URL(fileURLWithPath: "/bin/sleep")
        adopted.arguments = ["30"]
        adopted.environment = ["MTPLX_APP_LAUNCH_ID": "prior-session"]
        try adopted.run()
        addTeardownBlock { if adopted.isRunning { adopted.terminate() } }
        let health = try healthPayload(
            launchID: "prior-session",
            pid: adopted.processIdentifier,
            modelPath: "/tmp/mtplx-hold-model"
        )
        let supervisor = DaemonSupervisor(initialHealthProbe: { _, _ in health })

        let result = try await supervisor.adoptExistingIfAppOwned(
            command: DaemonCommand(
                executableURL: URL(fileURLWithPath: "/bin/sh"),
                arguments: ["--model", "/tmp/mtplx-hold-model"]
            ),
            healthBaseURL: baseURL
        )
        XCTAssertEqual(result, health)
        let epoch = supervisor.supervisionSnapshot().lifecycleEpoch
        XCTAssertEqual(
            supervisor.currentHold(),
            .held(HeldDaemon(lifecycleEpoch: epoch, launchID: "prior-session", baseURL: baseURL, adopted: true))
        )

        await supervisor.stop(graceSeconds: 1)
        XCTAssertEqual(supervisor.currentHold(), .none)
        for _ in 0..<100 where adopted.isRunning {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertFalse(adopted.isRunning, "Stop signalled the adopted daemon it could confirm")
    }

    /// The supervisor contract itself is unchanged: `start` on a held slot
    /// still refuses. The store asks `currentHold()` first.
    func testStartOnAHeldSlotStillThrowsAlreadyRunning() async throws {
        let supervisor = DaemonSupervisor()
        _ = try await supervisor.start(
            command: idleDaemonCommand(launchID: "first"),
            healthBaseURL: baseURL,
            probeHealth: false
        )
        do {
            _ = try await supervisor.start(
                command: idleDaemonCommand(launchID: "second"),
                healthBaseURL: baseURL,
                probeHealth: false
            )
            XCTFail("a second start on a held slot must throw")
        } catch {
            XCTAssertEqual(error as? DaemonSupervisorError, .alreadyRunning)
        }
        guard case .held(let held) = supervisor.currentHold() else {
            return XCTFail("the first daemon is still held")
        }
        XCTAssertEqual(held.launchID, "first")
        await supervisor.stop(graceSeconds: 1)
    }
}
