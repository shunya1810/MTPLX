import Darwin
import XCTest

@testable import MTPLXAppCore

/// The hero gauge shows the rate of the phase the answer is in, not the
/// whole answer's average. On 2026-09-29 reasoning ran at 36 to 49 tok/s and
/// answers at 57 to 79, and the gauge read about 40 through the answers
/// because it divided every token by the time since the first one. The
/// server now sends `phase_tok_s`; the finished request is still held at its
/// receipt's cumulative `decode_tok_s`.
final class PhaseRateGaugeTests: XCTestCase {
    func testLiveFramePrefersThePhaseRateOverTheCumulativeRate() {
        let frame: [String: JSONValue] = [
            "decode_tok_s": .number(40.0),
            "decode_phase": .string("answer"),
            "phase_tok_s": .number(71.5),
        ]
        XCTAssertEqual(MTPLXBackendStore.headlineDecodeTPS(values: frame, live: true), 71.5)
    }

    func testCompletedReadingKeepsTheCumulativeReceiptRate() {
        // A snapshot merges the receipt into the live values, so the phase
        // rate of the last frame can still be present: the held figure must
        // not pick it up.
        let merged: [String: JSONValue] = [
            "decode_tok_s": .number(40.0),
            "decode_phase": .string("answer"),
            "phase_tok_s": .number(71.5),
        ]
        XCTAssertEqual(MTPLXBackendStore.headlineDecodeTPS(values: merged, live: false), 40.0)
    }

    func testPhaseAwareFrameWithoutARateYetLeavesTheGaugeAlone() {
        // The first second of decode: the phase is known, its rate is not.
        // The early cumulative figure spikes, so it must not be shown.
        let early: [String: JSONValue] = [
            "decode_tok_s": .number(210.0),
            "decode_phase": .string("reasoning"),
            "phase_tok_s": .null,
        ]
        XCTAssertNil(MTPLXBackendStore.headlineDecodeTPS(values: early, live: true))
    }

    func testFrameFromAnOlderServerStillUsesTheCumulativeRate() {
        let old: [String: JSONValue] = ["decode_tok_s": .number(52.0)]
        XCTAssertEqual(MTPLXBackendStore.headlineDecodeTPS(values: old, live: true), 52.0)
        XCTAssertNil(MTPLXBackendStore.headlineDecodeTPS(values: [:], live: true))
        let invalid: [String: JSONValue] = [
            "decode_tok_s": .number(0),
            "phase_tok_s": .number(-3),
        ]
        XCTAssertNil(MTPLXBackendStore.headlineDecodeTPS(values: invalid, live: true))
    }

    @MainActor
    func testGaugeShowsThePhaseRateLiveAndHoldsTheReceiptRateAfterwards() async throws {
        let port = try Self.freeTCPPort()
        let script = try makeExecutable(
            named: "fake-phase-rate-metrics",
            body: """
            #!/bin/sh
            exec python3 -u - <<'PY'
            import json
            import time
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            PORT = \(port)

            def write_event(writer, event, payload):
                writer.write((f"event: {event}\\n"
                              f"data: {json.dumps(payload)}\\n\\n").encode("utf-8"))
                writer.flush()

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    return

                def do_GET(self):
                    if self.path.startswith("/v1/mtplx/metrics/stream"):
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.end_headers()
                        write_event(self.wfile, "progress", {
                            "kind": "progress",
                            "request_id": "r-phase",
                            "progress": {
                                "request_id": "r-phase",
                                "completion_tokens": 3000,
                                "decode_tok_s": 41.0,
                                "decode_phase": "answer",
                                "phase_tok_s": 72.0,
                            },
                        })
                        time.sleep(1.0)
                        write_event(self.wfile, "completed", {
                            "kind": "completed",
                            "envelope": {
                                "request_id": "r-phase",
                                "completion_tokens": 3400,
                                "decode_tok_s": 44.0,
                            },
                        })
                        time.sleep(1.0)
                    else:
                        self.send_response(404)
                        self.end_headers()

            ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
            PY
            """
        )
        let process = Process()
        process.executableURL = script
        try process.run()
        defer { process.terminate() }

        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(port: port),
            settingsStore: MTPLXSettingsStore(
                settingsURL: temporaryDirectory().appendingPathComponent("settings.json")
            )
        )
        backend.startMetricsStream()

        var liveValue: Double?
        var heldValue: Double?
        let deadline = Date().addingTimeInterval(8)
        while Date() < deadline, heldValue == nil {
            switch backend.headlineDecode {
            case .live(let value):
                liveValue = value
            case .held(let value, _):
                heldValue = value
            case .absent:
                break
            }
            try await Task.sleep(for: .milliseconds(25))
        }
        XCTAssertEqual(liveValue ?? -1, 72.0, accuracy: 0.01)
        XCTAssertEqual(heldValue ?? -1, 44.0, accuracy: 0.01)
    }

    // MARK: Helpers

    private func temporaryDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-phase-rate-tests-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func makeExecutable(named name: String, body: String) throws -> URL {
        let url = temporaryDirectory().appendingPathComponent(name)
        try body.write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        return url
    }

    private static func freeTCPPort() throws -> Int {
        let socketFD = socket(AF_INET, SOCK_STREAM, 0)
        guard socketFD >= 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .ENOTSUP)
        }
        defer { Darwin.close(socketFD) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(0).bigEndian
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
        var bindAddress = address
        let bindResult = withUnsafePointer(to: &bindAddress) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(socketFD, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bindResult == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .ENOTSUP)
        }
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        var bound = sockaddr_in()
        let nameResult = withUnsafeMutablePointer(to: &bound) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(socketFD, $0, &length)
            }
        }
        guard nameResult == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .ENOTSUP)
        }
        return Int(UInt16(bigEndian: bound.sin_port))
    }
}
