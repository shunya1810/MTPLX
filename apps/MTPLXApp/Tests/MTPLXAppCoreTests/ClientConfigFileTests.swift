import XCTest

@testable import MTPLXAppCore

/// Pi and OpenCode accept comments and trailing commas in their config
/// files. The app read them as strict JSON and, on any parse error, moved the
/// file aside and wrote MTPLX's provider alone: a valid commented file lost
/// every other provider and the user's own window pair. Every fixture lives
/// in a directory each test creates for itself.
final class ClientConfigFileTests: XCTestCase {
    private let qwen36 = "/models/Qwen3.6-27B-MTPLX-Optimized-Speed"

    // MARK: Reader

    func testReaderAcceptsWhatTheClientsAccept() throws {
        let text = """
        \u{FEFF}{
          // Pi and OpenCode both read this.
          "providers": {
            "other": {"baseUrl": "http://example.invalid/v1", "models": [{"id": "o"},],},
          },
          /* block comment */ "note": "a // and /* inside a string stay", "quote": "say \\"hi\\" // still a string",
        }
        """
        let object = try ClientConfigFile.parseObject(Data(text.utf8))
        XCTAssertEqual(
            object["providers"]?.objectValue?["other"]?.objectValue?["models"]?.arrayValue?.count,
            1
        )
        XCTAssertEqual(object["note"], .string("a // and /* inside a string stay"))
        XCTAssertEqual(object["quote"], .string("say \"hi\" // still a string"))
    }

    func testReaderRejectsWhatNoClientAccepts() {
        for text in ["{bad json", "{'single': 'quotes'}", "{\"a\": 1 \"b\": 2}", "[1, 2]"] {
            XCTAssertThrowsError(try ClientConfigFile.parseObject(Data(text.utf8)), text)
        }
    }

    // MARK: Pi

    func testCommentedPiConfigKeepsOtherProvidersAndTheUserCap() throws {
        let directory = try makeTemporaryDirectory()
        let url = directory.appendingPathComponent("models.json")
        let integration = PiIntegration(configURL: url)
        let configuration = MTPLXAppConfiguration(model: qwen36, port: 8000, contextWindow: 131_072)
        _ = try integration.sync(configuration: configuration)

        // The user caps answers at 65,536, adds a provider of their own and a
        // comment, and leaves a trailing comma: all of it valid for Pi 0.85.1.
        var root = try decodeObject(at: url)
        var providers = try XCTUnwrap(root["providers"]?.objectValue)
        var own = try XCTUnwrap(providers["mtplx"]?.objectValue)
        var entries = try XCTUnwrap(own["models"]?.arrayValue)
        var entry = try XCTUnwrap(entries.first?.objectValue)
        entry["maxTokens"] = .number(65_536)
        entries[0] = .object(entry)
        own["models"] = .array(entries)
        providers["mtplx"] = .object(own)
        providers["other"] = .object([
            "baseUrl": .string("http://example.invalid/v1"),
            "models": .array([.object(["id": .string("their-model")])]),
        ])
        root["providers"] = .object(providers)
        let json = String(decoding: try JSONEncoder().encode(root), as: UTF8.self)
        let original = "// My Pi models.\n" + json.dropLast() + ",}\n"
        try original.write(to: url, atomically: true, encoding: .utf8)

        // Nothing to change: the file stays exactly as written, comment
        // included, before and after the daemon publishes its window.
        for served in [nil, ServedExecutionWindow(tokens: 98_304)] {
            let result = try integration.sync(configuration: configuration, servedWindow: served)
            XCTAssertNil(result.backupPath)
            XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), original)
        }

        // A change of port rewrites the file: the user's provider and cap
        // survive, and the original text is kept beside it.
        let moved = MTPLXAppConfiguration(model: qwen36, port: 8001, contextWindow: 131_072)
        let result = try integration.sync(configuration: moved)
        let after = try decodeObject(at: url)
        let afterProviders = try XCTUnwrap(after["providers"]?.objectValue)
        XCTAssertNotNil(afterProviders["other"])
        let model = try XCTUnwrap(
            afterProviders["mtplx"]?.objectValue?["models"]?.arrayValue?.first?.objectValue
        )
        XCTAssertEqual(model["maxTokens"], .number(65_536))
        XCTAssertEqual(afterProviders["mtplx"]?.objectValue?["baseUrl"], .string("http://127.0.0.1:8001/v1"))
        let backup = try XCTUnwrap(result.backupPath)
        XCTAssertEqual(try String(contentsOfFile: backup, encoding: .utf8), original)
    }

    func testUnreadablePiConfigIsLeftUntouchedAndReported() throws {
        let directory = try makeTemporaryDirectory()
        let url = directory.appendingPathComponent("models.json")
        let broken = "{\"providers\": {\"other\": {}}, oops\n"
        try broken.write(to: url, atomically: true, encoding: .utf8)

        XCTAssertThrowsError(
            try PiIntegration(configURL: url).sync(
                configuration: MTPLXAppConfiguration(model: qwen36, port: 8000, contextWindow: 131_072)
            )
        ) { error in
            print("Unreadable Pi config: \(error)")
            let failure = error as? ClientConfigFileError
            XCTAssertEqual(failure?.path, url.path)
            XCTAssertFalse(failure?.reason.isEmpty ?? true)
            XCTAssertTrue(String(describing: error).contains(url.path))
        }
        XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), broken)
        XCTAssertEqual(try directoryEntries(directory), ["models.json"])
    }

    // MARK: OpenCode

    func testCommentedOpenCodeConfigKeepsOtherProvidersAndSettings() throws {
        let directory = try makeTemporaryDirectory()
        let url = directory.appendingPathComponent("opencode.json")
        let original = """
        {
          // OpenCode's own example style: comments and trailing commas.
          "$schema": "https://opencode.ai/config.json",
          "theme": "tokyonight",
          "provider": {
            "other": {"name": "Other", "options": {"baseURL": "http://example.invalid/v1"},},
          },
        }

        """
        try original.write(to: url, atomically: true, encoding: .utf8)
        let integration = OpenCodeIntegration(
            configURL: url,
            desktopSettingsStoreURL: directory.appendingPathComponent("default.dat")
        )
        let result = try integration.sync(
            configuration: MTPLXAppConfiguration(model: qwen36, port: 8000, contextWindow: 65_536)
        )

        let root = try decodeObject(at: url)
        XCTAssertEqual(root["theme"], .string("tokyonight"))
        XCTAssertNotNil(root["provider"]?.objectValue?["other"])
        XCTAssertNotNil(root["provider"]?.objectValue?["mtplx"])
        let backup = try XCTUnwrap(result.backupPath)
        XCTAssertEqual(try String(contentsOfFile: backup, encoding: .utf8), original)

        // The next launch finds nothing to change and leaves the file alone.
        let written = try String(contentsOf: url, encoding: .utf8)
        XCTAssertNil(
            try integration.sync(
                configuration: MTPLXAppConfiguration(model: qwen36, port: 8000, contextWindow: 65_536)
            ).backupPath
        )
        XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), written)
    }

    func testUnreadableOpenCodeConfigIsLeftUntouchedAndReported() throws {
        let directory = try makeTemporaryDirectory()
        let url = directory.appendingPathComponent("opencode.json")
        let broken = "{\n  \"provider\": {\n    \"x\": }\n}\n"
        try broken.write(to: url, atomically: true, encoding: .utf8)

        XCTAssertThrowsError(
            try OpenCodeIntegration(
                configURL: url,
                desktopSettingsStoreURL: directory.appendingPathComponent("default.dat")
            ).sync(configuration: MTPLXAppConfiguration(model: qwen36, port: 8000, contextWindow: 65_536))
        ) { error in
            XCTAssertEqual((error as? ClientConfigFileError)?.path, url.path)
        }
        XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), broken)
        XCTAssertEqual(try directoryEntries(directory), ["opencode.json"])
    }

    // MARK: Helpers

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-client-config-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func decodeObject(at url: URL) throws -> [String: JSONValue] {
        try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: url))
    }

    private func directoryEntries(_ url: URL) throws -> [String] {
        try FileManager.default.contentsOfDirectory(atPath: url.path).sorted()
    }
}

private extension JSONValue {
    var objectValue: [String: JSONValue]? {
        if case .object(let value) = self { return value }
        return nil
    }

    var arrayValue: [JSONValue]? {
        if case .array(let value) = self { return value }
        return nil
    }
}
