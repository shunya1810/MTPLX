import Foundation

/// A client config file the app could not read. The file is left exactly as
/// it was and nothing is written: the user fixes or moves it.
public struct ClientConfigFileError: Error, Equatable, CustomStringConvertible, LocalizedError {
    public let path: String
    public let reason: String

    public init(path: String, reason: String) {
        self.path = path
        self.reason = reason
    }

    public var description: String {
        tr(
            "MTPLX left %@ unchanged because it could not be read (%@). Fix or move that file, then start again.",
            path,
            reason
        )
    }

    public var errorDescription: String? { description }
}

/// Pi's `models.json` and OpenCode's `opencode.json` read the way those
/// apps read them: JSON that may carry `//` and `/* */` comments, trailing
/// commas and a byte order mark. A file the client accepts is merged, never
/// replaced; one nobody can read is reported and left alone. SYNC: the CLI
/// writers read the same files with mtplx/jsonc.py.
enum ClientConfigFile {
    /// The file's top-level object, or nil when there is no file or it is
    /// empty. Throws `ClientConfigFileError` for anything else that does not
    /// parse to an object.
    static func readObject(at url: URL) throws -> [String: JSONValue]? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw ClientConfigFileError(path: url.path, reason: error.localizedDescription)
        }
        guard !data.isEmpty else { return nil }
        do {
            return try parseObject(data)
        } catch {
            throw ClientConfigFileError(path: url.path, reason: reason(for: error))
        }
    }

    static func parseObject(_ data: Data) throws -> [String: JSONValue] {
        let decoder = JSONDecoder()
        if let object = try? decoder.decode([String: JSONValue].self, from: data) {
            return object
        }
        return try decoder.decode(
            [String: JSONValue].self,
            from: Data(blankingCommentsAndTrailingCommas(Array(data)))
        )
    }

    /// Replaces comments and trailing commas outside strings with spaces and
    /// drops a leading byte order mark. Newlines stay, so a parse error still
    /// names the line it is on. Every byte this touches is ASCII, and UTF-8
    /// continuation bytes never are, so working on bytes is exact.
    static func blankingCommentsAndTrailingCommas(_ input: [UInt8]) -> [UInt8] {
        let bytes = input.starts(with: [0xEF, 0xBB, 0xBF]) ? Array(input.dropFirst(3)) : input
        let newline: UInt8 = 0x0A, quote: UInt8 = 0x22, backslash: UInt8 = 0x5C
        let slash: UInt8 = 0x2F, star: UInt8 = 0x2A, comma: UInt8 = 0x2C, space: UInt8 = 0x20
        let count = bytes.count

        var uncommented = bytes
        var index = 0
        var inString = false
        while index < count {
            let byte = bytes[index]
            if inString {
                if byte == backslash {
                    index += 2
                    continue
                }
                if byte == quote { inString = false }
                index += 1
                continue
            }
            if byte == quote {
                inString = true
            } else if byte == slash, index + 1 < count, bytes[index + 1] == slash {
                while index < count, bytes[index] != newline {
                    uncommented[index] = space
                    index += 1
                }
                continue
            } else if byte == slash, index + 1 < count, bytes[index + 1] == star {
                var end = index + 2
                while end + 1 < count, !(bytes[end] == star && bytes[end + 1] == slash) {
                    end += 1
                }
                end = end + 1 < count ? end + 2 : count
                for position in index..<end where bytes[position] != newline {
                    uncommented[position] = space
                }
                index = end
                continue
            }
            index += 1
        }

        var result = uncommented
        index = 0
        inString = false
        while index < count {
            let byte = uncommented[index]
            if inString {
                if byte == backslash {
                    index += 2
                    continue
                }
                if byte == quote { inString = false }
            } else if byte == quote {
                inString = true
            } else if byte == comma {
                var next = index + 1
                while next < count, [space, 0x09, 0x0D, newline].contains(uncommented[next]) {
                    next += 1
                }
                if next < count, uncommented[next] == 0x7D || uncommented[next] == 0x5D {
                    result[index] = space
                }
            }
            index += 1
        }
        return result
    }

    private static func reason(for error: Error) -> String {
        let detail: String
        switch error {
        case DecodingError.typeMismatch:
            detail = "the top-level value is not an object"
        case DecodingError.dataCorrupted(let context):
            let underlying = (context.underlyingError as NSError?)?
                .userInfo[NSDebugDescriptionErrorKey] as? String
            detail = underlying ?? context.debugDescription
        default:
            detail = String(describing: error)
        }
        return detail.trimmingCharacters(in: CharacterSet(charactersIn: ". "))
    }
}
