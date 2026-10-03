import Foundation

/// Why a request reads part of its prompt again: the server's prefill event
/// field `reread` (mtplx/prefill_plan.py). The event that starts the replay,
/// or the cold read, brings it; every later prefill event of the request
/// carries it, and the request's receipt keeps it.
///
/// On 2026-09-29 each screenshot turn of a Pi session re-read 123,000 to
/// 138,000 tokens while the app showed only a token counter. The app turns
/// these facts into sentences in the user's language
/// (`CacheExplanation.rereadSentences`); it never shows the codes.
///
/// Decoded leniently: a field of an unexpected type is dropped and never
/// fails the prefill state or the in-flight snapshot that holds it.
public struct PrefillReread: Codable, Equatable, Sendable {
    public var promptTokens: Int?
    /// How much of the prompt matches this conversation's cached history.
    public var historyMatchedTokens: Int?
    /// The token the running state resumes at (0 for a cold read).
    public var restorePointTokens: Int?
    public var recomputeTokens: Int?
    /// Where the resumed state came from: "ram", "ssd" or "none".
    public var source: String?
    /// Why the cache did not cover the prompt; nil when the prompt only
    /// extends the saved history.
    public var cause: String?
    public var causeAtToken: Int?
    /// Why the state resumes before the end of the matched history.
    public var resumeLimit: String?
    public var resumeLimitAtToken: Int?
    /// The SSD tier was consulted for this conversation.
    public var ssdChecked: Bool?
    /// The server's estimate from its measured prefill rate; nil before any
    /// prefill was measured.
    public var etaS: Double?

    public init(
        promptTokens: Int? = nil,
        historyMatchedTokens: Int? = nil,
        restorePointTokens: Int? = nil,
        recomputeTokens: Int? = nil,
        source: String? = nil,
        cause: String? = nil,
        causeAtToken: Int? = nil,
        resumeLimit: String? = nil,
        resumeLimitAtToken: Int? = nil,
        ssdChecked: Bool? = nil,
        etaS: Double? = nil
    ) {
        self.promptTokens = promptTokens
        self.historyMatchedTokens = historyMatchedTokens
        self.restorePointTokens = restorePointTokens
        self.recomputeTokens = recomputeTokens
        self.source = source
        self.cause = cause
        self.causeAtToken = causeAtToken
        self.resumeLimit = resumeLimit
        self.resumeLimitAtToken = resumeLimitAtToken
        self.ssdChecked = ssdChecked
        self.etaS = etaS
    }

    /// From an untyped prefill payload (the SSE stream's `prefill` frames).
    public init(values: [String: JSONValue]) {
        func int(_ key: CodingKeys) -> Int? { values[key.rawValue]?.intValue }
        func string(_ key: CodingKeys) -> String? {
            guard let text = values[key.rawValue]?.stringValue, !text.isEmpty else { return nil }
            return text
        }
        self.init(
            promptTokens: int(.promptTokens),
            historyMatchedTokens: int(.historyMatchedTokens),
            restorePointTokens: int(.restorePointTokens),
            recomputeTokens: int(.recomputeTokens),
            source: string(.source),
            cause: string(.cause),
            causeAtToken: int(.causeAtToken),
            resumeLimit: string(.resumeLimit),
            resumeLimitAtToken: int(.resumeLimitAtToken),
            ssdChecked: values[CodingKeys.ssdChecked.rawValue]?.boolValue,
            etaS: values[CodingKeys.etaS.rawValue]?.doubleValue
        )
    }

    public init(from decoder: Decoder) throws {
        let values = (try? decoder.singleValueContainer().decode([String: JSONValue].self)) ?? [:]
        self.init(values: values)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(promptTokens, forKey: .promptTokens)
        try container.encodeIfPresent(historyMatchedTokens, forKey: .historyMatchedTokens)
        try container.encodeIfPresent(restorePointTokens, forKey: .restorePointTokens)
        try container.encodeIfPresent(recomputeTokens, forKey: .recomputeTokens)
        try container.encodeIfPresent(source, forKey: .source)
        try container.encodeIfPresent(cause, forKey: .cause)
        try container.encodeIfPresent(causeAtToken, forKey: .causeAtToken)
        try container.encodeIfPresent(resumeLimit, forKey: .resumeLimit)
        try container.encodeIfPresent(resumeLimitAtToken, forKey: .resumeLimitAtToken)
        try container.encodeIfPresent(ssdChecked, forKey: .ssdChecked)
        try container.encodeIfPresent(etaS, forKey: .etaS)
    }

    /// Matched history is computed again (a cause or a resume limit says
    /// why). A prompt that only extends its saved history reads its new
    /// tokens and needs no explanation on the gauge.
    public var explainsAReread: Bool { cause != nil || resumeLimit != nil }

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case historyMatchedTokens = "history_matched_tokens"
        case restorePointTokens = "restore_point_tokens"
        case recomputeTokens = "recompute_tokens"
        case source
        case cause
        case causeAtToken = "cause_at_token"
        case resumeLimit = "resume_limit"
        case resumeLimitAtToken = "resume_limit_at_token"
        case ssdChecked = "ssd_checked"
        case etaS = "eta_s"
    }
}
