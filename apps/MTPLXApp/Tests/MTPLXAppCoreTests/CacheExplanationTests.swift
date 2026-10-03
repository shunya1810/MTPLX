import XCTest

@testable import MTPLXAppCore

/// Why a prompt is read again is said in plain words while the wait happens,
/// every cache or frontier miss code has a sentence (never the code), and a
/// low-disk banner follows the SSD tier's `/health` stats.
///
/// On 2026-09-29 each screenshot turn of a Pi session re-read 123,000 to
/// 138,000 tokens with only a token counter on screen, and the Activity tab
/// then read "miss · ssd_prefix_miss".
final class CacheExplanationTests: XCTestCase {
    override func setUp() {
        super.setUp()
        L10n.resetBundleCache()
        L10n.activate(.english)
    }

    override func tearDown() {
        L10n.activate(.english)
        super.tearDown()
    }

    private func line(_ reread: PrefillReread, ssd: SSDLowDiskNotice? = nil) -> String {
        CacheExplanation.rereadLine(reread, ssd: ssd)
    }

    // MARK: The re-read, one case per restore lane

    func testAPromptThatExtendsItsHistoryReadsOnlyItsNewTokens() {
        let ramHit = PrefillReread(
            promptTokens: 130, historyMatchedTokens: 100, restorePointTokens: 100,
            recomputeTokens: 30, source: "ram"
        )
        XCTAssertFalse(ramHit.explainsAReread)
        XCTAssertEqual(line(ramHit), "Reading 30 new tokens")
    }

    func testACheckpointReplayNamesTheSavedStateItResumesFrom() {
        let replay = PrefillReread(
            promptTokens: 130, historyMatchedTokens: 120, restorePointTokens: 96,
            recomputeTokens: 34, source: "ram", resumeLimit: "saved_state", resumeLimitAtToken: 96,
            etaS: 130
        )
        XCTAssertTrue(replay.explainsAReread)
        XCTAssertEqual(
            line(replay),
            "Resuming from the saved state at token 96 · Re-reading 34 tokens, about 2m 10s"
        )
    }

    func testAnSSDRestoreSaysHowMuchCameFromTheSSD() {
        let restore = PrefillReread(
            promptTokens: 530, historyMatchedTokens: 500, restorePointTokens: 500,
            recomputeTokens: 30, source: "ssd", etaS: 0.4
        )
        XCTAssertEqual(
            line(restore),
            "Restored 500 tokens from the SSD cache · Reading 30 new tokens, about 1s"
        )
    }

    func testAChangedScreenshotAndAnUnchangedOneTheRestoreStoppedBefore() {
        let changed = PrefillReread(
            recomputeTokens: 25, source: "ram", cause: "screenshot_changed", causeAtToken: 113_467
        )
        XCTAssertEqual(
            line(changed),
            "The screenshot at token \(CacheExplanation.tokens(113_467)) changed · Re-reading 25 tokens"
        )
        let before = PrefillReread(
            recomputeTokens: 70, source: "ram", cause: "history_changed", causeAtToken: 407,
            resumeLimit: "screenshot", resumeLimitAtToken: 357
        )
        XCTAssertEqual(
            line(before),
            "History changed at token 407 · Resuming before the screenshot at token 357 · Re-reading 70 tokens"
        )
    }

    func testAColdStartAndEveryCauseHaveASentence() {
        let cold = PrefillReread(
            promptTokens: 900, restorePointTokens: 0, recomputeTokens: 900, source: "none",
            cause: "new_conversation"
        )
        XCTAssertEqual(line(cold), "New conversation or first turn since MTPLX started · Re-reading 900 tokens")
        let causes = [
            "history_changed", "screenshot_changed", "new_conversation", "not_cached",
            "switched_conversation", "freed_for_memory", "too_large", "evicted",
            "settings_changed", "cache_off",
        ]
        for cause in causes {
            let sentences = CacheExplanation.rereadSentences(
                PrefillReread(recomputeTokens: 10, cause: cause, causeAtToken: 5)
            )
            XCTAssertEqual(sentences.count, 2, cause)
            XCTAssertFalse(sentences[0].contains("_"), "\(cause) shows a code: \(sentences[0])")
        }
    }

    func testACauseThisAppDoesNotKnowShowsTheNumbersNotTheCode() {
        let future = PrefillReread(recomputeTokens: 40, cause: "some_future_cause", causeAtToken: 3)
        XCTAssertEqual(line(future), "Re-reading 40 tokens")
    }

    func testAnSSDThatCouldNotKeepACopySaysWhatTheSSDIsDoing() {
        let freed = PrefillReread(
            recomputeTokens: 900, source: "none", cause: "freed_for_memory", ssdChecked: true
        )
        XCTAssertEqual(
            line(freed, ssd: SSDLowDiskNotice(severity: .full)),
            "Saved state was freed to relieve memory pressure · The SSD cache is not saving right now: the disk is almost full · Re-reading 900 tokens"
        )
        XCTAssertTrue(line(freed, ssd: SSDLowDiskNotice(severity: .low)).contains("free disk space is low"))
        // A restore that came from the SSD, or no SSD trouble: nothing added.
        var fromSSD = freed
        fromSSD.source = "ssd"
        XCTAssertFalse(line(fromSSD, ssd: SSDLowDiskNotice(severity: .full)).contains("almost full"))
        XCTAssertFalse(line(freed).contains("SSD"))
    }

    // MARK: Miss codes

    /// Every code the server can send today (session bank, RAM lane, SSD
    /// tier, request bypasses, agent frontier).
    static let serverMissCodes = [
        "new_session", "prefix_divergence_at_token", "model_mismatch", "template_mismatch",
        "policy_mismatch", "evicted", "background_bypass", "session_busy", "snapshot_desync",
        "no_snapshot_coverage", "oversized_snapshot_skipped",
        "block_prefix_disabled", "no_gdn_boundaries", "below_block_min_match:512",
        "ssd_cache_off", "ssd_cache_write_only", "ssd_empty_lookup", "ssd_restore_error:OSError",
        "ssd_prefix_miss", "ssd_prefix_not_better_than_ram", "ssd_prefix_shadowed_by_resident_duplicate",
        "ssd_format_mismatch", "ssd_mtp_epoch_mismatch", "ssd_payload_missing",
        "legacy_ssd_cache_archived", "ssd_missing_mtp_generation_state", "ssd_missing_mtp_history",
        "vision_request_cache_bypass", "opencode_tool_history_cache_bypass", "request_cache_bypass",
        "ar_batch_nonmergeable_history_cache", "ar_batch_full_prefix_not_insertable",
        "ar_batch_vision_keyed_entry", "mtp_batch_cold_prefill",
        "miss_no_tool_result", "miss_no_assistant_tool_frontier", "miss_unknown_tool_id",
        "miss_live_frontier_not_armed", "miss_template_changed", "miss_policy_changed",
        "miss_model_changed", "miss_cache_evicted", "miss_snapshot_desync",
        "miss_live_frontier_consumed_or_missing", "miss_prompt_prefix_changed",
        "miss_wrong_session_or_no_prior_frontier", "miss_session_busy", "miss_ssd_prefix_miss",
    ]

    func testEveryMissCodeHasItsOwnPlainSentence() {
        let general = CacheExplanation.missReason("miss_unknown")
        XCTAssertEqual(general, "The cache could not be used for this request")
        for code in Self.serverMissCodes {
            let sentence = CacheExplanation.missReason(code)
            XCTAssertNotNil(sentence, code)
            XCTAssertNotEqual(sentence, general, "\(code) fell through to the general sentence")
            XCTAssertFalse(sentence?.contains("_") ?? true, "\(code) shows a code: \(sentence ?? "")")
        }
        XCTAssertEqual(
            CacheExplanation.missReason("ssd_prefix_miss"),
            "No saved state for this conversation in memory or on the SSD"
        )
        XCTAssertEqual(
            CacheExplanation.missReason("miss_prompt_prefix_changed"),
            CacheExplanation.missReason("prefix_divergence_at_token")
        )
    }

    func testNoCodeIsNoSentenceAndAnUnknownCodeIsNeverShown() {
        XCTAssertNil(CacheExplanation.missReason(nil))
        XCTAssertNil(CacheExplanation.missReason("  "))
        XCTAssertEqual(
            CacheExplanation.missReason("brand_new_reason"),
            "The cache could not be used for this request"
        )
    }

    func testMissSentencesAreTranslated() {
        L10n.activate(.german)
        XCTAssertEqual(CacheExplanation.missReason("ssd_cache_off"), "Der SSD-Cache ist ausgeschaltet")
        XCTAssertEqual(
            line(PrefillReread(recomputeTokens: 12, cause: "history_changed", causeAtToken: 7)),
            "Der Verlauf hat sich bei Token 7 geändert · 12 Token werden erneut gelesen"
        )
    }

    // MARK: Decoding

    func testPrefillStateCarriesTheRereadAndToleratesOddFields() throws {
        let json = """
        {"phase": "chunk", "tokens_total": 130, "tokens_done": 96,
         "reread": {"prompt_tokens": 130, "history_matched_tokens": 120,
                    "restore_point_tokens": 96, "recompute_tokens": 34, "source": "ram",
                    "cause": 7, "resume_limit": "saved_state", "resume_limit_at_token": 96,
                    "eta_s": null, "text": "Resuming from the saved state at token 96."}}
        """
        let state = try JSONDecoder().decode(PrefillState.self, from: Data(json.utf8))
        let reread = try XCTUnwrap(state.reread)
        XCTAssertNil(reread.cause, "a wrong-typed field is dropped, not fatal")
        XCTAssertEqual(reread.restorePointTokens, 96)
        XCTAssertEqual(reread.resumeLimit, "saved_state")
        XCTAssertNil(reread.etaS)
        let older = try JSONDecoder().decode(
            PrefillState.self, from: Data(#"{"phase": "chunk", "tokens_total": 5}"#.utf8)
        )
        XCTAssertNil(older.reread)
        XCTAssertEqual(
            PrefillReread(values: ["cause": .string("evicted"), "recompute_tokens": .number(9)]),
            PrefillReread(recomputeTokens: 9, cause: "evicted")
        )
    }

    // MARK: Low-disk banner

    private func health(_ values: [String: JSONValue]) -> SSDSessionCacheHealth {
        SSDSessionCacheHealth(values: values)
    }

    func testTheBannerFollowsTheDiskState() {
        XCTAssertNil(SSDLowDiskNotice.from(nil))
        XCTAssertNil(SSDLowDiskNotice.from(health(["enabled": .bool(true), "disk_state": .string("ok")])))
        XCTAssertNil(SSDLowDiskNotice.from(health(["enabled": .bool(false), "disk_state": .string("full")])))

        let gib = 1_073_741_824.0
        let low = SSDLowDiskNotice.from(health([
            "enabled": .bool(true), "disk_state": .string("low"), "low_disk": .bool(true),
            "disk_free_bytes": .number(25 * gib), "disk_floor_bytes": .number(10 * gib),
            "largest_session_bytes": .number(4 * gib),
        ]))
        XCTAssertEqual(low?.severity, .low)
        XCTAssertEqual(low?.title, "Low disk space for the SSD cache")
        XCTAssertEqual(
            low?.message,
            "Free disk space is 25.0 GB. The SSD cache needs room for two copies of your largest conversation (4.0 GB) while it saves a new one, so it may skip saving new progress. Copies it already saved are kept."
        )

        let full = SSDLowDiskNotice.from(health([
            "enabled": .bool(true), "disk_state": .string("full"),
            "disk_free_bytes": .number(8 * gib), "disk_floor_bytes": .number(10 * gib),
        ]))
        XCTAssertEqual(full?.severity, .full)
        XCTAssertEqual(full?.title, "SSD cache paused: disk almost full")
        XCTAssertEqual(
            full?.message,
            "Free disk space is 8.0 GB, below the 10.0 GB the SSD cache always leaves free, so it has stopped saving conversations. Free up disk space to turn it back on."
        )
    }

    func testTheBannerReadsOlderTiersAndOddTypesDefensively() {
        // An older tier: only the last write's verdict.
        let legacy = SSDLowDiskNotice.from(health(["enabled": .bool(true), "low_disk_writes_disabled": .bool(true)]))
        XCTAssertEqual(legacy?.severity, .full)
        XCTAssertEqual(
            legacy?.message,
            "Free disk space is below the floor the SSD cache always leaves free, so it has stopped saving conversations. Free up disk space to turn it back on."
        )
        let lowOnly = SSDLowDiskNotice.from(health(["low_disk": .bool(true)]))
        XCTAssertEqual(lowOnly?.severity, .low)
        XCTAssertEqual(
            lowOnly?.message,
            "Free disk space is low. The SSD cache may skip saving new progress. Copies it already saved are kept."
        )
        XCTAssertNil(SSDLowDiskNotice.from(health([
            "disk_state": .number(3), "low_disk": .string("yes"), "low_disk_writes_disabled": .bool(false),
        ])))
    }

    func testHealthSurvivesAnUnexpectedSSDFieldType() throws {
        // A tier field of an unexpected type must not make /health
        // undecodable (the watchdog then counts a live daemon's answers as
        // unusable).
        let json = """
        {"ok": true, "model": "m", "model_path": "/m", "generation_mode": "mtp",
         "load_mtp": true, "mtp_enabled": true, "depth": 3, "profile": {},
         "context_window": 262144, "active_requests": 0, "reasoning_parser": "qwen3",
         "ssd_session_cache": {"enabled": true, "entries": "many", "disk_state": "low",
                               "disk_free_bytes": 26843545600, "largest_session_bytes": 4294967296}}
        """
        let health = try JSONDecoder().decode(HealthPayload.self, from: Data(json.utf8))
        XCTAssertTrue(health.ok)
    }

    func testHealthCarriesTheSSDStatsForTheBanner() throws {
        let json = """
        {"ok": true, "model": "m", "model_path": "/m", "generation_mode": "mtp",
         "load_mtp": true, "mtp_enabled": true, "depth": 3, "profile": {},
         "context_window": 262144, "active_requests": 0, "reasoning_parser": "qwen3",
         "ssd_session_cache": {"enabled": true, "disk_state": "full",
                               "disk_free_bytes": 8589934592, "disk_floor_bytes": 10737418240}}
        """
        let health = try JSONDecoder().decode(HealthPayload.self, from: Data(json.utf8))
        XCTAssertEqual(SSDLowDiskNotice.from(health.ssdSessionCache)?.severity, .full)
    }
}
