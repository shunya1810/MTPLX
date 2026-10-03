import Foundation

/// `/health` `ssd_session_cache`: the SSD tier's stats, read leniently.
///
/// The tier adds fields over time (the free-disk budget adds `disk_state`,
/// `low_disk`, `disk_free_bytes`, `disk_floor_bytes` and
/// `largest_session_bytes`; older tiers only report
/// `low_disk_writes_disabled`). A field of an unexpected type must never make
/// the whole `/health` payload undecodable: on 2026-07-06 one schema drift
/// did that and the watchdog counted a live daemon's answers as misses.
public struct SSDSessionCacheHealth: Codable, Equatable, Sendable {
    public var values: [String: JSONValue]

    public init(values: [String: JSONValue] = [:]) {
        self.values = values
    }

    public init(from decoder: Decoder) throws {
        values = (try? decoder.singleValueContainer().decode([String: JSONValue].self)) ?? [:]
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(values)
    }

    public var enabled: Bool? { values["enabled"]?.boolValue }
    /// "ok", "low" (two copies of the largest conversation no longer fit
    /// above the floor) or "full" (at or under the floor: no saves).
    public var diskState: String? { values["disk_state"]?.stringValue }
    public var lowDisk: Bool? { values["low_disk"]?.boolValue }
    /// The last write's verdict on older tiers: under the free-disk floor.
    public var lowDiskWritesDisabled: Bool? { values["low_disk_writes_disabled"]?.boolValue }
    public var diskFreeBytes: Int? { values["disk_free_bytes"]?.intValue }
    public var diskFloorBytes: Int? { values["disk_floor_bytes"]?.intValue }
    public var largestSessionBytes: Int? { values["largest_session_bytes"]?.intValue }
}

/// The low-disk banner's state, from the SSD tier's `/health` stats.
public struct SSDLowDiskNotice: Equatable, Sendable {
    public enum Severity: Equatable, Sendable {
        /// Saves may be skipped: two copies of the largest conversation no
        /// longer fit above the free-disk floor.
        case low
        /// Saves stopped: free disk is at or under the floor.
        case full
    }

    public var severity: Severity
    public var freeBytes: Int?
    public var floorBytes: Int?
    public var largestSessionBytes: Int?

    public init(
        severity: Severity,
        freeBytes: Int? = nil,
        floorBytes: Int? = nil,
        largestSessionBytes: Int? = nil
    ) {
        self.severity = severity
        self.freeBytes = freeBytes
        self.floorBytes = floorBytes
        self.largestSessionBytes = largestSessionBytes
    }

    /// nil when the SSD cache is off, the disk is fine, or the daemon does
    /// not report a disk state.
    public static func from(_ health: SSDSessionCacheHealth?) -> SSDLowDiskNotice? {
        guard let health, health.enabled != false else { return nil }
        let severity: Severity
        switch health.diskState?.lowercased() {
        case "full":
            severity = .full
        case "low":
            severity = .low
        case "ok":
            return nil
        default:
            // No disk state from this tier: fall back to its booleans.
            if health.lowDiskWritesDisabled == true {
                severity = .full
            } else if health.lowDisk == true {
                severity = .low
            } else {
                return nil
            }
        }
        return SSDLowDiskNotice(
            severity: severity,
            freeBytes: health.diskFreeBytes,
            floorBytes: health.diskFloorBytes,
            largestSessionBytes: health.largestSessionBytes
        )
    }

    public var title: String {
        switch severity {
        case .full: return tr("SSD cache paused: disk almost full")
        case .low: return tr("Low disk space for the SSD cache")
        }
    }

    public var message: String {
        switch severity {
        case .full:
            if let freeBytes, let floorBytes {
                return tr(
                    "Free disk space is %@, below the %@ the SSD cache always leaves free, so it has stopped saving conversations. Free up disk space to turn it back on.",
                    Self.gigabytes(freeBytes),
                    Self.gigabytes(floorBytes)
                )
            }
            return tr("Free disk space is below the floor the SSD cache always leaves free, so it has stopped saving conversations. Free up disk space to turn it back on.")
        case .low:
            if let freeBytes, let largestSessionBytes, largestSessionBytes > 0 {
                return tr(
                    "Free disk space is %@. The SSD cache needs room for two copies of your largest conversation (%@) while it saves a new one, so it may skip saving new progress. Copies it already saved are kept.",
                    Self.gigabytes(freeBytes),
                    Self.gigabytes(largestSessionBytes)
                )
            }
            return tr("Free disk space is low. The SSD cache may skip saving new progress. Copies it already saved are kept.")
        }
    }

    /// One decimal in GiB, the unit the tier's own messages use.
    static func gigabytes(_ bytes: Int) -> String {
        tr("%.1f GB", Double(max(0, bytes)) / 1_073_741_824.0)
    }
}
