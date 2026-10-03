import SwiftUI
import MTPLXAppCore

struct CacheTab: View {
    @EnvironmentObject private var backend: MTPLXBackendStore

    @State private var pendingClearAll = false
    @State private var clearingAll = false
    @State private var clearingSession: String? = nil

    var body: some View {
        let sessions = backend.sessions
        let sessionBank = backend.sessionBank ?? sessions?.sessionBank

        Group {
            if backend.daemonState.kind == .stopped {
                EmptyStateView(
                    symbol: "tray.full",
                    title: tr("Cache is empty"),
                    message: tr("Start a model to see what's been cached.")
                ) {
                    Task { await backend.startDaemon() }
                }
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        summaryCard(sessions: sessions, sessionBank: sessionBank)
                        bankCard(sessionBank: sessionBank)
                        sessionsListCard(sessions: sessions)
                    }
                    .padding(20)
                }
            }
        }
        .navigationTitle(tr("Cache"))
        .confirmationDialog(
            tr("Clear all cache entries?"),
            isPresented: $pendingClearAll
        ) {
            Button(tr("Clear All"), role: .destructive) {
                clearingAll = true
                Task {
                    defer { Task { @MainActor in clearingAll = false } }
                    try? await backend.clearCache()
                }
            }
            Button(tr("Cancel"), role: .cancel) {}
        } message: {
            Text(tr("Clears every saved prompt cache. Anything mid-flight keeps running. Next chats will start from scratch."))
        }
    }

    @ViewBuilder
    private func summaryCard(sessions: SessionsPayload?, sessionBank: SessionBank?) -> some View {
        Card(tr("Cache Summary")) {
            HStack(spacing: 24) {
                StatTile(
                    label: tr("Sessions"),
                    value: Format.integer(sessions?.count ?? sessions?.sessions.count),
                    systemImage: "person.2.fill",
                    tint: .accentColor
                )
                Divider().frame(height: 36)
                StatTile(
                    label: tr("Bank size"),
                    value: Format.bytes(sessionBank?.totalNbytes),
                    systemImage: "tray.full",
                    tint: .secondary
                )
                Divider().frame(height: 36)
                StatTile(
                    label: tr("Bank capacity"),
                    value: Format.integer(sessionBank?.maxEntries),
                    systemImage: "square.stack.3d.up",
                    tint: .secondary
                )
                Divider().frame(height: 36)
                StatTile(
                    label: tr("Last miss"),
                    value: CacheExplanation.missReason(sessionBank?.lastMissReason) ?? "—",
                    systemImage: "questionmark.circle",
                    tint: (sessionBank?.lastMissReason).flatMap { $0.isEmpty ? nil : $0 } == nil
                        ? .secondary : .mtplxWarning
                )
            }
        }
    }

    @ViewBuilder
    private func bankCard(sessionBank: SessionBank?) -> some View {
        Card(tr("SessionBank"), subtitle: tr("Block-prefix reuse across requests and restarts.")) {
            if backend.capabilities?.features["cache_clear"] != false {
                Button {
                    pendingClearAll = true
                } label: {
                    Label(tr("Clear All"), systemImage: "trash")
                        .font(.caption.weight(.medium))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(clearingAll)
            }
        } content: {
            if let prefixes = sessionBank?.prefixes, !prefixes.isEmpty {
                LazyVGrid(
                    columns: [
                        GridItem(.adaptive(minimum: 220, maximum: 280), spacing: 12)
                    ],
                    spacing: 12
                ) {
                    ForEach(prefixes) { prefix in
                        prefixTile(prefix: prefix)
                    }
                }
            } else {
                Text(tr("Cached chats will show up here as you use the app."))
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, minHeight: 100)
            }
            if let evictions = sessionBank?.evictionLog, !evictions.isEmpty {
                Divider().padding(.vertical, 4)
                MicroHeader(tr("Recent evictions"), systemImage: "tray.and.arrow.up")
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(Array(evictions.prefix(6).enumerated()), id: \.offset) { _, eviction in
                        HStack(spacing: 8) {
                            Image(systemName: "minus.circle")
                                .foregroundStyle(Color.mtplxWarning)
                                .font(.caption2)
                            Text(eviction.string("reason") ?? "unknown")
                                .font(.caption)
                            Spacer()
                            if let bytes = eviction.int("bytes") {
                                Text(Format.bytes(bytes))
                                    .font(.caption2)
                                    .foregroundStyle(.tertiary)
                                    .monospacedDigit()
                            }
                            if let session = eviction.string("session_id") {
                                Text(String(session.prefix(8)))
                                    .font(.caption2)
                                    .foregroundStyle(.tertiary)
                                    .monospacedDigit()
                            }
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func prefixTile(prefix: SessionBankPrefix) -> some View {
        let age = max(0, Date().timeIntervalSince1970 - prefix.lastAccessS)
        let isHot = age < 60
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Circle()
                    .fill(isHot ? Color.mtplxSuccess : Color.gray)
                    .frame(width: 8, height: 8)
                Text(String(prefix.sessionId.prefix(12)))
                    .font(.system(.caption, design: .monospaced).weight(.medium))
                    .lineLimit(1)
                Spacer()
                Text(Format.relative(from: age))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            HStack(spacing: 10) {
                Label(tr("%@ tok", Format.integer(prefix.prefixLen)), systemImage: "text.alignleft")
                    .labelStyle(.titleAndIcon)
                    .font(.caption2)
                Label(Format.bytes(prefix.nbytes), systemImage: "internaldrive")
                    .labelStyle(.titleAndIcon)
                    .font(.caption2)
            }
            .foregroundStyle(.secondary)
            HStack {
                if prefix.hasLiveRef == true {
                    PillBadge(text: tr("live"), systemImage: "bolt.fill", tint: .accentColor)
                } else {
                    PillBadge(text: tr("cached"), systemImage: "checkmark", tint: .secondary)
                }
                Text(tr("%lld hits", prefix.hits))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Brand.raisedSurface)
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .strokeBorder(
                            isHot ? Brand.success.opacity(0.55) : Brand.separator,
                            lineWidth: 0.75
                        )
                )
        )
    }

    @ViewBuilder
    private func sessionsListCard(sessions: SessionsPayload?) -> some View {
        Card(tr("Engine Sessions")) {
            if let sessions, !sessions.sessions.isEmpty {
                VStack(spacing: 4) {
                    ForEach(sessions.sessions) { session in
                        sessionRow(session)
                        Divider().opacity(0.3)
                    }
                }
            } else {
                Text(tr("No active sessions."))
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, minHeight: 80)
            }
        }
    }

    @ViewBuilder
    private func sessionRow(_ session: SessionRow) -> some View {
        let canClearSession = backend.capabilities?.features["session_clear"] != false
        HStack {
            Image(systemName: "circle.fill")
                .font(.caption2)
                .foregroundStyle(session.inFlight == true ? Color.mtplxSuccess : .secondary)
            Text(session.sessionId)
                .font(.system(.callout, design: .monospaced))
                .lineLimit(1)
            Spacer()
            Label(tr("%@ tok", Format.integer(session.prefixLen)),
                  systemImage: "text.alignleft")
                .labelStyle(.titleAndIcon)
                .font(.caption)
                .foregroundStyle(.secondary)
            Label(Format.bytes(session.bytes),
                  systemImage: "internaldrive")
                .labelStyle(.titleAndIcon)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(Format.relative(from: session.ageSeconds))
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .frame(width: 64, alignment: .trailing)
            if canClearSession {
                Button {
                    clearingSession = session.sessionId
                    Task {
                        defer { Task { @MainActor in clearingSession = nil } }
                        try? await backend.clearSession(sessionId: session.sessionId)
                    }
                } label: {
                    if clearingSession == session.sessionId {
                        ProgressView().controlSize(.mini)
                    } else {
                        Image(systemName: "trash")
                    }
                }
                .buttonStyle(.borderless)
                .help(tr("Drop this session's cached prefix"))
                .accessibilityLabel(tr("Clear session cache"))
            }
        }
        .padding(.vertical, 4)
    }
}
