import SwiftUI
import MTPLXAppCore

struct RequestsTab: View {
    @EnvironmentObject private var backend: MTPLXBackendStore

    @State private var cancellingId: String? = nil

    var body: some View {
        let inFlight = backend.inFlight
        let recent = backend.snapshot?.recent ?? []

        Group {
            if backend.daemonState.kind == .stopped {
                EmptyStateView(
                    symbol: "tray.2",
                    title: tr("Request log offline"),
                    message: tr("Start the daemon to see in-flight and recent requests.")
                ) {
                    Task { await backend.startDaemon() }
                }
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        inFlightCard(requests: inFlight)
                        recentRequestsCard(recent: recent)
                    }
                    .padding(20)
                }
            }
        }
        .navigationTitle(tr("Requests"))
    }

    @ViewBuilder
    private func inFlightCard(requests: [InFlightRequest]) -> some View {
        Card(tr("In Flight"), subtitle: requests.isEmpty ? tr("No active requests.") : tr("%lld active", requests.count)) {
            if requests.isEmpty {
                PillBadge(text: tr("idle"), systemImage: "moon.stars", tint: .secondary)
            }
        } content: {
            if requests.isEmpty {
                Text(tr("In-flight requests appear here in real time. Each row carries a Cancel button that flips the worker's cancel_event on a best-effort basis."))
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, minHeight: 80, alignment: .leading)
            } else {
                VStack(spacing: 8) {
                    ForEach(requests) { request in
                        inFlightRow(request)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func inFlightRow(_ request: InFlightRequest) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Image(systemName: request.cancelled ? "stop.circle.fill" : "circle.fill")
                    .foregroundStyle(request.cancelled ? Color.mtplxDanger : Color.mtplxSuccess)
                    .font(.caption2)
                Text(request.shortId)
                    .font(.system(.callout, design: .monospaced).weight(.semibold))
                Text(tr("· %@ ago", Format.duration(request.ageS)))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                if let session = request.sessionId {
                    PillBadge(text: tr("session %@", String(session.prefix(8))),
                              systemImage: "person.crop.circle",
                              tint: .accentColor)
                }
                if let prefill = request.prefillState, prefill.isActive {
                    PillBadge(text: tr("PREFILL %@", Format.percent(prefill.progress, fractionDigits: 0)),
                              systemImage: "gauge.with.dots.needle.bottom.50percent",
                              tint: .mtplxWarning,
                              emphasized: true)
                }
                Spacer()
                if backend.capabilities?.features["request_cancel"] != false {
                    Button {
                        cancellingId = request.requestId
                        Task {
                            defer { Task { @MainActor in cancellingId = nil } }
                            try? await backend.cancel(requestId: request.requestId)
                        }
                    } label: {
                        if cancellingId == request.requestId {
                            ProgressView().controlSize(.mini)
                        } else {
                            Label(tr("Cancel"), systemImage: "xmark.circle")
                        }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(request.cancelled || cancellingId == request.requestId)
                }
            }
            Text(request.promptDigest)
                .font(.callout)
                .foregroundStyle(.secondary)
                .lineLimit(2)
            if let prefill = request.prefillState, prefill.isActive,
               let reread = prefill.reread, reread.explainsAReread {
                RereadExplanationText(
                    reread: reread,
                    ssd: SSDLowDiskNotice.from(backend.health?.ssdSessionCache)
                )
            }
            HStack(spacing: 14) {
                Label(tr("%@ prompt tok", Format.integer(request.promptTokens)),
                      systemImage: "text.alignleft")
                    .labelStyle(.titleAndIcon)
                    .font(.caption2)
                if let model = request.model {
                    Label(model,
                          systemImage: "cube.box")
                        .labelStyle(.titleAndIcon)
                        .font(.caption2)
                        .lineLimit(1)
                }
            }
            .foregroundStyle(.tertiary)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Brand.raisedSurface)
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .strokeBorder(Brand.separator, lineWidth: 0.5)
                )
        )
    }

    @ViewBuilder
    private func recentRequestsCard(recent: [MetricsLatest]) -> some View {
        Card(tr("Recent"), subtitle: recent.isEmpty ? tr("Nothing recorded yet.") : tr("%lld requests", recent.count)) {
            if recent.isEmpty {
                EmptyView()
            } else {
                PillBadge(text: tr("live updates"), systemImage: "antenna.radiowaves.left.and.right", tint: .mtplxSuccess)
            }
        } content: {
            if recent.isEmpty {
                Text(tr("Completed requests will appear here as the daemon serves traffic."))
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, minHeight: 80, alignment: .leading)
            } else {
                VStack(spacing: 0) {
                    headerRow
                    Divider()
                    ForEach(Array(recent.enumerated()), id: \.offset) { _, metric in
                        recentRow(metric)
                        Divider().opacity(0.3)
                    }
                }
            }
        }
    }

    private var headerRow: some View {
        HStack {
            Text(tr("Session")).frame(width: 120, alignment: .leading)
            Text(tr("Decode")).frame(width: 80, alignment: .trailing)
            Text(tr("Prefill")).frame(width: 80, alignment: .trailing)
            Text(tr("TTFT")).frame(width: 70, alignment: .trailing)
            Text(tr("Tokens")).frame(width: 70, alignment: .trailing)
            Text(tr("Cache")).frame(width: 60, alignment: .trailing)
            Spacer()
        }
        .font(.system(size: 10, weight: .semibold))
        .tracking(0.6)
        .foregroundStyle(.tertiary)
        .padding(.bottom, 4)
    }

    private func recentRow(_ metric: MetricsLatest) -> some View {
        HStack {
            Text(metric.sessionId.map { String($0.prefix(12)) } ?? "—")
                .font(.system(.caption, design: .monospaced))
                .frame(width: 120, alignment: .leading)
                .lineLimit(1)
            Text(Format.tps(metric.decodeTokS))
                .font(.system(.caption, design: .rounded))
                .monospacedDigit()
                .frame(width: 80, alignment: .trailing)
            Text(Format.tps(metric.prefillTokS))
                .font(.system(.caption, design: .rounded))
                .monospacedDigit()
                .foregroundStyle(.secondary)
                .frame(width: 80, alignment: .trailing)
            Text(Format.duration(metric.ttftS))
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 70, alignment: .trailing)
            Text(Format.integer(metric.generatedTokens ?? metric.completionTokens))
                .font(.system(.caption, design: .rounded))
                .monospacedDigit()
                .frame(width: 70, alignment: .trailing)
            Text(metric.sessionCacheHit == true ? "hit" : "miss")
                .font(.caption2)
                .foregroundStyle(metric.sessionCacheHit == true ? Color.mtplxSuccess : .secondary)
                .frame(width: 60, alignment: .trailing)
            Spacer()
        }
        .padding(.vertical, 4)
    }
}
