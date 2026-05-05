import SwiftUI
import UIKit

struct ChatInputBar: View {
    @Binding var text: String
    let isProcessing: Bool
    let attachedImage: UIImage?
    let isListening: Bool
    let isTranscribing: Bool
    let listeningPartial: String
    /// Non-nil when the last voice attempt failed (permissions denied,
    /// session refused, etc.). Shown inline so "nothing happens" stops
    /// being the only feedback on failure. Cleared when the user taps
    /// the mic again or types.
    let voiceError: String?
    let onSend: () -> Void
    let onMicTap: () -> Void
    let onCameraTap: () -> Void
    let onClearAttachment: () -> Void

    @Environment(\.brand) private var brand

    /// Drives the pulsing recording dot in the listening strip. Local
    /// to this view — no upstream dependency.
    @State private var pulseOn: Bool = false

    var body: some View {
        VStack(spacing: 6) {
            if let image = attachedImage {
                attachmentChip(image)
            }
            if isListening || isTranscribing {
                listeningStrip
            }
            if let err = voiceError, !isListening && !isTranscribing {
                voiceErrorStrip(err)
            }
            HStack(spacing: 8) {
                cameraButton
                TextField(brand.chatPlaceholder, text: $text, axis: .vertical)
                    .lineLimit(1...4)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(brand.surfaceElevated, in: RoundedRectangle(cornerRadius: brand.bubbleCornerRadius))
                    .submitLabel(.send)
                    .onSubmit { if canSend { onSend() } }
                micButton
                sendButton
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(
            Rectangle()
                .fill(brand.surfaceBackground)
                .shadow(color: .black.opacity(0.06), radius: 8, y: -2)
        )
    }

    private var canSend: Bool {
        !isProcessing && (!text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || attachedImage != nil)
    }

    private func attachmentChip(_ image: UIImage) -> some View {
        HStack(spacing: 8) {
            Image(uiImage: image)
                .resizable()
                .scaledToFill()
                .frame(width: 36, height: 36)
                .clipShape(RoundedRectangle(cornerRadius: 6))
            Text("Photo attached — will analyze on-device")
                .font(.caption)
                .foregroundStyle(brand.textSecondary)
            Spacer()
            Button(action: onClearAttachment) {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(brand.textSecondary)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(brand.surfaceElevated, in: RoundedRectangle(cornerRadius: 10))
    }

    private var listeningStrip: some View {
        HStack(alignment: .center, spacing: 10) {
            // Pulsing red dot — unambiguous "we are recording" cue.
            // Matches the iOS system convention (Messages, Voice Memos)
            // so the affordance is instantly recognizable.
            Circle()
                .fill(brand.danger)
                .frame(width: 10, height: 10)
                .opacity(pulseOn ? 1.0 : 0.35)
                .animation(
                    .easeInOut(duration: 0.8).repeatForever(autoreverses: true),
                    value: pulseOn
                )
                .onAppear { pulseOn = true }
                .onDisappear { pulseOn = false }

            VStack(alignment: .leading, spacing: 2) {
                Text(voiceStatusText)
                    .font(.callout)
                    .foregroundStyle(brand.textPrimary)
                    .lineLimit(3)
                    .fixedSize(horizontal: false, vertical: true)
                if !listeningPartial.isEmpty || isTranscribing {
                    Text("On-device · nothing leaves your phone")
                        .font(.caption2)
                        .foregroundStyle(brand.textSecondary)
                }
            }

            Spacer()

            if isListening {
                // Full-size Stop button — primary-filled so it reads as
                // the main action on this strip.
                Button(action: onMicTap) {
                    HStack(spacing: 6) {
                        Image(systemName: "stop.fill")
                        Text("Stop")
                    }
                    .font(.subheadline).fontWeight(.semibold)
                    .padding(.horizontal, 14).padding(.vertical, 8)
                    .background(brand.primary, in: Capsule())
                    .foregroundStyle(brand.onPrimary)
                }
                .accessibilityLabel("Stop recording and place text in input field")
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(brand.primary.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(brand.primary.opacity(0.25), lineWidth: 1)
        )
    }

    private var voiceStatusText: String {
        if isTranscribing {
            return listeningPartial.isEmpty ? "Transcribing locally…" : listeningPartial
        }
        return listeningPartial.isEmpty ? "Listening… tap Stop when done" : listeningPartial
    }

    private var cameraButton: some View {
        Button(action: onCameraTap) {
            Image(systemName: "camera")
                .font(.system(size: 22))
                .foregroundStyle(brand.textSecondary)
                .frame(width: 36, height: 36)
        }
        .accessibilityLabel("Attach photo")
    }

    private func voiceErrorStrip(_ message: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(brand.danger)
            Text(message)
                .font(.caption)
                .foregroundStyle(brand.textPrimary)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(brand.danger.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
    }

    private var micButton: some View {
        Button(action: onMicTap) {
            Image(systemName: isListening ? "mic.fill" : "mic")
                .font(.system(size: 22))
                .foregroundStyle(micForeground)
                .frame(width: 36, height: 36)
        }
        .disabled(isTranscribing)
        .accessibilityLabel(isListening ? "Stop recording" : "Voice input")
    }

    private var micForeground: Color {
        if isListening { return brand.primary }
        if isProcessing || isTranscribing { return brand.textSecondary.opacity(0.35) }
        return brand.textSecondary
    }

    private var sendButton: some View {
        Button(action: onSend) {
            Image(systemName: "arrow.up")
                .font(.system(size: 16, weight: .bold))
                .foregroundStyle(canSend ? brand.onPrimary : brand.textSecondary.opacity(0.5))
                .frame(width: 36, height: 36)
                .background(canSend ? brand.primary : brand.surfaceElevated, in: Circle())
        }
        .disabled(!canSend)
        .accessibilityLabel("Send message")
    }
}
