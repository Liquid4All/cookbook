import Foundation
@testable import VerizonSupportPOC

enum KBFixtures {
    static let restartRouter = KBEntry(
        id: "restart-router",
        topic: "Restart Router",
        aliases: ["Turn off and on again router", "reboot router"],
        category: "Troubleshooting",
        answer: "Steps to restart your router.",
        deepLinks: [DeepLink(label: "Restart Router", url: "telco://restart-router")],
        tags: ["router", "restart"],
        requiresToolExecution: true
    )

    static let changePassword = KBEntry(
        id: "change-wifi-password",
        topic: "Change Wi-Fi Password",
        aliases: ["update wifi password"],
        category: "Password",
        answer: "Steps to change your Wi-Fi password.",
        deepLinks: [DeepLink(label: "Network", url: "telco://network")],
        tags: ["wifi", "password"],
        requiresToolExecution: false
    )

    static let weakSignal = KBEntry(
        id: "weak-signal-troubleshoot",
        topic: "Troubleshoot Weak Wi-Fi Signal",
        aliases: ["why is my signal weak", "bad wifi signal"],
        category: "Troubleshooting",
        answer: "Common causes and fixes for weak Wi-Fi.",
        deepLinks: [],
        tags: ["signal", "troubleshoot"],
        requiresToolExecution: false
    )

    static let speedTest = KBEntry(
        id: "router-speed-test",
        topic: "Router Speed Test",
        aliases: ["Speed Test", "Internet Speed"],
        category: "Speed",
        answer: "Steps to run a speed test.",
        deepLinks: [DeepLink(label: "Speed Test", url: "telco://speed-test")],
        tags: ["speed", "test"],
        requiresToolExecution: true
    )

    static let all: [KBEntry] = [restartRouter, changePassword, weakSignal, speedTest]
}
