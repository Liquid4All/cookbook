# LocalCowork Packaging Research (Issue #52)

Notes from reviewing issue #52 and PR #51 on the Liquid4All/cookbook repo.

---

## Issue #52 and PR #51 Summary

**Issue #52** is a proposal by an external contributor (pierretokns) to distribute LocalCowork via Homebrew, lowering the installation barrier for macOS users.

**PR #51** is a prerequisite CI/CD workflow that builds LocalCowork as `.dmg` installers for arm64 and x86_64, triggered by `localcowork-v*` tags. It adds a single file: `.github/workflows/release-localcowork.yml`.

### PR #51 positives

- Clean matrix build for both architectures
- Rust cache via `Swatinem/rust-cache@v2` for faster CI
- Separate `build-macos` and `create-release` jobs
- SHA256 checksums generated correctly

### PR #51 issues

1. **Hard-coded personal Homebrew tap in release notes** -- the release body references `brew tap pierretokns/localcowork`, a personal community tap, inside what would become official Liquid AI release notes. This is the main blocker before merging.
2. **x86_64 builds on `macos-14` runner** -- `macos-14` is Apple Silicon only. Cross-compilation should work via the Rust target, but worth flagging.
3. **`cargo install tauri-cli` on every run** -- slow, not cached. Should use a pinned version or `cargo-binstall`.
4. **`permissions: contents: write` is too broad** -- set at top level, both jobs get write access. Only `create-release` needs it.
5. **Tauri updater signing vs. macOS Gatekeeper signing** -- the workflow handles Tauri's updater key (`TAURI_SIGNING_PRIVATE_KEY`) but not Apple Developer ID signing (`APPLE_CERTIFICATE`, `APPLE_TEAM_ID`), which is what Homebrew and SmartScreen require.

---

## macOS Distribution: Homebrew

### Option 1: Official org tap (recommended near-term)

Create a `Liquid4All/homebrew-tap` repository under the Liquid AI GitHub org. This is the standard pattern for companies distributing via Homebrew before or instead of the official cask registry (examples: `hashicorp/homebrew-tap`, `github/homebrew-gh`, `aws/homebrew-tap`).

Install command:
```bash
brew tap Liquid4All/tap
brew install --cask localcowork
```

The release notes in PR #51 should point to this instead of the personal tap. Until the official tap exists, the Homebrew section should be omitted from release notes entirely.

### Option 2: Official homebrew-cask (longer-term)

Submit to [Homebrew/homebrew-cask](https://github.com/Homebrew/homebrew-cask). Requires:
- App signed with an Apple Developer ID (Gatekeeper)
- Notability thresholds: 75+ stars, 30+ forks, 30+ watchers

### Apple Developer ID / Authenticode signing

To get an official Apple Developer ID certificate, Liquid AI would need to go through Apple's business verification process. This is not required for the org tap approach, but is required for official `brew.sh` inclusion.

---

## Windows Distribution: Scoop (recommended, no signing required)

Scoop is the simplest Windows path -- no code signing required, no SmartScreen issues at the package manager level.

**Authenticode (Windows code signing) is NOT required for Scoop.** It is required for Winget (the official Microsoft package manager) and recommended for Chocolatey.

### Scoop setup

1. Create a `Liquid4All/scoop-bucket` repo under the org
2. Add a JSON manifest:

```json
{
    "version": "0.1.1",
    "description": "On-device AI agent powered by LFM2-24B-A2B",
    "homepage": "https://github.com/Liquid4All/cookbook",
    "url": "https://github.com/Liquid4All/cookbook/releases/download/localcowork-v0.1.1/LocalCowork_0.1.1_x64-setup.exe",
    "hash": "<sha256>",
    "depends": ["extras/llama"],
    "installer": { "script": "Start-Process $dir\\LocalCowork_0.1.1_x64-setup.exe -Wait" }
}
```

3. Users install with:
```powershell
scoop bucket add liquid4all https://github.com/Liquid4All/scoop-bucket
scoop install localcowork
```

**Note:** Windows will still show a SmartScreen warning on first run since the binary is unsigned. This is user-level friction, not a distribution blocker.

### Winget (longer-term)

Winget is Microsoft's official package manager, pre-installed on Windows 11. Publishing to [microsoft/winget-pkgs](https://github.com/microsoft/winget-pkgs) requires signed binaries (Authenticode certificate) for SmartScreen trust.

### Authenticode certificate requirements (for future reference)

- Purchase an OV or EV certificate from a CA (DigiCert, GlobalSign, etc.)
- EV is recommended for new apps to get immediate SmartScreen trust
- CA will verify Liquid AI as a legal entity (business docs, phone, contact)
- Private key must be stored on an HSM or hardware token (YubiKey FIPS, etc.) -- soft keys no longer accepted
- As of February 15, 2026, certificates are capped at 459-day maximum validity

---

## Dependency handling (llama.cpp)

Both Homebrew and Scoop support automatic dependency installation:

- **Homebrew:** the contributor's tap already lists `llama.cpp` as a dependency. An official tap should do the same.
- **Scoop:** `llama.cpp` is available in the `extras` bucket. Declare it in the manifest's `depends` field.

This means `brew install --cask localcowork` or `scoop install localcowork` would pull in llama.cpp automatically.

---

## Hardware Requirements

From the PRD (section 3.2). The model (LFM2-24B-A2B Q4_K_M) requires approximately 14-16 GB of RAM/VRAM.

### Single model mode

| Platform | Minimum | Recommended |
|---|---|---|
| macOS | M1 Pro / 16 GB unified memory | M2 Pro+ / 32 GB unified memory |
| Windows (GPU) | RTX 3090 / 24 GB VRAM | RTX 4090 / 24 GB VRAM |
| Windows (CPU) | Ryzen AI 9 / 32 GB RAM | Ryzen AI 9 HX / 64 GB RAM |
| Windows (NPU) | Snapdragon X Elite | Snapdragon X Elite + 32 GB |

### Dual-model orchestrator mode

Requires ~14.5 GB VRAM (planner ~13 GB + router ~1.5 GB). Same hardware specs apply, but the minimum Mac and CPU-only configs become impractical.

---

## Linux

Linux is not officially supported. The PRD and README only mention macOS and Windows. Tauri supports Linux natively, so the app would likely compile -- but no requirements, testing, or support commitment exists yet.

Homebrew runs on Linux too, so if Linux support is added in the future, the same `Liquid4All/homebrew-tap` would cover all three platforms from a single tap.

---

## Verifying Windows builds from macOS

Options for a Mac-only team to verify Windows binaries:

1. **GitHub Actions (easiest):** Add a `windows-latest` CI job with a smoke test. No local Windows setup needed.
2. **Virtual machine:** UTM (free, Apple Silicon), VMware Fusion (free personal), or Parallels (paid). Use a free 90-day Windows 11 evaluation image from Microsoft.
3. **Community testers:** Ask Windows users (e.g., the commenter on issue #52) to test the installer and report back.

Recommended: CI for automated regression, community testers for real-world validation.

---

## Referencing PR #51 in issue comments

Use `#51` -- GitHub automatically links it to the PR.
