# Forking Telco Triage

This cookbook folder is already self-contained. You can copy
`examples/telco-triage-ios/` into a new repository if you want a customer- or
carrier-specific fork.

## What To Copy

```bash
cp -R examples/telco-triage-ios ./my-telco-triage
cd my-telco-triage
git init
git add .
git commit -m "initial Telco Triage import"
```

Do not copy generated `.build`, `.swiftpm`, `DerivedData`, or GGUF model files
unless you intentionally use Git LFS.

## Optional Product Rename

The default project, target, module, bundle ID, and display name are already
generic Telco Triage values. For a carrier-specific fork, rename only the
customer-facing pieces first: display name, bundle ID, theme, knowledge base,
and tool catalog.

If you also need a customer-specific Swift module, rename it deliberately:

1. Rename `TelcoTriage/` and `TelcoTriageTests/`.
2. Replace target, scheme, source paths, test host, and `@testable import`
   values in `project.yml` and tests.
3. Run `xcodegen generate`.
4. Run the fast test pass from `TESTING.md`.

## Models

Large GGUFs are not committed to the cookbook. For a fork, choose one:

- Keep the current `bootstrap-models.sh` workflow and set `TELCO_MODELS_DIR`.
- Store GGUFs in Git LFS.
- Download GGUFs from an internal release or model registry during CI.

Required runtime artifacts are listed in `README.md`.

## Customization Checklist

1. Replace `TelcoTriage/Resources/knowledge-base.json`.
2. Add carrier branding in `TelcoTriage/Core/Branding/`.
3. Register carrier-specific tools in `ToolRegistry`.
4. Update support taxonomy labels and retrain the shared classifier adapter if
   the carrier workflow differs materially.
5. Keep the router deterministic: model signals in, typed policy decision out.
