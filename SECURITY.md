# Security Policy

## Supported Versions

The `main` branch is the only supported line. Tagged releases receive fixes
by advancing `main`, not by backporting.

## Reporting a Vulnerability

Please report security issues privately through
[GitHub private vulnerability reporting](https://github.com/al3obdi/Thaqafa-RepE/security/advisories/new)
rather than a public issue. You should receive a response within a week.

In scope, beyond ordinary code execution issues:

- **Credential leakage paths.** Nothing in this repository may log, print,
  serialise, or embed the `HF_TOKEN`. If you find a code path that can, that
  is a security bug even if it requires unusual configuration.
- **Result-integrity bypasses.** The results pipeline is designed so that
  fabricated or unverified numbers cannot reach the paper (provenance
  markers, refusal paths). A way around those guards is in scope.

## Handling Secrets

- Tokens live only in `.env` (git-ignored) or Space secrets, never in code,
  notebooks, or dataset files.
- `detect-private-key` runs in pre-commit; CI never prints environment
  variables.
