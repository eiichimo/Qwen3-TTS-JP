# Security Notes

## Scope and Operating Assumption

- This repository is intended for local/non-public use.
- Internet-facing service operation is out of scope.

## Accepted Risk (Documented, Not Remediated)

- The Gradio demo default bind address remains `0.0.0.0` for local network convenience.
  - Related code: `qwen_tts/cli/demo.py` (`--ip` default)
  - Operational rule: do not expose this process directly to the public Internet.
  - Recommended usage: set `--ip 127.0.0.1` unless LAN sharing is explicitly needed.

## Remediated Risks

- Restricted remote audio URL loading:
  - Block private/loopback/link-local hosts by default.
  - Add URL fetch timeout and maximum download size guard.
  - Allow explicit override only via `QWEN_TTS_ALLOW_PRIVATE_URLS=1`.
- Reduced GitHub Actions supply-chain risk:
  - Pin external actions to commit SHAs.
  - Minimize job permissions.
  - Add Dependabot updates for `pip` and `github-actions`.

## Additional Guidance

- Keep branch protection enabled on `main`.
- Review Dependabot PRs regularly.
- If Internet-facing deployment becomes required, add authentication and a reverse proxy with network restrictions.
