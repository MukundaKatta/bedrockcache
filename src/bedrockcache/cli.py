"""bedrockcache CLI: audit a request payload from a JSON file or stdin.

Usage:
    bedrockcache audit request.json --backend bedrock-converse
    cat request.json | bedrockcache audit --backend litellm
"""

from __future__ import annotations

import argparse
import json
import sys

from bedrockcache import Backend, audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bedrockcache")
    sub = parser.add_subparsers(dest="cmd", required=True)

    audit_p = sub.add_parser("audit", help="Audit a request JSON file or stdin")
    audit_p.add_argument("file", nargs="?", default="-",
                         help="path to JSON file, or '-' / omit to read stdin")
    audit_p.add_argument("--backend", required=True,
                         choices=[b.value for b in Backend],
                         help="backend shape of the request")
    audit_p.add_argument("--strict", action="store_true",
                         help="exit non-zero if caching will not apply")

    args = parser.parse_args(argv)

    if args.cmd == "audit":
        return _cmd_audit(args.file, args.backend, args.strict)

    parser.print_help()
    return 2


def _cmd_audit(path: str, backend: str, strict: bool) -> int:
    raw = sys.stdin.read() if path == "-" else open(path, "r").read()
    request = json.loads(raw)
    report = audit(request, Backend(backend))

    print(f"backend:           {report.backend.value}")
    print(f"will_cache:        {report.will_cache}")
    print(f"breakpoint_count:  {report.breakpoint_count}")
    if report.reasons:
        print("reasons:")
        for sev, msg in report.reasons:
            print(f"  [{sev}] {msg}")
    if report.recommendations:
        print("recommendations:")
        for r in report.recommendations:
            print(f"  - {r}")

    if strict and not report.will_cache:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
