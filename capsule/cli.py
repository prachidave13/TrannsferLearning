"""Command-line entry point."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .build import build_capsule, make_resume_prompt, update_capsule
from .schema import SELF_SUMMARIZE_PROMPT


def _read(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def _write(path: str, content: str) -> None:
    if path == "-":
        sys.stdout.write(content)
        return
    Path(path).write_text(content, encoding="utf-8")
    print(f"wrote {path} ({len(content)} bytes)", file=sys.stderr)


def _resolve_mode(args) -> str:
    if getattr(args, "llm", False):
        return "llm"
    if getattr(args, "skeleton", False):
        return "skeleton"
    return "algorithmic"


def cmd_build(args: argparse.Namespace) -> int:
    transcript = _read(args.transcript)
    capsule = build_capsule(
        transcript,
        mode=_resolve_mode(args),
        provider=args.provider,
        model=args.model,
    )
    _write(args.output, capsule)
    if args.resume:
        _write(args.resume, make_resume_prompt(args.output, capsule))
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    old = _read(args.capsule)
    new = _read(args.transcript)
    merged = update_capsule(
        old,
        new,
        mode=_resolve_mode(args),
        provider=args.provider,
        model=args.model,
    )
    _write(args.output, merged)
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    _write(args.output, make_resume_prompt(args.capsule))
    return 0


def cmd_self_prompt(_: argparse.Namespace) -> int:
    """Print the prompt to paste into the dying chat."""
    sys.stdout.write(SELF_SUMMARIZE_PROMPT)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="capsule",
        description="Transfer learning for LLM chats — build a structured handoff.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    common_llm = argparse.ArgumentParser(add_help=False)
    mode_group = common_llm.add_mutually_exclusive_group()
    mode_group.add_argument("--llm", action="store_true",
                            help="Use LLM (OpenAI/Anthropic). Requires API key.")
    mode_group.add_argument("--skeleton", action="store_true",
                            help="Empty schema + regex hints only.")
    common_llm.add_argument("--provider", choices=["openai", "anthropic"],
                            help="Force LLM provider (with --llm).")
    common_llm.add_argument("--model", help="Override default model.")

    pb = sub.add_parser("build", parents=[common_llm],
                        help="Build a capsule from a chat transcript.")
    pb.add_argument("transcript", help="Path to transcript (or - for stdin).")
    pb.add_argument("-o", "--output", default="capsule.md",
                    help="Output capsule path. Default: capsule.md")
    pb.add_argument("--resume", default="resume.txt",
                    help="Also write a resume prompt to this path. "
                         "Pass empty string to skip.")
    pb.set_defaults(func=cmd_build)

    pu = sub.add_parser("update", parents=[common_llm],
                        help="Merge a follow-up transcript into an existing capsule.")
    pu.add_argument("capsule", help="Path to existing capsule.")
    pu.add_argument("transcript", help="Path to new transcript.")
    pu.add_argument("-o", "--output", default="capsule.md",
                    help="Output path (overwrites by default).")
    pu.set_defaults(func=cmd_update)

    pr = sub.add_parser("resume", help="Generate the paste-into-new-chat prompt.")
    pr.add_argument("capsule", help="Path to capsule (referenced in prompt).")
    pr.add_argument("-o", "--output", default="-", help="Output path. Default: stdout.")
    pr.set_defaults(func=cmd_resume)

    ps = sub.add_parser(
        "self-prompt",
        help="Print a prompt to paste into the DYING chat to self-summarize."
    )
    ps.set_defaults(func=cmd_self_prompt)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
