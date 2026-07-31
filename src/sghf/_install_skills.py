"""Install the bundled SGHF agent skill for any agent framework.

Skills are written once to a canonical cross-runtime directory,
``~/.agents/skills/``, which Codex, Copilot CLI and Gemini CLI read directly.
Runtimes that keep their own directory, such as Claude Code's
``~/.claude/skills/``, receive a relative symlink pointing at that one copy, so
there is a single payload on disk no matter how many agent frameworks are
installed.

Run ``sghf-install-skills`` after ``pip install sghf``. A wheel cannot do this
itself: installing a wheel extracts files and does not execute code.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections.abc import Iterator
from importlib import resources
from pathlib import Path

SKILL_NAME = "sghf"


def _payload() -> Path:
    """Locate the bundled skill directory.

    Installed wheels carry it as package data; a source checkout keeps it at the
    repository root so the plugin manifests can point at the same files.
    """

    packaged = Path(str(resources.files("sghf"))) / "_skills" / SKILL_NAME
    if (packaged / "SKILL.md").is_file():
        return packaged
    checkout = (
        Path(__file__).resolve().parents[2] / "skills" / SKILL_NAME
    )
    if (checkout / "SKILL.md").is_file():
        return checkout
    raise FileNotFoundError(
        "Could not locate the bundled skill payload. Reinstall sghf, or run "
        "this from a source checkout containing skills/"
    )


def canonical_root() -> Path:
    """Cross-runtime skills directory that every framework can read."""

    return Path.home() / ".agents" / "skills"


def runtime_roots() -> Iterator[tuple[str, Path]]:
    """Yield ``(runtime, skills directory)`` for frameworks needing a symlink.

    Only frameworks that keep a private skills directory appear here. Codex,
    Copilot CLI and Gemini CLI read the canonical directory directly and so
    need no entry.
    """

    claude_config = os.environ.get("CLAUDE_CONFIG_DIR")
    base = Path(claude_config) if claude_config else Path.home() / ".claude"
    yield "claude-code", base / "skills"


def _link_target(link: Path, target: Path) -> str:
    """Relative path from ``link``'s directory to ``target``."""

    return os.path.relpath(target, link.parent)


def _install_payload(
    source: Path, destination: Path, *, link: bool, force: bool
) -> str:
    if destination.exists() or destination.is_symlink():
        if not force:
            raise FileExistsError(
                f"{destination} already exists; pass --force to replace it"
            )
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if link:
        destination.symlink_to(source, target_is_directory=True)
        return f"linked  {destination} -> {source}"
    shutil.copytree(source, destination)
    return f"copied  {destination}"


def _install_symlink(link: Path, target: Path, *, force: bool) -> str:
    relative = _link_target(link, target)
    if link.is_symlink():
        if os.readlink(link) == relative:
            return f"ok      {link} (already linked)"
        if not force:
            raise FileExistsError(
                f"{link} is a symlink to something else; pass --force"
            )
        link.unlink()
    elif link.exists():
        if not force:
            raise FileExistsError(
                f"{link} already exists; pass --force to replace it"
            )
        if link.is_dir():
            shutil.rmtree(link)
        else:
            link.unlink()
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(relative, target_is_directory=True)
    return f"linked  {link} -> {relative}"


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``sghf-install-skills`` console script."""

    parser = argparse.ArgumentParser(
        prog="sghf-install-skills",
        description=(
            "Install the SGHF agent skill into the cross-runtime skills "
            "directory and link it into each detected agent framework."
        ),
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="override the canonical skills directory "
        "(default: ~/.agents/skills)",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="symlink the payload instead of copying it; for development only, "
        "since it breaks if the environment is removed",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace anything already installed at the target paths",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="report the payload and detected runtimes without writing",
    )
    arguments = parser.parse_args(argv)

    try:
        source = _payload()
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    root = arguments.dest if arguments.dest is not None else canonical_root()
    destination = root / SKILL_NAME

    if arguments.list_only:
        print(f"payload   {source}")
        print(f"canonical {destination}")
        for runtime, skills_root in runtime_roots():
            marker = "present" if skills_root.parent.is_dir() else "absent"
            print(f"runtime   {runtime}: {skills_root / SKILL_NAME} ({marker})")
        return 0

    actions: list[str] = []
    try:
        actions.append(
            _install_payload(
                source, destination, link=arguments.link, force=arguments.force
            )
        )
        for _, skills_root in runtime_roots():
            if not skills_root.parent.is_dir():
                continue
            actions.append(
                _install_symlink(
                    skills_root / SKILL_NAME, destination, force=arguments.force
                )
            )
    except (FileExistsError, OSError) as error:
        for action in actions:
            print(action)
        print(f"error: {error}", file=sys.stderr)
        return 1

    for action in actions:
        print(action)
    print("\nSGHF skill installed. Restart your agent to pick it up.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
