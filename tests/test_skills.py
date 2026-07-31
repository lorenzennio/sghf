from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills"
SKILL = SKILLS / "sghf"
REFERENCES = SKILL / "references"

# Manifests that advertise the skill directory to a specific agent framework.
# Every one must point at the same payload; the skill has one source of truth.
SKILL_PATH_MANIFESTS = (
    ROOT / ".claude-plugin" / "plugin.json",
    ROOT / ".codex-plugin" / "plugin.json",
    ROOT / ".cursor-plugin" / "plugin.json",
)
MARKETPLACE_MANIFESTS = (
    ROOT / ".claude-plugin" / "marketplace.json",
    ROOT / ".agents" / "plugins" / "marketplace.json",
)


def _frontmatter(path: Path) -> tuple[dict[str, str], str]:
    """Split a skill document into its YAML frontmatter and body.

    Parsed by hand rather than with PyYAML: the frontmatter is flat
    ``key: value`` pairs and SGHF must not gain a dependency for a test.
    """

    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{path} must open with YAML frontmatter"
    _, block, body = text.split("---\n", 2)
    fields = {}
    for line in block.splitlines():
        if not line.strip():
            continue
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields, body


def test_skill_directory_layout() -> None:
    assert (SKILL / "SKILL.md").is_file()
    assert REFERENCES.is_dir()
    assert sorted(path.name for path in REFERENCES.glob("*.md")) == [
        "api.md",
        "pitfalls.md",
        "validation.md",
        "workflows.md",
    ]


def test_skill_frontmatter_is_valid() -> None:
    fields, body = _frontmatter(SKILL / "SKILL.md")

    assert re.fullmatch(r"[a-z0-9-]+", fields["name"]), fields["name"]
    assert fields["name"] == SKILL.name
    # The runtime injects the frontmatter into the system prompt.
    raw = (SKILL / "SKILL.md").read_text(encoding="utf-8").split("---\n")[1]
    assert len(raw) <= 1024, f"frontmatter is {len(raw)} characters"
    # Descriptions state triggering conditions only. A description that
    # summarises the workflow gets followed instead of the skill body.
    assert fields["description"].startswith("Use when"), fields["description"]
    assert body.strip()


def test_skill_only_references_files_that_exist() -> None:
    documents = [SKILL / "SKILL.md", *REFERENCES.glob("*.md")]
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            resolved = (document.parent / target.split("#", 1)[0]).resolve()
            assert resolved.exists(), f"{document.name}: missing {target}"


@pytest.mark.parametrize(
    "manifest", SKILL_PATH_MANIFESTS, ids=lambda path: path.parent.name
)
def test_runtime_manifest_points_at_the_one_skill_directory(
    manifest: Path,
) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["name"] == "sghf"
    for field in ("version", "description", "author", "license"):
        assert payload.get(field), f"{manifest} lacks {field}"
    resolved = (ROOT / payload["skills"]).resolve()
    assert resolved == SKILLS.resolve(), f"{manifest} points at {resolved}"


@pytest.mark.parametrize(
    "manifest", MARKETPLACE_MANIFESTS, ids=lambda path: str(path.parent.name)
)
def test_marketplace_manifest_is_valid(manifest: Path) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["name"] == "sghf"
    entries = payload["plugins"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["name"] == "sghf"
    source = entry["source"]
    location = source if isinstance(source, str) else source["url"]
    assert (ROOT / location).resolve() == ROOT.resolve()


def test_manifest_versions_track_the_package() -> None:
    import sghf

    for manifest in SKILL_PATH_MANIFESTS:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        assert payload["version"] == sghf.__version__, manifest


# --- installer -------------------------------------------------------------


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A throwaway HOME so the installer cannot touch the real one."""

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    return tmp_path


def _install(*argv: str) -> int:
    from sghf import _install_skills

    return _install_skills.main(list(argv))


def test_installer_writes_canonical_copy_and_runtime_symlink(home, capsys):
    (home / ".claude").mkdir()  # pretend Claude Code is installed

    assert _install() == 0

    canonical = home / ".agents" / "skills" / "sghf"
    assert (canonical / "SKILL.md").is_file()
    assert (canonical / "references" / "api.md").is_file()
    assert not canonical.is_symlink(), "canonical payload must be a real copy"

    link = home / ".claude" / "skills" / "sghf"
    assert link.is_symlink()
    # Relative, so moving HOME does not break it.
    assert not Path(os.readlink(link)).is_absolute()
    assert link.resolve() == canonical.resolve()


def test_installer_skips_runtimes_that_are_not_present(home):
    assert _install() == 0

    assert (home / ".agents" / "skills" / "sghf" / "SKILL.md").is_file()
    assert not (home / ".claude").exists()


def test_installer_refuses_to_overwrite_then_accepts_force(home, capsys):
    assert _install() == 0
    canonical = home / ".agents" / "skills" / "sghf"
    (canonical / "SKILL.md").write_text("clobbered", encoding="utf-8")

    assert _install() == 1
    assert "--force" in capsys.readouterr().err
    assert (canonical / "SKILL.md").read_text(encoding="utf-8") == "clobbered"

    assert _install("--force") == 0
    assert (canonical / "SKILL.md").read_text(encoding="utf-8") != "clobbered"


def test_installer_relinks_an_existing_correct_symlink(home):
    (home / ".claude").mkdir()
    assert _install() == 0

    # Re-running with --force must stay idempotent, not error on its own link.
    assert _install("--force") == 0
    link = home / ".claude" / "skills" / "sghf"
    assert link.resolve() == (home / ".agents" / "skills" / "sghf").resolve()


def test_installer_link_mode_does_not_copy(home):
    assert _install("--link") == 0

    canonical = home / ".agents" / "skills" / "sghf"
    assert canonical.is_symlink()
    assert (canonical / "SKILL.md").is_file()


def test_installer_list_reports_without_writing(home, capsys):
    assert _install("--list") == 0

    assert not (home / ".agents").exists()
    output = capsys.readouterr().out
    assert "payload" in output and "canonical" in output


def test_installer_honours_dest_override(home, tmp_path):
    elsewhere = tmp_path / "custom"

    assert _install("--dest", str(elsewhere)) == 0

    assert (elsewhere / "sghf" / "SKILL.md").is_file()
    assert not (home / ".agents").exists()
