from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REQUIRED_PAGES = {
    "index.md",
    "simplified-likelihood.md",
    "workspace-compilation.md",
    "fitting.md",
    "hypothesis-testing.md",
    "toy-inference.md",
    "upper-limits.md",
    "api-reference.md",
    "validation-and-applicability.md",
    "references.md",
}
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")


def test_required_documentation_pages_exist_and_are_linked():
    assert REQUIRED_PAGES <= {path.name for path in DOCS.glob("*.md")}
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    for page in REQUIRED_PAGES - {"index.md"}:
        assert f"]({page})" in index


def test_relative_markdown_links_resolve():
    files = [ROOT / "README.md", *(DOCS / name for name in REQUIRED_PAGES)]
    for source in files:
        text = source.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (source.parent / target).resolve()
            assert resolved.exists(), f"{source}: missing link target {raw_target}"
