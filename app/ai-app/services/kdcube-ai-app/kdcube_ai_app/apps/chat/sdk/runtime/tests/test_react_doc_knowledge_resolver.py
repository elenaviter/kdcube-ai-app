import importlib.util
import pathlib
import sys


def _load_resolver_module():
    module_name = "_test_react_doc_knowledge_resolver"
    resolver_path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "examples"
        / "bundles"
        / "react.doc@2026-03-02-22-10"
        / "knowledge"
        / "resolver.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, resolver_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_react_doc_resolver_uses_bundle_storage_dir_when_knowledge_root_is_unset(tmp_path, monkeypatch):
    resolver = _load_resolver_module()
    knowledge_root = tmp_path / "bundle-storage" / "tenant" / "project" / "react.doc__main"
    docs_root = knowledge_root / "docs"
    docs_root.mkdir(parents=True)
    (docs_root / "intro.md").write_text("# Intro\n", encoding="utf-8")
    (knowledge_root / "index.json").write_text(
        """
        {
          "items": [
            {
              "path": "ks:docs/intro.md",
              "title": "Intro Guide",
              "summary": "Knowledge root smoke test",
              "tags": ["guide"],
              "keywords": ["intro"]
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("BUNDLE_STORAGE_DIR", str(knowledge_root))
    resolver.KNOWLEDGE_ROOT = None

    hits = resolver.search_knowledge(query="intro")
    assert hits and hits[0]["path"] == "ks:docs/intro.md"

    doc = resolver.read_knowledge(path="ks:docs/intro.md")
    assert doc.get("missing") is not True
    assert doc["text"].startswith("# Intro")
    assert doc["physical_path"].endswith("docs/intro.md")
