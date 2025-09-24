import pathlib

import mkdocs_gen_files

root = (pathlib.Path(__file__).parent.parent / "src").resolve()
skip_dirs = ["__pycache__", "scripts", "interactive"]


def main():
    nav = mkdocs_gen_files.Nav()

    for path in sorted(root.glob("**/*.py")):
        if any(skip in str(path) for skip in skip_dirs):
            continue

        module_path = pathlib.Path(
            str(path.relative_to(root).with_suffix("")).replace("__init__", "")
        )
        ident = ".".join(module_path.parts)
        doc_path = pathlib.Path(
            str(path.relative_to(root / "saev").with_suffix(".md")).replace(
                "__init__.md", f"{ident}.md"
            )
        )
        full_doc_path = pathlib.Path("api", doc_path)

        nav[ident] = doc_path

        with mkdocs_gen_files.open(full_doc_path, "w") as f:
            print("::: " + ident, file=f)

    with mkdocs_gen_files.open("api/summary.md", "w") as nav_file:
        nav_file.writelines(nav.build_literate_nav())


# No if __name__ == '__main__': https://github.com/oprypin/mkdocs-gen-files/pull/36
main()
