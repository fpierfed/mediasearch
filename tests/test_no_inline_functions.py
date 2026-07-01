import ast
from pathlib import Path


def test_mediasearch_modules_keep_helpers_top_level():
    """Keep production helpers importable and directly monkeypatchable.

    The package tests rely on patching implementation helpers by module path.
    Nested functions and lambdas hide behavior inside call sites, making those
    boundaries harder to test without exercising heavyweight media/model paths.
    """
    project_root = Path(__file__).resolve().parents[1]
    inline_functions: list[str] = []

    for path in sorted((project_root / 'mediasearch').glob('*.py')):
        tree = ast.parse(path.read_text(), filename=str(path))
        parents: list[ast.AST] = []

        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                setattr(child, '_parent', node)

        for node in ast.walk(tree):
            if isinstance(node, ast.Lambda):
                inline_functions.append(f'{path.relative_to(project_root)}:{node.lineno}: lambda')
                continue

            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            parents.clear()
            parent = getattr(node, '_parent', None)
            while parent is not None:
                parents.append(parent)
                parent = getattr(parent, '_parent', None)

            if any(
                isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
                for parent in parents
            ):
                inline_functions.append(
                    f'{path.relative_to(project_root)}:{node.lineno}: {node.name}'
                )

    assert inline_functions == []
