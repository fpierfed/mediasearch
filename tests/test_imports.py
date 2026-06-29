import subprocess
import sys


def test_cli_import_does_not_import_mlx_runtimes():
    """Importing CLI metadata must not require MLX model runtimes."""
    code = """
import builtins

real_import = builtins.__import__
blocked = {'mlx_whisper', 'mlx_embeddings'}

def guarded_import(name, *args, **kwargs):
    if any(name == module or name.startswith(f'{module}.') for module in blocked):
        raise RuntimeError(f'{name} imported eagerly')
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import mediasearch.cli
"""
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
