"""Native importer modes and direct-converter controls with real Pandoc."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

APP = Path(__file__).resolve().parents[3]
PROSE = 'The harness matters most for the open-weight models you run yourself.'


def fixture_tool(root):
    tool = root / 'app/scripts/latex-importer'
    shutil.copytree(APP / 'scripts/latex-importer', tool,
        ignore=shutil.ignore_patterns('tests', '__pycache__', '.pytest_cache'))
    (tool / 'input').mkdir(exist_ok=True)
    (root / 'app/src/content').mkdir(parents=True)
    return tool


def native_cli(tool, script, args):
    node = os.environ.get('PDF_NODE_BINARY') or shutil.which('node')
    pandoc = os.environ.get('PANDOC_BINARY') or shutil.which('pandoc')
    assert node and pandoc, 'Install Node/Pandoc or provide PDF_NODE_BINARY/PANDOC_BINARY.'
    env = dict(os.environ, PATH=str(Path(pandoc).parent) + os.pathsep + os.environ.get('PATH', ''))
    result = subprocess.run([node, str(tool / script), *args],
        capture_output=True, text=True, timeout=15, env=env)
    print(result.stdout + result.stderr, flush=True)
    return result


def test_bibliography_only_uses_actual_bib_without_latex_and_creates_output():
    with tempfile.TemporaryDirectory(prefix='article-bib-only-') as td:
        root = Path(td).resolve()  # Match Node's canonical module URLs on macOS.
        tool = fixture_tool(root)
        source = tool / 'input/paper.bib'
        shutil.copyfile(APP / 'src/content/bibliography.bib', source)
        # The real direct cleaner is the positive control and expected content producer.
        control = native_cli(tool, 'bib-cleaner.mjs',
            [f'--input={source}', f'--output={root / "control.bib"}'])
        assert control.returncode == 0
        output = root / 'fresh-output'
        result = native_cli(tool, 'index.mjs',
            ['--bib-only', f'--input={source.with_suffix(".tex")}', f'--output={output}'])
        assert result.returncode == 0, result.stdout + result.stderr
        assert (output / 'main.bib').read_bytes() == (root / 'control.bib').read_bytes()
        assert sorted(path.name for path in output.iterdir()) == ['main.bib']
        assert not (root / 'app/src/content/article.mdx').exists()


def test_index_help_describes_index_modes_without_conversion():
    with tempfile.TemporaryDirectory(prefix='article-import-help-') as td:
        root = Path(td).resolve()
        tool = fixture_tool(root)
        result = native_cli(tool, 'index.mjs', ['--help'])
        assert result.returncode == 0
        assert 'LaTeX to Markdown Toolkit' in result.stdout
        assert '--bib-only' in result.stdout
        assert '--convert-only' in result.stdout
        assert not (tool / 'output').exists()


def test_direct_latex_converter_still_runs_real_pandoc():
    with tempfile.TemporaryDirectory(prefix='article-direct-converter-') as td:
        root = Path(td).resolve()
        tool = fixture_tool(root)
        source = tool / 'input/main.tex'
        source.write_text('\\documentclass{article}\n\\begin{document}\n' + PROSE + '\n\\end{document}\n')
        output = root / 'direct-output'
        result = native_cli(tool, 'latex-converter.mjs',
            [f'--input={source}', f'--output={output}'])
        assert result.returncode == 0, result.stdout + result.stderr
        assert PROSE in (output / 'main.md').read_text()
        assert not (output / 'main.mdx').exists()
