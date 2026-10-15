"""Native full importer input paths, real Pandoc and copied article checks."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest

APP = Path(__file__).resolve().parents[3]
PROSE = 'The harness matters most for the open-weight models you run yourself.'
TITLE = 'Native input source title'


@pytest.mark.parametrize('name,external', [('main', False), ('paper', False), ('main', True)])
def test_full_import_uses_requested_source_and_generated_markdown(name, external):
    node = os.environ.get('PDF_NODE_BINARY') or shutil.which('node')
    pandoc = os.environ.get('PANDOC_BINARY') or shutil.which('pandoc')
    assert node and pandoc, 'Install Node/Pandoc or provide PDF_NODE_BINARY/PANDOC_BINARY.'
    with tempfile.TemporaryDirectory(prefix='article-import-path-') as td:
        root = Path(td).resolve()
        tool = root / 'app/scripts/latex-importer'
        shutil.copytree(APP / 'scripts/latex-importer', tool,
            ignore=shutil.ignore_patterns('tests', '__pycache__', '.pytest_cache'))
        (root / 'app/src/content').mkdir(parents=True)
        folder = root / 'paper-source' if external else tool / 'input'
        folder.mkdir(exist_ok=True)
        source = folder / f'{name}.tex'
        source.write_text('\\documentclass{article}\n\\title{' + TITLE + '}\n'
            '\\begin{document}\n\\section{Input path control}\n' + PROSE + '\n\\end{document}\n')
        output = tool / 'output'
        result = subprocess.run([node, str(tool / 'index.mjs'),
            f'--input={source}', f'--output={output}'],
            capture_output=True, text=True, timeout=15,
            env=dict(os.environ, PATH=str(Path(pandoc).parent) + os.pathsep + os.environ.get('PATH', '')))
        print(result.stdout + result.stderr, flush=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert PROSE in (output / f'{name}.md').read_text()
        mdx = (output / 'main.mdx').read_text()
        assert PROSE in mdx
        assert f'title: "{TITLE}"' in mdx
        assert (root / 'app/src/content/article.mdx').read_text() == mdx
        if name != 'main':
            assert not (output / 'main.md').exists()
