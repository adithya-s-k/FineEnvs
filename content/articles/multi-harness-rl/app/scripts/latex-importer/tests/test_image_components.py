"""Native Astro consumer of the documented Markdown-to-MDX converter CLI."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from html.parser import HTMLParser

import pytest

APP = Path(__file__).resolve().parents[3]


class RenderedImages(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []

    def handle_starttag(self, tag, attrs):
        if tag == 'img':
            self.images.append(dict(attrs))


@pytest.mark.parametrize('source', ['prose', 'markdown-image', 'pandoc-figure'])
def test_imported_figures_render_with_shipped_components(source):
    node = os.environ.get('PDF_NODE_BINARY') or shutil.which('node')
    assert node, 'Install Node or provide PDF_NODE_BINARY.'
    modules = Path(os.environ.get('ARTICLE_NODE_MODULES', str(APP / 'node_modules')))
    assert modules.is_dir(), 'Install article dependencies or provide ARTICLE_NODE_MODULES.'
    with tempfile.TemporaryDirectory(prefix='article-import-image-') as td:
        root = Path(td)
        (root / 'src/pages').mkdir(parents=True)
        (root / 'src/content/assets/image').mkdir(parents=True)
        (root / 'src/components').mkdir()
        (root / 'node_modules').symlink_to(modules, target_is_directory=True)
        (root / 'package.json').write_text('{"type":"module"}')
        (root / 'astro.config.mjs').write_text(
            "import mdx from '@astrojs/mdx';export default {integrations:[mdx()],output:'static'};\n")
        (root / 'src/pages/index.astro').write_text(
            "---\nimport Article from '../content/article.mdx';\n---\n<html><body><Article /></body></html>\n")
        shutil.copyfile(APP / 'src/components/Image.astro', root / 'src/components/Image.astro')
        image = 'joel-harness-pairing.png'
        shutil.copyfile(APP / 'src/content/assets/image' / image, root / 'src/content/assets/image' / image)
        prose = 'The harness matters most for the open-weight models you run yourself.'
        markdown = f'## Figure import control\n\n{prose}\n\n'
        if source == 'markdown-image':
            markdown += f'![Joel harness plot](./assets/image/{image})\n'
        elif source == 'pandoc-figure':
            pandoc = os.environ.get('PANDOC_BINARY') or shutil.which('pandoc')
            assert pandoc, 'Install Pandoc or provide PANDOC_BINARY.'
            latex = '\\documentclass{article}\n\\usepackage{graphicx}\n\\begin{document}\n'
            latex += '\\begin{figure}\n\\includegraphics{./assets/image/' + image + '}\n'
            latex += '\\caption{Current Joel harness figure caption.}\n\\label{fig:joel}\n'
            latex += '\\end{figure}\n\\end{document}\n'
            generated = subprocess.run([pandoc, '-f', 'latex', '-t', 'gfm+raw_html'],
                input=latex, capture_output=True, text=True, check=True, timeout=10)
            markdown += generated.stdout
        input_path = root / 'input.md'
        input_path.write_text(markdown)
        output_path = root / 'src/content/article.mdx'
        convert = subprocess.run([node, str(APP / 'scripts/latex-importer/mdx-converter.mjs'),
            f'--input={input_path}', f'--output={output_path}'],
            capture_output=True, text=True, timeout=10)
        assert convert.returncode == 0, convert.stdout + convert.stderr
        build = subprocess.run([node, str(modules / 'astro/astro.js'), 'build'], cwd=root,
            capture_output=True, text=True, timeout=45,
            env=dict(os.environ, ASTRO_TELEMETRY_DISABLED='1'))
        print(json.dumps({'source': source, 'converter_exit': convert.returncode,
            'astro_exit': build.returncode, 'astro_stdout': build.stdout,
            'astro_stderr': build.stderr}), flush=True)
        assert build.returncode == 0, build.stdout + build.stderr
        html = (root / 'dist/index.html').read_text()
        assert prose in html
        parser = RenderedImages()
        parser.feed(html)
        if source == 'prose':
            assert parser.images == []
        else:
            assert len(parser.images) == 1
            rendered = parser.images[0]
            assert (root / 'dist' / rendered['src'].lstrip('/')).is_file()
            assert int(rendered['width']) > 0
            assert int(rendered['height']) > 0
            assert 'joel-harness-pairing' in rendered['src']
            if source == 'markdown-image':
                assert rendered['alt'] == 'Joel harness plot'
            else:
                assert rendered['alt'] == 'Current Joel harness figure caption.'
                assert '<figcaption' in html
                assert 'Current Joel harness figure caption.' in html
