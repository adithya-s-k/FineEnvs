"""Native export CLI regression using current canvas, SVG and PNG producers."""
import functools
import http.server
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import pytest

APP=pathlib.Path(__file__).resolve().parents[2]

@pytest.fixture(scope='module')
def exported_figures():
    app=APP
    node=os.environ.get('PDF_NODE_BINARY') or shutil.which('node')
    assert node, 'Install the article Node dependencies or provide PDF_NODE_BINARY.'
    modules=pathlib.Path(os.environ.get('SCREENSHOT_NODE_MODULES',str(app/'node_modules')))
    assert modules.is_dir(), 'Install the article dependencies or provide SCREENSHOT_NODE_MODULES.'
    with tempfile.TemporaryDirectory(prefix='article-canvas-export-') as td:
        root=pathlib.Path(td);(root/'scripts').mkdir();(root/'public').mkdir()
        (root/'node_modules').symlink_to(modules,target_is_directory=True)
        shutil.copyfile(app/'scripts/screenshot-elements.mjs',root/'scripts/screenshot-elements.mjs')
        shutil.copytree(app/'public/logos',root/'public/logos')
        shutil.copyfile(app/'src/content/assets/image/joel-harness-pairing.png',root/'public/joel.png')
        canvas=(app/'src/content/embeds/banner-opaque.html').read_text()
        svg=(app/'src/content/embeds/d3-harness-anatomy.html').read_text()
        html='''<!doctype html><meta charset="utf-8"><html data-theme="light"><style>body{margin:0;font-family:system-ui;background:white;color:#14161c;}main{width:960px;margin:auto;}figure{margin:0 0 30px;}img{width:960px;height:auto;} :root{--text-color:rgb(20,22,28);--muted-color:rgb(100,104,115);--border-color:rgba(128,128,128,.28);}</style><main>'''
        html+=f'<figure id="banner-opaque" class="html-embed"><div class="html-embed__card">{canvas}</div></figure>'
        html+=f'<figure id="harness-anatomy" class="html-embed"><figcaption class="html-embed__title">Inside a harness</figcaption><div class="html-embed__card">{svg}</div></figure>'
        html+='<figure class="image-wrapper" id="joel"><img src="/joel.png" alt="Current Joel harness plot"></figure></main></html>'
        (root/'public/index.html').write_text(html)
        bootstrap=root/'native-chrome.mjs'
        bootstrap.write_text('''import {chromium} from 'playwright';
    const launch=chromium.launch.bind(chromium);
    chromium.launch=async options=>{
     const b=await launch({...options,...(process.env.PLAYWRIGHT_CHROME?{executablePath:process.env.PLAYWRIGHT_CHROME}:{})});
     console.log('Native browser:',b.version());
     let busy=false;
     const timer=setInterval(async()=>{
      if(busy)return;const p=b.contexts().flatMap(c=>c.pages()).find(p=>p.url().startsWith('http:'));if(!p)return;busy=true;
      try{const data=await p.evaluate(()=>{
       const c=document.querySelector('#banner-opaque canvas');if(!c||!c.width||!c.height)return null;
       const px=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let colored=0,painted=0;
       for(let i=0;i<px.length;i+=4){if(px[i+3])painted++;if(px[i+3]&&Math.max(px[i],px[i+1],px[i+2])-Math.min(px[i],px[i+1],px[i+2])>30)colored++;}
       return {width:c.width,height:c.height,painted,colored};
      });if(data?.colored>1000){console.log('Native source canvas:',JSON.stringify(data));clearInterval(timer);}}
      catch{}finally{busy=false;}
     },50);b.on('disconnected',()=>clearInterval(timer));return b;
    };''')
        class Quiet(http.server.SimpleHTTPRequestHandler):
            def log_message(self,*args):pass
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(root/'public')))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        env=dict(os.environ,SCREENSHOT_URL=f'http://127.0.0.1:{server.server_port}/?og',SCREENSHOT_TIMEOUT_MS='10000')
        process=None
        try:
            process=subprocess.Popen([node,'--import',str(bootstrap),str(root/'scripts/screenshot-elements.mjs')],cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
            stdout,_=process.communicate(timeout=45);print(stdout,flush=True)
            assert process.returncode==0,stdout
            assert 'Captured 3 screenshots' in stdout,stdout
            outputs=list((root/'screenshots').glob('*.png'));assert len(outputs)==3,outputs
            stats_script=root/'stats.mjs';stats_script.write_text('''import sharp from 'sharp';import {readdir} from 'node:fs/promises';
    for(const f of (await readdir('screenshots')).filter(f=>f.endsWith('.png'))){
     const {data,info}=await sharp('screenshots/'+f).ensureAlpha().raw().toBuffer({resolveWithObject:true});let colored=0;
     for(let i=0;i<data.length;i+=4)if(data[i+3]&&Math.max(data[i],data[i+1],data[i+2])-Math.min(data[i],data[i+1],data[i+2])>30)colored++;
     console.log(JSON.stringify({file:f,width:info.width,height:info.height,colored}));
    }''')
            stats=subprocess.run([node,str(stats_script)],cwd=root,text=True,capture_output=True,timeout=10,check=True)
            print(stats.stdout,flush=True)
            receipt=os.environ.get('SCREENSHOT_RECEIPT_DIR')
            if receipt:
                target=pathlib.Path(receipt);target.mkdir(parents=True,exist_ok=True)
                for output in outputs:shutil.copyfile(output,target/output.name)
            match=re.search(r'Native source canvas: (\{[^\n]+\})',stdout)
            assert match,stdout
            result={'source':json.loads(match.group(1)), 'outputs':{
                data['file']:data for data in (json.loads(line) for line in stats.stdout.splitlines())}}
            return result
        finally:
            if process:
                try:
                    if hasattr(os,'killpg'):os.killpg(process.pid,signal.SIGTERM)
                    elif process.poll() is None:process.terminate()
                except ProcessLookupError:pass
                try:process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    if hasattr(os,'killpg'):os.killpg(process.pid,signal.SIGKILL)
                    else:process.kill()
                    process.wait(timeout=5)
            server.shutdown();server.server_close();thread.join(timeout=2)


def test_current_native_banner_canvas_has_real_graphics(exported_figures):
    source=exported_figures['source']
    assert source['painted']>1000
    assert source['colored']>1000


def test_canvas_graphics_survive_actual_cli_export(exported_figures):
    output=exported_figures['outputs']['1-embed--banner-opaque.png']
    assert output['colored']>1000
    assert output['height']>=exported_figures['source']['height']//2


def test_existing_svg_and_current_png_export_controls(exported_figures):
    outputs=exported_figures['outputs']
    assert outputs['2-embed--inside-a-harness.png']['colored']>1000
    assert outputs['3-image--joel.png']['colored']>1000
