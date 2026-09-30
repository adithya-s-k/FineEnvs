import asyncio,json,sys
from pathlib import Path
from urllib.parse import urlencode
sys.path.insert(0,str(Path('HuggingEnvs/04-data-agent/hf').resolve()))
from consolidate_async_runs import PROJECT, ARMS, DEFAULT_OUT, native
from playwright.async_api import async_playwright
URL='https://huggingenvs-data-agent-training-comparison-trackio.hf.space/'
async def main():
 tasks=[];errors=[];proof={}
 async with async_playwright() as p:
  import os
  browser=await p.chromium.launch(headless=True,args=['--no-sandbox'],env={**os.environ,'LD_LIBRARY_PATH':'/tmp/data-agent-ui-browser-libs/root/usr/lib/x86_64-linux-gnu'})
  page=await browser.new_page(viewport={'width':1800,'height':1100})
  page.on('pageerror',lambda e:errors.append(str(e)))
  async def observe(response):
   if '/api/get_logs_batch' not in response.url or response.status!=200:return
   payload=await response.json()
   for row in payload.get('data',[]):
    if row.get('run') not in ARMS:continue
    logs=row.get('logs') or []
    if not logs:continue
    train=[r['step'] for r in logs if 'train/reward' in r]
    scores={r['step']:r['eval/pass_at_1'] for r in logs if 'eval/pass_at_1' in r}
    if train:assert sorted(train)==list(range(1,1001)),(row['run'],len(train))
    proof[row['run']]={'training_steps':len(train),'evaluation_scores':scores,'metric_names':sorted({k for r in logs for k in r if k.startswith(('train/','eval/'))})}
  page.on('response',lambda r:tasks.append(asyncio.create_task(observe(r))))
  ids=','.join(native.digest([PROJECT,a])[:32] for a in ARMS)
  params={'project':PROJECT,'run_ids':ids,'smoothing':'0','metric_filter':'^(eval/pass_at_1|train/reward_rolling20)$'}
  url=URL+'?'+urlencode(params)
  response=await page.goto(url,wait_until='networkidle',timeout=60000)
  assert response.status==200
  await page.get_by_text(ARMS[0],exact=True).first.wait_for(timeout=30000)
  await page.wait_for_timeout(3000)
  await asyncio.gather(*tasks)
  assert set(proof)==set(ARMS),proof
  assert abs(proof[ARMS[0]]['evaluation_scores'][500]-.37)<1e-10
  assert abs(proof[ARMS[1]]['evaluation_scores'][400]-.264)<1e-10
  assert all(v['training_steps']==1000 for v in proof.values())
  assert await page.locator('canvas').count()>=2
  await page.screenshot(path=str(DEFAULT_OUT/'dashboard-overview.png'),full_page=False)
  for name,regex in [('difficulty','^eval/difficulty/'),('harness','^eval/harness/'),('harness-difficulty','^eval/harness_difficulty/')]:
   await page.get_by_placeholder('e.g., loss|ndcg@10|gpu').fill(regex)
   await page.wait_for_timeout(1500)
   expected={'difficulty':3,'harness':4,'harness-difficulty':12}[name]
   assert await page.locator('canvas').count()>=expected,(name,await page.locator('canvas').count())
   await page.screenshot(path=str(DEFAULT_OUT/f'dashboard-{name}.png'),full_page=False)
  assert not errors,errors
  result={'url':url,'runs':proof,'javascript_errors':errors,'public_unauthenticated_browser':True,'views_checked':['overview','difficulty','harness','harness-difficulty']}
  (DEFAULT_OUT/'UI_VERIFIED.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps({'url':url,'training_steps':{k:v['training_steps'] for k,v in proof.items()},'views_checked':result['views_checked'],'errors':errors}))
  await browser.close()
asyncio.run(main())
