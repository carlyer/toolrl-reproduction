"""Run full BFCL V3 comparison; retain context-budget failures in denominator."""
import fcntl,json,os,pathlib,signal,subprocess,sys,time,traceback,urllib.request
from bfcl_result_audit import audit
ROOT=pathlib.Path(__file__).resolve().parents[2]
PY=os.environ.get('BFCL_PYTHON',str(ROOT/'.venv-bfcl/bin/python'));LEGACY=os.environ.get('TOOLRL_PYTHON',str(ROOT/'.venv/bin/python'))
OUT=ROOT/'results/bfcl-v3';OUT.mkdir(exist_ok=True)
GPU=os.getenv('TOOLRL_EVAL_GPU','1');PORT='1061'
MODELS={'base':os.environ['BASE_MODEL'],'grpo':os.environ['GRPO_MODEL']}
(ROOT/'.cache').mkdir(exist_ok=True);(ROOT/'logs').mkdir(exist_ok=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_ENDPOINT='127.0.0.1',VLLM_PORT=PORT,NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost',BFCL_PROJECT_ROOT=str(OUT))
def status(**kwargs):
    (OUT/'pipeline-status.json').write_text(json.dumps(dict(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),gpu=GPU,metrics_reproduced=False,**kwargs),indent=2)+'\n')
def failed(cls,exc,tb):
    status(stage='failed',error=str(exc));traceback.print_exception(cls,exc,tb)
sys.excepthook=failed
for arm,model in MODELS.items():
    PORT='1071' if arm=='base' else '1072'
    env['VLLM_PORT']=PORT
    dest=OUT/'responses'/('toolrl-'+arm)
    files=list(dest.glob('*_result.json'))
    count=sum(1 for p in files for line in p.open() if line.strip())
    if count!=4441:
        with (ROOT/'.cache'/f'gpu{GPU}.lock').open('a+') as lock:
            status(stage='waiting_gpu',arm=arm);fcntl.flock(lock,fcntl.LOCK_EX)
            status(stage='loading_model',arm=arm)
            with (ROOT/f'logs/bfcl-fixed-server-{arm}.log').open('w') as log:
                server=subprocess.Popen([LEGACY,'-m','vllm.entrypoints.openai.api_server','--model',model,'--host','127.0.0.1','--port',PORT,'--tensor-parallel-size','1','--gpu-memory-utilization','0.6','--max-model-len','32768','--dtype','bfloat16','--enforce-eager','--guided-decoding-backend','lm-format-enforcer','--disable-log-requests'],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                try:
                    deadline=time.monotonic()+900
                    while True:
                        if server.poll() is not None:raise RuntimeError('vLLM server failed')
                        try:
                            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health',timeout=3) as response:
                                if response.status==200:break
                        except Exception:pass
                        if time.monotonic()>deadline:raise TimeoutError('Server startup timeout')
                        time.sleep(3)
                    status(stage='generation',arm=arm)
                    subprocess.run([PY,'-m','bfcl_eval','generate','--model','toolrl-'+arm,'--test-category','all','--temperature','0.0001','--num-threads','1','--num-gpus','1','--skip-server-setup','--local-model-path',model,'--result-dir',str(OUT/'responses')],env=env,check=True)
                finally:
                    try:os.killpg(server.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:server.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        try:os.killpg(server.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        server.wait()
                    # Ensure owned engine subprocesses cannot retain GPU after lock release.
                    try:os.killpg(server.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
    details=audit(dest)
    (OUT/f'{arm}-generation-audit.json').write_text(json.dumps(details,indent=2)+'\n')
    status(stage='official_checker',arm=arm,audit=details)
    subprocess.run([PY,'-m','bfcl_eval','evaluate','--model','toolrl-'+arm,'--test-category','all','--result-dir',str(OUT/'responses'),'--score-dir',str(OUT/'scores')],env=env,check=True)
status(stage='complete',benchmark_commit='ea13468e4423454d0c213704fb87cf7cb3990433',protocol='Shared 32768 context, 4096 generation cap; official checker retains context exhaustion as failure; no history truncation')
