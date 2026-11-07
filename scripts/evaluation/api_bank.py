"""Complete API-Bank evaluation using the author's prompts and soft-match protocol."""
import argparse, hashlib, json, os, pathlib, time, importlib.metadata as metadata
ROOT=pathlib.Path(__file__).resolve().parents[2]
def calls_from_text(text):
    calls=[]
    text=text.split('<tool_call>')[-1].split('</tool_call>')[0].strip()
    for line in text.splitlines():
        try: calls.append(json.loads(line))
        except (json.JSONDecodeError,TypeError): pass
    return calls

def score(calls,answer,strict=False):
    answer=answer[0] if isinstance(answer,list) else answer
    if strict:
        return int(len(calls)==1 and calls[0]==answer)
    # Preserve the author's permissive name/parameter matching for comparability.
    for call in calls:
        try:
            if isinstance(call,str):call=json.loads(call)
            if 'name' not in call or 'parameters' not in call:
                name,params=answer['name'],call
            else:name,params=call['name'],call['parameters']
            if name==answer['name'] and params==answer['parameters']:return 1
        except (TypeError,ValueError,KeyError):pass
    return 0

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    out=pathlib.Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    samples=[];hashes={}
    for level in (1,2,3):
        src=ROOT/f'upstream/benchmarks/API-Bank/level-{level}-api_processed.json'
        hashes[str(level)]=hashlib.sha256(src.read_bytes()).hexdigest()
        for i,item in enumerate(json.loads(src.read_text())):samples.append((f'Level{level}_{i}',level,item))
    assert len(samples)==597
    assert score([{'name':'f','parameters':{'x':1}}],{'name':'f','parameters':{'x':1}})==1
    assert score([{'name':'g','parameters':{'x':1}}],{'name':'f','parameters':{'x':1}})==0
    assert score([{'x':1}],{'name':'f','parameters':{'x':1}})==1 # Author fallback.
    assert score([{'x':1}],{'name':'f','parameters':{'x':1}},strict=True)==0
    if a.preflight:
        print(json.dumps({'samples':len(samples),'levels':{str(l):sum(x[1]==l for x in samples) for l in (1,2,3)},'source_sha256':hashes,'model':a.model}));return
    path=out/'predictions.jsonl';records={}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r=json.loads(line);assert r['model']==a.model;records[r['id']]=r
    expected={x[0] for x in samples}
    assert set(records)<=expected
    pending=[x for x in samples if x[0] not in records]
    if pending:
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
        llm=LLM(model=a.model,tensor_parallel_size=1,gpu_memory_utilization=0.45,max_model_len=4096,enforce_eager=True)
        params=SamplingParams(max_tokens=4096,temperature=0.0001,seed=42)
        started=time.monotonic()
        with path.open('a') as f:
            for start in range(0,len(pending),a.batch_size):
                batch=pending[start:start+a.batch_size]
                prompts=[tokenizer.apply_chat_template([{'role':'system','content':item['system']},{'role':'user','content':item['user']}],tokenize=False,add_generation_prompt=True) for _,_,item in batch]
                for prompt in prompts:
                    if len(tokenizer.encode(prompt,add_special_tokens=False))>=4096:raise ValueError('Prompt exceeds original context limit; no silent truncation')
                outputs=llm.generate(prompts,params,use_tqdm=False)
                assert len(outputs)==len(batch)
                for (sid,level,item),output in zip(batch,outputs):
                    text=output.outputs[0].text.strip();calls=calls_from_text(text)
                    row={'id':sid,'level':level,'model':a.model,'raw_output':text,'tool_calls':calls,'answer':item['answer'],'author_soft_correct':score(calls,item['answer']),'strict_correct':score(calls,item['answer'],True)}
                    records[sid]=row;f.write(json.dumps(row,ensure_ascii=False)+'\n')
                f.flush();print(json.dumps({'completed':len(records),'total':597,'elapsed_seconds':round(time.monotonic()-started,2)}),flush=True)
    assert set(records)==expected
    metrics={'benchmark':'API-Bank','protocol':'author processed prompts + soft-match against first gold call; any matching prediction accepted','strict_protocol':'one predicted call exactly equals first gold call','model':a.model,'source_sha256':hashes,'samples':597,'author_soft_accuracy':sum(r['author_soft_correct'] for r in records.values())/597,'strict_accuracy':sum(r['strict_correct'] for r in records.values())/597,'levels':{},'vllm':metadata.version('vllm'),'requested_temperature':0.0001,'effective_temperature':0.01,'temperature_note':'vLLM 0.6.3 clamps positive temperatures below 0.01 to 0.01','seed':42,'max_model_len':4096,'max_tokens':4096,'implementation_change':'batched offline inference; source script generated sequentially'}
    for l in (1,2,3):
        rows=[r for r in records.values() if r['level']==l];metrics['levels'][str(l)]={'samples':len(rows),'author_soft_accuracy':sum(r['author_soft_correct'] for r in rows)/len(rows),'strict_accuracy':sum(r['strict_correct'] for r in rows)/len(rows)}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
