"""Distinguish model context-budget failures from infrastructure failures."""
import json,pathlib

def is_budget_failure(row):
    text=str(row.get('result',''))
    return text.startswith('Error during inference:') and (
        'TOOLRL_CONTEXT_BUDGET_EXHAUSTED:' in text or
        ('Error code: 400' in text and "This model's maximum context length is 32768 tokens" in text and 'Please reduce the length of the messages or completion.' in text))

def audit(directory):
    files=list(pathlib.Path(directory).glob('*_result.json'))
    rows=[json.loads(line) for p in files for line in p.read_text().splitlines() if line.strip()]
    errors=[r for r in rows if 'traceback' in r or str(r.get('result','')).startswith('Error during inference:')]
    budget=[r for r in errors if is_budget_failure(r)]
    infra=[r for r in errors if not is_budget_failure(r)]
    if len(rows)!=4441 or len({r['id'] for r in rows})!=4441:
        raise RuntimeError(f'Incomplete generation: {len(rows)}/4441')
    if infra:raise RuntimeError(f'{len(infra)} infrastructure errors; first={infra[0]["id"]}')
    return dict(samples=len(rows),context_budget_failures=len(budget),context_budget_failure_ids=[r['id'] for r in budget],infrastructure_errors=0,policy='All 4441 tasks scored by official checker; budget failures retained as task failures; no omitted samples')
