import importlib.util,json,pathlib,sys,tempfile,unittest
R=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'scripts/evaluation'))
from api_bank import calls_from_text,score
from bfcl_result_audit import audit
class EvaluationTests(unittest.TestCase):
 def test_author_soft_match_is_not_strict(self):
  gold={'name':'lookup','parameters':{'x':1}}
  self.assertEqual(score([{'x':1}],gold),1)
  self.assertEqual(score([{'x':1}],gold,strict=True),0)
 def test_wrong_named_tool_not_accepted(self):
  self.assertEqual(score([{'name':'wrong','parameters':{'x':1}}],{'name':'lookup','parameters':{'x':1}}),0)
 def test_incomplete_benchmark_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   pathlib.Path(d,'sample_result.json').write_text(json.dumps({'id':'only_one','result':'ok'})+'\n')
   with self.assertRaises(RuntimeError):audit(d)
 def test_budget_failures_stay_in_denominator(self):
  with tempfile.TemporaryDirectory() as d:
   rows=[{'id':str(i),'result':'ok'} for i in range(4441)]
   rows[0]['result']='Error during inference: TOOLRL_CONTEXT_BUDGET_EXHAUSTED: retained'
   pathlib.Path(d,'sample_result.json').write_text(''.join(json.dumps(r)+'\n' for r in rows))
   r=audit(d);self.assertEqual(r['samples'],4441);self.assertEqual(r['context_budget_failures'],1)
 def test_infrastructure_errors_fail_evaluation(self):
  with tempfile.TemporaryDirectory() as d:
   rows=[{'id':str(i),'result':'ok'} for i in range(4441)]
   rows[0]['result']='Error during inference: connection refused'
   pathlib.Path(d,'sample_result.json').write_text(''.join(json.dumps(r)+'\n' for r in rows))
   with self.assertRaises(RuntimeError):audit(d)
if __name__=='__main__':unittest.main()
