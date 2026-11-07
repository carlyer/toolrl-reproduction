import json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts/evaluation'))
from api_bank import score
gold={'name':'lookup','parameters':{'id':1}}
prediction=[{'id':1}]
print(json.dumps({'synthetic_demo':True,'author_soft_match':score(prediction,gold),'strict_single_call_match':score(prediction,gold,strict=True)},indent=2))
