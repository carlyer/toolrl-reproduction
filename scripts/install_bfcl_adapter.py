"""Install the recorded ToolRL handler and patch into an exact BFCL checkout."""
import argparse,pathlib,subprocess,shutil
R=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--gorilla',required=True);a=p.parse_args()
repo=pathlib.Path(a.gorilla).resolve()
actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
if actual!='ea13468e4423454d0c213704fb87cf7cb3990433':raise ValueError('Use the frozen BFCL V3 commit')
target=repo/'berkeley-function-call-leaderboard/bfcl_eval/model_handler/local_inference/rlla_qwen.py'
if target.exists():raise FileExistsError('Handler already exists; do not overwrite a modified checkout')
patch=R/'notes/bfcl-v3-integration.patch'
subprocess.run(['git','apply','--check',str(patch)],cwd=repo,check=True)
subprocess.run(['git','apply',str(patch)],cwd=repo,check=True)
shutil.copyfile(R/'adapters/bfcl/rlla_qwen.py',target)
print('Pinned BFCL patch and ToolRL handler installed.')
