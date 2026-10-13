"""Regenerate numerical CSVs and PNG diagnostics from recorded inference JSONs."""
import argparse, importlib.util, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    output=a.output.resolve()
    repo=ROOT.parent
    for name in ('revision_2026_10_08','reviewer_tools','model','lib','experiments',
                 'analysis','scripts','validation','.git','.github'):
        if output.is_relative_to((repo/name).resolve()):
            raise ValueError('Output must be outside supplied source/evidence directories.')
    if output.exists(): raise FileExistsError('Use a fresh output directory.')
    (output/'data').mkdir(parents=True)
    for source in (ROOT/'data').glob('*.json'):
        if source.name.startswith(('dataset_','inference_','persistence_')):
            shutil.copy2(source,output/'data'/source.name)
    (output/'figures').mkdir()
    spec=importlib.util.spec_from_file_location('recorded_builder',ROOT/'analysis'/'retrain_build_artifacts.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.build(output,output/'data',output/'figures')
if __name__=='__main__': main()
