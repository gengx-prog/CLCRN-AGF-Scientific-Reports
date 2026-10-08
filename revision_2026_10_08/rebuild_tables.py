"""Regenerate supplied tables/figures from recorded inference data, without weather arrays."""
import argparse, importlib.util, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    output=a.output.resolve()
    if output.exists(): raise FileExistsError('Use a fresh output directory.')
    shutil.copytree(ROOT/'data',output/'data')
    (output/'figures').mkdir()
    spec=importlib.util.spec_from_file_location('recorded_builder',ROOT/'analysis'/'retrain_build_artifacts.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.build(output,output/'data',output/'figures')
if __name__=='__main__': main()
