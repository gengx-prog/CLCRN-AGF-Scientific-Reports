"""Rebuild the complete figure/table replacement pack from its frozen data snapshot."""
from pathlib import Path
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import importlib.metadata
import pymupdf

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path):
    return json.loads(path.read_text(encoding='utf-8'))

def latex(text):
    mapping = {'&':r'\&', '%':r'\%', '$':r'\$', '#':r'\#', '_':r'\_',
               '{':r'\{', '}':r'\}', '~':r'\textasciitilde{}', '^':r'\textasciicircum{}',
               '\\':r'\textbackslash{}', '\u2212':r'$-$', '\u00d7':r'$\times$',
               '\u2013':'--','\u2014':'---', '\u00b1':r'$\pm$'}
    return ''.join(mapping.get(c,c) for c in text)

def main():
    manifest = load(ROOT/'metadata'/'input_manifest.json')
    for rec in manifest['files']:
        assert sha(ROOT/rec['file']) == rec['sha256'], f"Input changed: {rec['file']}"
    logs = ROOT/'validation'
    logs.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONUTF8='1', SOURCE_DATE_EPOCH='1791417600')
    for script in ['schematics.py','comparisons.py','diagnostics.py','tables.py','portfolio.py']:
        args = [sys.executable,'-X','utf8',str(ROOT/'scripts'/script),'--output-root',str(ROOT)]
        if script in ['comparisons.py','diagnostics.py','tables.py']:
            args += ['--data-root',str(ROOT/'data')]
        run = subprocess.run(args,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',env=env)
        (logs/(script.removesuffix('.py')+'.log')).write_text(run.stdout+'\n'+run.stderr,encoding='utf-8')
        if run.returncode:
            raise RuntimeError(f'{script} failed; read validation/{script.removesuffix(".py")}.log')
        print(f'{script}: passed',flush=True)

    captions={}
    for meta in ['schematics','comparisons','diagnostics']:
        for key,value in load(ROOT/'metadata'/f'{meta}.json')['figures'].items():
            number=int(key[3:].split('_')[0])
            captions[number]=value['caption']
    manuscript=ROOT/'manuscript_snippets'
    manuscript.mkdir(exist_ok=True)
    blocks=[]
    mappings=[]
    old_files=['fig1_background.png','fig_architecture_overview.png','fig_A1_main_baselines.pdf',
               'fig_primary_replicate_comparison.pdf','fig_A2_horizon_curves.pdf',
               'fig_B2_schedule_decomp_mae.pdf','fig_missing_nodes_mae.pdf','fig_gate_distribution.pdf']
    changes=['Original conceptual art retained; illustrative meaning clarified.',
             'Pastel three-panel design redrawn to match implemented pre-CLConv fusion.',
             'Heatmap retains eight metric columns; rows are the three measured current methods.',
             'Five-seed scatter and sample SD; all points and change labels recalculated.',
             'Same four-panel curve styles; five AGF seeds and sample SD, full 657 windows.',
             'Grouped bars retained; current control/AGF in four task-specific axes.',
             'Four-colour line style retained; relative MAE on matched 64-window subset.',
             'Horizontal distributions retained; all four tasks, full scalar gates, binned quantiles.']
    for n in range(1,9):
        pdf=next((ROOT/'figures').glob(f'fig{n}_*.pdf'))
        rel=pdf.relative_to(ROOT).as_posix()
        blocks.append('\\begin{figure}[t]\n\\centering\n'
                      f'\\includegraphics[width=\\linewidth]{{{rel}}}\n'
                      f'\\caption{{{latex(captions[n])}}}\n'
                      f'\\label{{fig:restyled_{n}}}\n\\end{{figure}}\n')
        mappings.append({'figure':n,'original_file':old_files[n-1],'replacement':rel,'content_change':changes[n-1]})
    (manuscript/'figures.tex').write_text('% Paths are relative to the replacement-pack root. Requires graphicx.\n\n'+'\n'.join(blocks),encoding='utf-8')
    (manuscript/'figure_captions.txt').write_text('\n\n'.join(f'Figure {n}. {captions[n]}' for n in range(1,9))+'\n',encoding='utf-8')
    with (manuscript/'replacement_map.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(mappings[0]))
        writer.writeheader();writer.writerows(mappings)

    # Final structural validation is in addition to the independent numeric checks in each script.
    checks=[]
    for n in range(1,9):
        pdf=next((ROOT/'figures').glob(f'fig{n}_*.pdf'))
        with pymupdf.open(pdf) as doc:
            assert len(doc)==1
            images=len(doc[0].get_images(full=True))
            assert images>0 if n==1 else images==0
            bad=[]
            for block in doc[0].get_text('dict')['blocks']:
                for line in block.get('lines',[]):
                    for span in line['spans']:
                        rect=pymupdf.Rect(span['bbox'])
                        if not (doc[0].rect+(-1,-1,1,1)).contains(rect):
                            bad.append(span['text'])
            assert not bad,(pdf.name,bad)
            checks.append({'figure':n,'pdf_pages':len(doc),'raster_images':images,'text_inside_page':True})
        for ext in ['png','svg']:
            assert pdf.with_suffix('.'+ext).stat().st_size>0
    assert sha(ROOT/'figures'/'fig1_motivation.png')==sha(ROOT/'reference_assets'/'fig1_background.png')
    with pymupdf.open(ROOT/'preview'/'figures_and_tables_preview.pdf') as doc:
        assert len(doc)==10
    tables=load(ROOT/'metadata'/'tables.json')
    assert tables['compilation']['status']=='passed'
    assert tables['compilation']['layout_warnings']==[]
    report={'status':'passed','input_files_verified':len(manifest['files']),
            'scope':'CPU-only plotting and statistical verification; no model training or GPU inference in this restyling step.',
            'figures':checks,'table_numeric_checks':tables['validation']['numeric_checks'],
            'table_maximum_absolute_error':tables['validation']['maximum_absolute_error'],
            'preview_pages':10,'original_motivation_png_unchanged':True,
            'environment':{'python':sys.version,'platform':platform.platform(),
                          'packages':{name:importlib.metadata.version(name) for name in ['matplotlib','numpy','pandas','scipy','pymupdf','Pillow']}}}
    (logs/'validation_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # All files, except the manifest itself and transient bytecode, have downloadable checksums.
    with (ROOT/'SHA256SUMS.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['file','bytes','sha256'])
        for p in sorted(ROOT.rglob('*')):
            if p.is_file() and p.name!='SHA256SUMS.csv' and '__pycache__' not in p.parts:
                writer.writerow([p.relative_to(ROOT).as_posix(),p.stat().st_size,sha(p)])
    print(json.dumps({'status':'passed','figures':8,'tables':4,'table_checks':report['table_numeric_checks'],
                      'preview':'preview/figures_and_tables_preview.pdf'},ensure_ascii=False))

if __name__=='__main__':
    main()
