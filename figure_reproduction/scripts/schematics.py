"""Retain the original conceptual illustration and build the corrected three-panel Figure 2."""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import shutil
import subprocess
import sys
import pymupdf
from PIL import Image

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    root=args.output_root.resolve()
    figures=root/'figures';figures.mkdir(exist_ok=True)
    (root/'metadata').mkdir(exist_ok=True)
    original=root/'reference_assets'/'fig1_background.png'
    shutil.copy2(original,figures/'fig1_motivation.png')
    with Image.open(original) as im:
        width,height=im.size
    doc=pymupdf.open();page=doc.new_page(width=864,height=864*height/width)
    page.insert_image(page.rect,filename=str(original))
    doc.save(figures/'fig1_motivation.pdf',deflate=True);doc.close()
    encoded=base64.b64encode(original.read_bytes()).decode('ascii')
    (figures/'fig1_motivation.svg').write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<title>Original conceptual illustration, embedded raster</title>'
        f'<image width="{width}" height="{height}" xlink:href="data:image/png;base64,{encoded}"/></svg>',encoding='utf-8')
    subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).with_name('draw_figure2.py')),
                    '--output-root',str(root)],check=True)
    architecture=json.loads((root/'figure2_metadata.json').read_text(encoding='utf-8'))
    architecture['caption']=architecture['caption'].replace(
        'CLCRN-AGF architecture retaining the original three-panel visual design.',
        'Implementation-aligned CLCRN-AGF architecture.')
    architecture['code_sources']=[{'path':p.relative_to(root).as_posix(),'sha256':sha(p)}
                                  for p in sorted((root/'reference_code').glob('*.py'))]
    metadata={'figures':{
        'fig1':{'caption':'Conceptual illustration of spatial geometry in weather modelling. The left panel illustrates variation in geodesic distance and relative orientation on the sphere; the right panel schematically depicts graph convolution and geometry-sensitive weighting. All surfaces, node positions and colour gradients are illustrative and are not measured forecasts or quantitative evidence of improved regularization.',
                'format':'Original 5740 by 2482 pixel raster retained; PDF and SVG embed the unchanged illustration.',
                'original_sha256':sha(original),'unchanged_png':sha(original)==sha(figures/'fig1_motivation.png')},
        'fig2':architecture},
        'script_sha256':sha(Path(__file__)),
        'artifacts':[{'file':p.relative_to(root).as_posix(),'sha256':sha(p)} for p in sorted(figures.glob('fig[12]_*'))]}
    (root/'metadata'/'schematics.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Retained original Figure 1 and generated corrected Figure 2 in the original three-panel style.')

if __name__=='__main__':
    main()
