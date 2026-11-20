"""Self-contained interactive HTML: H-statistic and path-score heatmaps, click a cell to see the 2-D
partial dependence of that pair. No external libraries; the data is embedded as JSON."""
from __future__ import annotations

import json

TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Tree interactions</title>
<style>
:root{--bg:#fff;--fg:#1d1d1f;--mut:#6b6b70;--line:#e3e3e6;--acc:#2b6cb0}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececf0;--mut:#9a9aa2;--line:#2c2c31;--acc:#7fb2ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}p{color:var(--mut);margin:4px 0 16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}
.card{border:1px solid var(--line);border-radius:10px;padding:14px}
.card h2{font-size:15px;margin:0 0 8px}
table.hm{border-collapse:collapse;font-size:11px}
table.hm td,table.hm th{width:34px;height:30px;text-align:center;padding:0}
table.hm th{color:var(--mut);font-weight:500;max-width:34px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
table.hm td.c{cursor:pointer;border:1px solid var(--bg)}
table.hm td.sel{outline:2px solid var(--acc)}
.legend{font-size:12px;color:var(--mut)}
svg text{fill:var(--mut);font-size:11px}
</style></head><body><main>
<h1>__TITLE__</h1>
<p>__SUB__</p>
<div class="grid">
  <div class="card"><h2>Friedman H² (share of the joint effect that is interaction)</h2><div id="hm1"></div>
  <div class="legend">Click a cell to plot that pair's 2-D partial dependence.</div></div>
  <div class="card"><h2>Tree-path score (parent→child split gain share)</h2><div id="hm2"></div>
  <div class="legend">Cheap, read from the trees; also high for additive features split one after the other.</div></div>
  <div class="card"><h2 id="pdt">2-D partial dependence</h2><div id="pd"></div><div class="legend" id="pdl"></div></div>
</div></main>
<script>
const D = __DATA__;
const names = D.features;
function color(v, max){const t=Math.min(1, v/(max||1));const l=96-t*58;return `hsl(212 70% ${l}%)`;}
function heat(el, M, fmt){
  const max=Math.max(...M.flat());
  let h='<table class="hm"><tr><th></th>'+names.map(n=>`<th title="${n}">${n.slice(0,5)}</th>`).join('')+'</tr>';
  M.forEach((row,i)=>{h+=`<tr><th title="${names[i]}">${names[i].slice(0,6)}</th>`;
    row.forEach((v,j)=>{h+= i===j?'<td></td>':`<td class="c" data-i="${i}" data-j="${j}" title="${names[i]} × ${names[j]}: ${v.toFixed(3)}" style="background:${color(v,max)}"></td>`});h+='</tr>'});
  el.innerHTML=h+'</table>';
  el.querySelectorAll('td.c').forEach(td=>td.onclick=()=>show(+td.dataset.i,+td.dataset.j));
}
function show(i,j){
  document.querySelectorAll('td.sel').forEach(t=>t.classList.remove('sel'));
  document.querySelectorAll(`td[data-i="${i}"][data-j="${j}"],td[data-i="${j}"][data-j="${i}"]`).forEach(t=>t.classList.add('sel'));
  const key=[Math.min(i,j),Math.max(i,j)].join('-'); const g=D.pd[key];
  document.getElementById('pdt').textContent=`2-D partial dependence: ${names[i]} × ${names[j]}`;
  if(!g){document.getElementById('pd').innerHTML='';document.getElementById('pdl').textContent='Only the top pairs by H² have a precomputed grid.';return;}
  const W=320,H=280,m=36, nx=g.x.length, ny=g.y.length, z=g.z.flat(), lo=Math.min(...z), hi=Math.max(...z);
  const cw=(W-m)/nx, ch=(H-m)/ny; let s=`<svg viewBox="0 0 ${W} ${H}" width="100%">`;
  g.z.forEach((row,a)=>row.forEach((v,b)=>{const t=(v-lo)/((hi-lo)||1);s+=`<rect x="${m+a*cw}" y="${H-m-(b+1)*ch}" width="${cw+0.5}" height="${ch+0.5}" fill="hsl(${220-t*190} 70% ${45+t*10}%)"/>`}));
  s+=`<text x="${m}" y="${H-12}">${g.x[0]}</text><text x="${W-4}" y="${H-12}" text-anchor="end">${g.x[nx-1]}</text>`;
  s+=`<text x="4" y="${H-m}">${g.y[0]}</text><text x="4" y="12">${g.y[ny-1]}</text></svg>`;
  document.getElementById('pd').innerHTML=s;
  document.getElementById('pdl').textContent=`x: ${names[i]}, y: ${names[j]}. Colour: average prediction (blue low, orange high), ${lo.toFixed(2)} to ${hi.toFixed(2)}. Parallel colour bands = additive; bent or diagonal bands = interaction.`;
}
heat(document.getElementById('hm1'), D.h, 3); heat(document.getElementById('hm2'), D.paths, 3);
const first=D.top[0]; if(first) show(first[0], first[1]);
</script></body></html>"""


def build(title: str, subtitle: str, features: list[str], h, paths, pd_grids: dict, top: list) -> str:
    data = {"features": features, "h": h, "paths": paths, "pd": pd_grids, "top": top}
    return (TEMPLATE.replace("__TITLE__", title).replace("__SUB__", subtitle)
            .replace("__DATA__", json.dumps(data)))
