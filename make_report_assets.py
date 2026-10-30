"""Generate reproducible report figures from experiment CSVs."""
from pathlib import Path
import cv2, json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'out'/'experiments'; FIG=OUT/'figures'; FIG.mkdir(parents=True,exist_ok=True)
r=pd.read_csv(OUT/'tables'/'object_level_results.csv'); s=pd.read_csv(OUT/'tables'/'abcd_summary.csv'); a=pd.read_csv(OUT/'tables'/'ablation_summary.csv')
colors={'A':'#6477d9','B':'#2ba67a','C':'#e39d30','D':'#9b63bf'}
# Overall known-shape metrics.
g=r[r.known].groupby('configuration').agg(correct=('correct_bin','mean'),first=('first_grasp_success','mean'),attempts=('attempts','mean'),time=('time_s','mean')).reindex(list('ABCD'))
fig,ax=plt.subplots(figsize=(10,5.2)); x=np.arange(4); w=.34
ax.bar(x-w/2,g.correct,w,label='Correct-bin success',color='#2b8cbe'); ax.bar(x+w/2,g['first'],w,label='First-grasp success',color='#7bccc4')
for i,row in enumerate(g.itertuples()): ax.text(i-w/2,row.correct+.015,f'{row.correct:.1%}',ha='center',fontsize=9); ax.text(i+w/2,row.first+.015,f'{row.first:.1%}',ha='center',fontsize=9)
ax.set_xticks(x,list('ABCD')); ax.set_ylim(0,1.08); ax.set_ylabel('Rate'); ax.set_title('Known-shape aggregate (1,487 paired objects/configuration)'); ax.grid(axis='y',alpha=.2); ax.legend(ncol=2); fig.tight_layout(); fig.savefig(FIG/'known_aggregate.png',dpi=190); plt.close(fig)
# Open-set behavior.
o=s[s.condition=='open_set'].set_index('configuration').reindex(list('ABCD')); fig,ax=plt.subplots(figsize=(10,5.2));
ax.bar(x-w/2,o.correct_bin_rate,w,label='Known correct-bin',color='#2b8cbe'); ax.bar(x+w/2,o.unknown_false_accept_rate.fillna(0),w,label='Unknown false acceptance',color='#d95f0e')
for i,row in enumerate(o.itertuples()): ax.text(i-w/2,row.correct_bin_rate+.015,f'{row.correct_bin_rate:.1%}',ha='center',fontsize=9); ax.text(i+w/2,(row.unknown_false_accept_rate if pd.notna(row.unknown_false_accept_rate) else 0)+.015,f'{row.unknown_false_accept_rate:.1%}',ha='center',fontsize=9)
ax.set_xticks(x,list('ABCD')); ax.set_ylim(0,1.08); ax.set_ylabel('Rate'); ax.set_title('Open-set evaluation: safety vs sorting quality'); ax.grid(axis='y',alpha=.2); ax.legend(ncol=2); fig.tight_layout(); fig.savefig(FIG/'open_set.png',dpi=190); plt.close(fig)
# Ablation.
o=a[a.condition=='open_set'].copy(); names=['D_full','D_no_geometry_novelty','D_no_reobserve','D_no_learned_selector','D_no_TTA_proxy']; o=o.set_index('configuration').reindex(names); fig,ax=plt.subplots(figsize=(11,5.5)); xx=np.arange(len(o));
ax.bar(xx-w/2,o.unknown_false_accept_rate,w,color='#d95f0e',label='Unknown false acceptance'); ax.bar(xx+w/2,o.first_grasp_success_rate,w,color='#31a354',label='Known first-grasp success'); ax.set_xticks(xx,['Full','No geometry OOD','No reobserve','No learned selector','No TTA proxy'],rotation=18,ha='right'); ax.set_ylim(0,1.05); ax.set_title('Ablation on open-set scene'); ax.grid(axis='y',alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(FIG/'ablation_open_set.png',dpi=190); plt.close(fig)
# Architecture figure.
fig,ax=plt.subplots(figsize=(13,4.2)); ax.axis('off'); boxes=[('RGB-D','Height mask\n+ geometry'),('Shape AI','MLP + TTA\nconfidence'),('Open set','Mahalanobis\n+ damage rules'),('Action','GRASP /\nREOBSERVE / HOLD'),('Candidates','XY / yaw /\nwidth / force'),('Predictor','P(success)\nfrom experience'),('Verify','finger width\n+ table clear'),('Place','hold +\nbin position')]
xs=np.linspace(.03,.89,len(boxes));
for i,((title,sub),x0) in enumerate(zip(boxes,xs)):
    ax.add_patch(plt.Rectangle((x0,.33),.105,.40,facecolor='#e8f1f5' if i<4 else '#f2eaf5',edgecolor='#35586c',lw=1.5)); ax.text(x0+.0525,.61,title,ha='center',va='center',weight='bold',fontsize=10); ax.text(x0+.0525,.44,sub,ha='center',va='center',fontsize=8.5)
    if i<len(boxes)-1: ax.annotate('',xy=(xs[i+1],.53),xytext=(x0+.105,.53),arrowprops=dict(arrowstyle='->',lw=1.5,color='#555'))
ax.text(.5,.12,'Failure-specific recovery feeds every labelled outcome back to grasp_experience.jsonl',ha='center',fontsize=10,color='#7a3e00'); ax.annotate('',xy=(.57,.31),xytext=(.82,.20),arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.35',color='#a45a16',lw=1.5)); fig.tight_layout(); fig.savefig(FIG/'architecture.png',dpi=190,bbox_inches='tight'); plt.close(fig)
# Failure examples as a compact evidence panel from actual logged rows.
examples=[]
for failure in ['empty_grasp','slip','contact_miss','held_unknown','held_low_utility']:
    q=r[r.failure==failure]
    if len(q): examples.append(q.iloc[0])
fig,axes=plt.subplots(1,len(examples),figsize=(3.1*len(examples),3.8));
if len(examples)==1: axes=[axes]
for ax,row in zip(axes,examples):
    ax.axis('off'); c='#f0f4f7' if str(row.failure).startswith('held') else '#fff0e8'; ax.add_patch(plt.Rectangle((.04,.06),.92,.88,facecolor=c,edgecolor='#6b6b6b'))
    ax.text(.5,.83,str(row.failure).replace('_','\n'),ha='center',va='center',weight='bold',fontsize=12)
    lines=[f"cfg {row.configuration} / {row.condition}",f"shape {row['shape']} -> {row.predicted_label}",f"candidate {row.candidate_id}",f"attempts {row.attempts}",f"conf {row.confidence:.2f}",f"novelty {row.novelty_score:.2f}"]
    ax.text(.12,.62,'\n'.join(lines),ha='left',va='top',fontsize=9,linespacing=1.45)
fig.suptitle('Representative logged failure/hold cases',fontsize=14,weight='bold'); fig.tight_layout(); fig.savefig(FIG/'failure_examples.png',dpi=190); plt.close(fig)
# Refresh video preview.
p=OUT/'videos'/'adaptive_benchmark_replay.mp4'; cap=cv2.VideoCapture(str(p)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.set(cv2.CAP_PROP_POS_FRAMES,max(n//2,0)); ok,frame=cap.read();
if ok: cv2.imwrite(str(OUT/'videos'/'adaptive_benchmark_replay_preview.png'),frame)
print(json.dumps({'figures':[p.name for p in sorted(FIG.glob('*.png'))],'video_preview':ok},indent=2))
