from __future__ import annotations
from pathlib import Path
import json, textwrap
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'out'/'experiments'
REPORT=ROOT/'report'/'AI_Shape_Sorting_Robot_Arm_Ultra_Report.pdf'
REPORT.parent.mkdir(parents=True,exist_ok=True)
FONT='/usr/share/fonts/truetype/noto/NotoSansCJK-VF.ttf.ttc'
pdfmetrics.registerFont(TTFont('NotoJP',FONT,subfontIndex=0))

PAGE_W,PAGE_H=A4
NAVY=colors.HexColor('#18324A'); BLUE=colors.HexColor('#2B6F8E'); TEAL=colors.HexColor('#2E8B78'); PURPLE=colors.HexColor('#76528C'); ORANGE=colors.HexColor('#D07A24'); LIGHT=colors.HexColor('#EDF3F6'); MID=colors.HexColor('#D7E3E8'); RED=colors.HexColor('#B34A3C'); DARK=colors.HexColor('#2B3035')

class Report(BaseDocTemplate):
    def __init__(self,path):
        super().__init__(str(path),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=20*mm,bottomMargin=18*mm,title='AI Shape Sorting Robot Arm - Ultra Adaptive Edition')
        frame=Frame(self.leftMargin,self.bottomMargin,self.width,self.height,id='normal')
        self.addPageTemplates(PageTemplate(id='all',frames=frame,onPage=self.decorate))
    def decorate(self,canvas,doc):
        canvas.saveState(); canvas.setFillColor(NAVY); canvas.rect(0,PAGE_H-11*mm,PAGE_W,11*mm,fill=1,stroke=0)
        canvas.setFont('NotoJP',7.7); canvas.setFillColor(colors.white); canvas.drawString(18*mm,PAGE_H-7.2*mm,'PHYSICAL AI / ULTRA ADAPTIVE SORTING')
        canvas.setFillColor(colors.HexColor('#6A747C')); canvas.drawString(18*mm,9*mm,'Reproducible benchmark report - seed 37')
        canvas.drawRightString(PAGE_W-18*mm,9*mm,f'PAGE {doc.page}'); canvas.restoreState()

styles=getSampleStyleSheet()
def ps(name,size,leading=None,color=DARK,spaceAfter=4,spaceBefore=0,align=TA_LEFT):
    return ParagraphStyle(name,parent=styles['BodyText'],fontName='NotoJP',fontSize=size,leading=leading or size*1.48,textColor=color,spaceAfter=spaceAfter*mm,spaceBefore=spaceBefore*mm,alignment=align,wordWrap='CJK')
TITLE=ps('title',25,32,NAVY,5,0); SUB=ps('sub',12,18,BLUE,5); H1=ps('h1',17,23,NAVY,4,3); H2=ps('h2',12.5,18,BLUE,2.5,2); BODY=ps('body',9.2,14,DARK,3); SMALL=ps('small',7.6,11,colors.HexColor('#46515A'),2); CAP=ps('cap',7.6,11,colors.HexColor('#59666F'),3,1,TA_CENTER); WHITE=ps('white',9,13,colors.white,0); CALLOUT=ps('callout',9.1,14,NAVY,1)

def P(text,style=BODY): return Paragraph(text,style)
def img(path,width=172*mm):
    from PIL import Image as PILImage
    im=PILImage.open(path); ratio=im.height/im.width
    return Image(str(path),width=width,height=width*ratio)
def table(data,widths,header=True,font=7.5):
    converted=[]
    for r,row in enumerate(data):
        converted.append([cell if hasattr(cell,'wrapOn') else Paragraph(str(cell),SMALL if r or not header else ParagraphStyle('th',parent=SMALL,textColor=colors.white,alignment=TA_CENTER)) for cell in row])
    t=Table(converted,colWidths=widths,repeatRows=1 if header else 0,hAlign='LEFT')
    cmd=[('FONTNAME',(0,0),(-1,-1),'NotoJP'),('FONTSIZE',(0,0),(-1,-1),font),('LEADING',(0,0),(-1,-1),font*1.35),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#B7C6CD'))]
    if header: cmd += [('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),colors.white)]
    for r in range(1 if header else 0,len(data)):
        if r%2==0: cmd.append(('BACKGROUND',(0,r),(-1,r),colors.HexColor('#F4F7F8')))
    t.setStyle(TableStyle(cmd)); return t

def banner(title,text,color=TEAL):
    data=[[P(title,ParagraphStyle('bt',parent=BODY,textColor=colors.white,fontSize=11,leading=15,spaceAfter=0)),P(text,WHITE)]]
    t=Table(data,colWidths=[45*mm,121*mm]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(0,0),color),('BACKGROUND',(1,0),(1,0),colors.HexColor('#F0F5F6')),('TEXTCOLOR',(1,0),(1,0),DARK),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8)])); return t

def pct(x): return '-' if pd.isna(x) else f'{100*x:.1f}%'

records=pd.read_csv(OUT/'tables'/'object_level_results.csv'); summary=pd.read_csv(OUT/'tables'/'abcd_summary.csv'); ablation=pd.read_csv(OUT/'tables'/'ablation_summary.csv'); registry=json.loads((ROOT/'out'/'run_registry.json').read_text()); q=json.loads((ROOT/'out'/'genesis'/'qualification.json').read_text())
overall=records[records.known].groupby('configuration').agg(correct=('correct_bin','mean'),first=('first_grasp_success','mean'),attempts=('attempts','mean'),mis=('mis_sort','mean'),hold=('held','mean'),time=('time_s','mean'),reobs=('reobservations','mean'),n=('known','size')).reindex(list('ABCD'))

story=[]
# Cover
story += [Spacer(1,14*mm),P('AI SHAPE SORTING<br/>ROBOT ARM',TITLE),P('Ultra Adaptive Edition',ParagraphStyle('accent',parent=TITLE,textColor=PURPLE,fontSize=21,leading=28)),P('不確実性と失敗経験から把持戦略を改善する適応型AI仕分けロボット',SUB),Spacer(1,7*mm),img(OUT/'figures'/'architecture.png',170*mm),Spacer(1,5*mm),banner('今回の到達点','多候補把持、経験学習、未知形状保留、再観測、保持/配置検証を実装し、A/B/C/D 6,400物体と機能除去2,000物体を固定seedで評価。',TEAL),Spacer(1,4*mm),banner('重要な検証境界','本PDFの定量結果は制御ロジック用の確率的接触ベンチマークであり、Genesis物理成功率ではない。管理環境でtorchを取得できず、Genesis / IK / 接触把持 / Genesis動画は未実行。',RED),Spacer(1,8*mm),P('PROJECT VERSION 3.0.0  |  RUN ultra-seed37-20261030  |  2026-10-30',SMALL),P('個人情報・所属・メールアドレスは記載していない。',SMALL),PageBreak()]

story += [P('1. エグゼクティブサマリー',H1),P('既存上級編の形状認識・明示的3D逆投影・IK/PD・二重把持判定・失敗別復帰を維持し、把持を単一ルールから「候補生成 -> 成功予測 -> 行動判断」へ拡張した。関節軌道全体の強化学習は行わず、既存制御器の前段だけを学習化している。'),img(OUT/'figures'/'known_aggregate.png',166*mm),P('図1. 既知形状1,487 paired objects / 構成の全条件集計。',CAP)]
head=[['構成','正しいビン','初回把持','平均試行','誤仕分け','保留','平均時間']]
for cfg,row in overall.iterrows(): head.append([cfg,pct(row.correct),pct(row['first']),f'{row.attempts:.2f}',pct(row.mis),pct(row.hold),f'{row.time:.2f}s'])
story += [table(head,[15*mm,27*mm,27*mm,23*mm,22*mm,22*mm,27*mm]),Spacer(1,3*mm),P('<b>主要所見。</b> BはAより初回把持が+8.0pt、正しいビンが+1.1pt、平均時間が-1.57s。Cは既知形状全体の正しいビン90.5%で最高だが、深度ノイズとcombined unseenでは再観測コストが増えた。Dは既知形状性能をほぼ維持しながら未知物体誤受け入れを100%から5.3%へ下げた。'),P('<b>解釈。</b> 学習型候補選択の効果は初回把持に明瞭。一方、再観測は観測品質が行動を変える場合に限って価値があり、極端な複合外乱では時間増になる。未知保留は安全性を大きく改善するが、既知分布の外側にある正常物体も保留し得る。'),PageBreak()]

story += [P('2. 実装アーキテクチャ',H1),img(OUT/'figures'/'architecture.png',172*mm),P('図2. 認識、open-set判定、行動、候補選択、検証、経験記録の閉ループ。',CAP),P('2.1 把持候補と成功予測',H2),P('各観測に対し8候補を生成する。候補はXYオフセット、顎yaw、pre-close指幅、把持力を同時に変える。入力は形状one-hot、confidence / margin / entropy、novelty、circularity / solidity / aspect / vertices、物体高、depth MAD、edge clearance、摩擦推定、位置/向き不確実性、物体間隔、候補パラメータ、試行番号の24次元。HistGradientBoostingClassifierで成功確率を予測し、動作量ペナルティを引いた効用最大候補を選ぶ。'),P('学習データは12,000候補行。9,600行を学習、2,400行を独立hold-outとし、形状は既知3種のみ。hold-out ROC-AUC 0.597、average precision 0.815、Brier 0.170であり、ランキング能力は限定的だが、候補選択後の初回把持はAより改善した。モデル指標とタスク指標を分けて解釈する。'),P('2.2 再観測・保留・検証',H2),table([['判断','実装'],['REOBSERVE','Frankaをhomeへ完全退避し、同じ固定上方RGB-Dから再撮影。最大1回。'],['HOLD','幾何support外、edge damage、または全候補の成功見込みが低い場合に把持しない。'],['GRASP','学習候補を既存IK/PDへ渡し、接触後に指幅 AND table-clearを要求。'],['Transport/Place','運搬後の指幅低下でslipを検出し、解放後は真値位置を評価専用に使ってbin半径を確認。']], [31*mm,135*mm]),PageBreak()]

story += [P('3. 未知形状検出',H1),P('closed-set MLPの最大確率だけでは未知形状を安全に拒否できるとは仮定していない。既知3クラスの幾何4特徴にLedoit-Wolf共分散を当て、予測クラス条件付きMahalanobis距離を求める。98.5% conformal calibration thresholdに、レンダリングdomain shift用の固定1.52倍marginを掛ける。さらにsolidity < 0.72またはedge damage > 0.22を拒否する。closed-set confidenceは記録するが単独では判定しない。'),img(OUT/'figures'/'open_set.png',166*mm),P('図3. open-set（circle/square/triangle + star/pentagon/crescent/damaged）。Dのみ幾何ベース保留を有効化。',CAP),table([['指標','A','B','C','D'],['既知 correct-bin','97.7%','97.7%','97.7%','97.7%'],['既知 初回把持','51.7%','66.7%','66.7%','66.7%'],['未知誤受け入れ','100.0%','100.0%','100.0%','5.3%'],['全体保留率','0.0%','0.0%','0.0%','53.5%'],['平均時間','18.35s','15.39s','15.87s','7.86s']], [38*mm,26*mm,26*mm,26*mm,26*mm]),P('Dの全体保留率53.5%は、open-setシーンの未知比率56.5%に近い。未知を作業せず保留するため、平均時間が短くなった。これはスループット向上というより、安全側の早期終了である。'),PageBreak()]

story += [P('4. 実験計画',H1),P('4.1 学習・評価分離',H2),table([['区分','形状','位置noise','回転noise','深度noise','摩擦','間隔'],['Training','既知3種','U[1,8] mm','U[0,12] deg','U[0.5,3.5] mm','U[0.45,0.85]','U[70,130] mm'],['Evaluation','既知+未知','2-14 mm','3-28 deg','0.8-8 mm','0.25-0.65','42-110 mm']], [22*mm,28*mm,25*mm,27*mm,27*mm,25*mm,26*mm]),P('評価はnominal、position shift、rotation shift、depth noise、low friction、crowded、combined unseen、open setの8条件。各条件200物体、同一object seedをA/B/C/Dで共有した。A/B/C/Dは計6,400物体、ablationはcombined unseenとopen setで計2,000物体。'),P('4.2 指標と統計',H2),P('正しいビンへの配置成功率、初回把持成功率、完了までの試行数、誤仕分け率、保留率、作業時間、再観測回数、未知物体誤受け入れ率をobject-levelで保存した。summaryには2,000回bootstrapの95%区間を収録。paired_differences.csvは同じseedのAとの差を保持する。'),P('4.3 成功ラベル',H2),P('候補学習のsuccessは、接触成立、指幅閾値、table-clear、運搬保持、配置位置のすべてを満たした場合だけ1。分類真値は評価にだけ使い、候補選択・再観測・保留には入力しない。'),banner('注意','接触結果は確率的代理モデルで生成した。Genesis物理エンジンの接触・摩擦・関節挙動ではないため、数値を実機またはGenesis成功率として報告してはならない。',ORANGE),PageBreak()]

story += [P('5. A/B/C/D結果',H1),img(OUT/'figures'/'abcd_comparison.png',172*mm),P('図4. 8評価条件の比較。unknown false acceptanceはopen-setのみ定義。',CAP)]
# compact condition table
rows=[['条件','A correct','B correct','C correct','D correct','D hold','D reobs']]
for cond in ['nominal','position_shift','rotation_shift','depth_noise','low_friction','crowded','combined_unseen','open_set']:
    x=summary[summary.condition==cond].set_index('configuration'); rows.append([cond,pct(x.loc['A','correct_bin_rate']),pct(x.loc['B','correct_bin_rate']),pct(x.loc['C','correct_bin_rate']),pct(x.loc['D','correct_bin_rate']),pct(x.loc['D','hold_rate']),f"{x.loc['D','mean_reobservations']:.2f}"])
story += [table(rows,[34*mm,22*mm,22*mm,22*mm,22*mm,22*mm,22*mm]),Spacer(1,3*mm),P('<b>未学習条件。</b> low frictionではA 76.0%に対しC/D 82.5%。crowdedではA 96.5%、B/C/D 97.5%。combined unseenではB 54.0%が最高で、C 52.5%、D 50.0%。複合外乱では再観測が0.99回/物体まで増え、時間もC 30.97sとなった。'),P('<b>誤仕分け。</b> 既知形状全体ではA/Bが0.07%、C/Dが0%。open-setの未知誤受け入れはDで5.3%。正しいビン成功だけを見るとopen-setの未知安全性を見落とすため、両方を併記する必要がある。'),PageBreak()]

story += [P('6. 機能除去比較',H1),img(OUT/'figures'/'ablation_open_set.png',167*mm),P('図5. open-set ablation。geometry noveltyを外すと未知誤受け入れが99.1%へ戻る。',CAP)]
ablrows=[['構成','条件','correct-bin','初回把持','保留','未知誤受入','時間']]
for _,r in ablation.iterrows():
    if r['condition']=='open_set': ablrows.append([r.configuration,'open set',pct(r.correct_bin_rate),pct(r.first_grasp_success_rate),pct(r.hold_rate),pct(r.unknown_false_accept_rate),f'{r.mean_time_s:.2f}s'])
story += [table(ablrows,[42*mm,23*mm,24*mm,24*mm,21*mm,25*mm,24*mm]),P('geometry noveltyの除去が最大の安全劣化を生んだ。learned selector除去ではopen-set既知の初回把持が73.6%から59.8%へ低下。no-TTA proxyはcombined unseenの保留を増やし、correct-binを44.0%から31.0%へ低下させた。再観測除去はcombined unseenの時間を短くするが、完全版との成功差は小さく、再観測閾値の再調整余地がある。'),PageBreak()]

story += [P('7. 失敗例と復帰',H1),img(OUT/'figures'/'failure_examples.png',172*mm),P('図6. object-level logから抽出した代表ケース。',CAP),table([['失敗/保留','検出信号','復帰/結果'],['empty_grasp','接触後の指幅不足、または元位置table-clear不成立','XY/yaw/forceを変え、最大3回まで再候補化'],['slip','把持直後から運搬後の指幅が4 mm超低下','安全上限内でforceを増加し、home退避後に再試行'],['contact_miss','候補中心と実物位置の残差が接触許容を超過','offset候補へ切替'],['held_unknown','geometry support外、低solidity、edge damage','把持せず保留。人確認または別タスクへ'],['held_low_utility','全候補の予測成功率が安全閾値未満','無理な把持を避けて保留']], [34*mm,68*mm,64*mm]),P('失敗ログはcandidateと観測を同じJSONL行に保持し、成功だけでなく失敗経験も次回学習へ戻せる。今回の学習器はオンライン更新を行わず、run終了後の再学習を想定している。これにより本番中の分布ドリフトで方策が急変する危険を避けた。'),PageBreak()]

story += [P('8. Genesis資格試験と未解決事項',H1),banner('QUALIFICATION: NOT PASSED','torch / genesis import不可。physics step、Franka IK、接触把持、camera.render、Genesis動画は実行されていない。',RED),P('`genesis3d/qualify_genesis.py` は次の順に検査する: (1) torch/genesis import、(2) CPU scene build、(3) plane + free boxを25 step、(4) Franka MJCF読込、(5) hand linkへのIKがfinite。今回のqualification.jsonは最初のimportでModuleNotFoundError: torchを記録した。管理package feedにはtorch配布がなく、外部index probeでも取得できなかった。'),table([['固定項目','値'],['Target Genesis','0.3.3（未検証target）'],['Target torch','2.7.1（未検証target）'],['Backend','CPU'],['seed','37'],['dt / substeps','0.01 s / 4'],['gravity','[0, 0, -9.81] m/s²'],['Franka','xml/franka_emika_panda/panda.xml'],['Camera','640x480, vertical FOV 50 deg, fixed overhead']], [52*mm,114*mm]),P('Genesisが利用できる環境では、qualification.jsonの`qualified=true`を確認してから`genesis_sort_ultra.py`を実行する。本番runnerには学習候補選択、幾何OOD、arm-clear再観測、指幅+table-clear、運搬保持、配置後位置、経験JSONLを統合済み。ただしAPI差、摩擦surface、finger index、カメラdepth単位は実走で再確認が必要。'),P('課題提出には1-3分のロボット実行動画が必須。付属64秒動画はログ再生であり、この要件のGenesis代替にはしない。'),PageBreak()]

story += [P('9. 再現性・成果物',H1),table([['成果物','内容'],['コード','Ultra統合runner、学習/未知/判断モジュール、tests 13件'],['設定','configs/ultra.json。scene / physics / threshold / evaluation'],['モデル','shape MLP、grasp success HGB、geometry novelty'],['結果','6,400 object A/B/C/D、2,000 object ablation、bootstrap区間'],['失敗例','failure_taxonomy.csv + failure_examples.png'],['動画','64秒 H.264 benchmark replay（Genesisではない）'],['来歴','run_registry.json + manifest.jsonのSHA-256']], [41*mm,125*mm]),P('再実行コマンド',H2),P('<font name="NotoJP">pip install -r requirements.txt<br/>python -m pytest -q<br/>python test_offline.py --out out/offline_ultra<br/>python run_experiments.py --seed 37 --training-rows 12000 --objects-per-condition 200<br/>python make_report_assets.py</font>',ParagraphStyle('code',parent=SMALL,backColor=colors.HexColor('#F1F3F4'),borderPadding=8,leading=14)),P('提出要件との対応',H2),table([['要件','状態','根拠'],['AIを用いたロボットタスク','実装済','MLP + grasp success model + open-set detector'],['コード一式','完了','ZIP内に全ソース、設定、tests、models'],['レポートPDF','完了','本PDF'],['1-3分ロボット動画','未完','64秒benchmark replayはあるがGenesis実映像ではない'],['個人情報を含めない','確認済','成果物に氏名・所属・メールを記載していない']], [55*mm,26*mm,85*mm]),P('結論',H2),P('超上級編の中核である「失敗経験から候補を選ぶ」「不確実なら再観測する」「未知なら保留する」は、コード・モデル・比較結果・ablation・失敗ログとして一貫して実装された。最終提出品質にする残作業はGenesis依存環境での資格試験、接触パラメータ調整、同じA/B/C/D seedでの物理再実験、実映像収録である。')]

# Full appendix table, balanced across two pages rather than leaving a short spill page.
def appendix_rows(configs):
    full=[['Cfg','Condition','Correct','1st grasp','Attempts','Mis-sort','Hold','Time','Reobs','Unknown FA']]
    part=summary[summary.configuration.isin(configs)].sort_values(['configuration','condition'])
    for _,r in part.iterrows(): full.append([r.configuration,r.condition,pct(r.correct_bin_rate),pct(r.first_grasp_success_rate),f'{r.mean_attempts:.2f}',pct(r.mis_sort_rate),pct(r.hold_rate),f'{r.mean_time_s:.2f}',f'{r.mean_reobservations:.2f}',pct(r.unknown_false_accept_rate)])
    return full
story += [PageBreak(),P('付録A. A/B/C/D 全条件メトリクス (A/B)',H1),table(appendix_rows(['A','B']),[10*mm,32*mm,18*mm,18*mm,16*mm,17*mm,15*mm,16*mm,14*mm,20*mm],font=6.5),PageBreak(),P('付録A. A/B/C/D 全条件メトリクス (C/D)',H1),table(appendix_rows(['C','D']),[10*mm,32*mm,18*mm,18*mm,16*mm,17*mm,15*mm,16*mm,14*mm,20*mm],font=6.5),Spacer(1,3*mm),P('95% bootstrap区間はout/experiments/tables/abcd_summary.csvに全て保存。個別予測、候補、失敗、seedはobject_level_results.csvを参照。',SMALL)]

Report(REPORT).build(story)
print(REPORT)
