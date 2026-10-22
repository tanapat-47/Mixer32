import sys, os, json, math
import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.signal import butter, sosfilt, sosfilt_zi
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
 QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QGridLayout,QPushButton,
 QLabel,QSlider,QFileDialog,QMessageBox,QScrollArea,QGroupBox,QDoubleSpinBox,
 QCheckBox,QLineEdit,QComboBox,QSpinBox,QInputDialog,QTabWidget,QFormLayout,QFrame
)

SR=48000
BLOCK=512
EXTS=(".wav",".flac",".aiff",".aif",".ogg")

class Track:
 def __init__(self,i):
  self.n=i+1; self.name=f"CH {i+1:02d}"; self.path=""; self.audio=None; self.pos=0
  self.gain=0.8; self.pan=0.0; self.mute=False; self.solo=False; self.phase=False
  self.hpf=35.0; self.gate_on=False; self.gate_thr=-55.0; self.gate_ratio=4.0
  self.comp_on=False; self.comp_thr=-18.0; self.comp_ratio=3.0; self.comp_attack=10.0; self.comp_release=100.0
  self.eq=[0.0,0.0,0.0,0.0]; self.eq_freq=[100.0,350.0,2500.0,10000.0]; self.eq_q=[0.7,1.0,1.0,0.7]
  self.sends=[0.0]*8; self.send_pre=[False]*8; self.dca=0; self.mute_group=0; self.audio_channels=2
  self.env=1.0; self.gate_env=1.0; self.filters={}; self.eq_filters={}

class Bus:
 def __init__(self,i):
  self.i=i; self.name=f"BUS {i+1}"; self.gain=0.8; self.pan=0.0; self.mute=False
  self.comp_on=False; self.comp_thr=-18.0; self.comp_ratio=3.0; self.eq=[0.,0.,0.,0.]
  self.audio=np.zeros((BLOCK,2),np.float32)

class FX:
 def __init__(self,i):
  self.i=i; self.name=["Room Reverb","Hall Reverb","Plate Reverb","Delay","Echo","Chorus","Flanger","Pitch/Utility"][i]
  self.kind=["reverb","reverb","reverb","delay","delay","chorus","chorus","utility"][i]
  self.mix=0.2; self.time=0.35; self.feedback=0.3; self.damping=0.5; self.enabled=True
  self.buf=np.zeros((SR*2,2),np.float32); self.p=0; self.phase=0.0

class Engine:
 def __init__(self):
  self.tracks=[Track(i) for i in range(32)]; self.buses=[Bus(i) for i in range(16)]
  self.fx=[FX(i) for i in range(8)]; self.master=0.85; self.playing=False; self.stream=None
  self.lock=False; self.last_peak=0.; self.master_mute=False; self.mute_groups=[False]*8
  self.dca_gains=[1.0]*8; self.dca_mutes=[False]*8
  self.main_eq=[0.0]*4; self.main_comp=False; self.main_comp_thr=-18.; self.main_comp_ratio=2.
  self.main_delay_ms=0.; self.main_delay_buf=np.zeros((SR,2),np.float32); self.main_delay_pos=0
  self.playhead=0; self.loop=False; self.metronome=False; self.metro_phase=0; self.bpm=120; self.beats=4
  self._main_filters={}; self._bus_filters=[{} for _ in range(16)]
 def load(self,idx,path):
  a,sr=sf.read(path,dtype="float32",always_2d=True)
  if sr!=SR:
   n=max(1,int(len(a)*SR/sr)); x=np.linspace(0,1,len(a),endpoint=False); xx=np.linspace(0,1,n,endpoint=False)
   a=np.stack([np.interp(xx,x,a[:,c]) for c in range(a.shape[1])],axis=1).astype(np.float32)
  if a.shape[1]==1: a=np.repeat(a,2,axis=1)
  elif a.shape[1]>2: a=a[:,:2]
  t=self.tracks[idx]; t.audio=a; t.path=path; t.pos=0; t.name=os.path.splitext(os.path.basename(path))[0][:28]
 def db(self,x): return 20*np.log10(max(float(x),1e-8))
 def gain_db(self,db): return 10**(db/20)
 def biquad(self,x,typ,freq,q,gain_db,store,key):
  freq=float(np.clip(freq,20,SR*.45)); q=max(.1,float(q))
  if key not in store:
   try:
    if typ=="low": sos=butter(2,freq,btype="lowpass",fs=SR,output="sos")
    elif typ=="high": sos=butter(2,freq,btype="highpass",fs=SR,output="sos")
    else:
     # Peaking EQ biquad
     A=10**(gain_db/40); w=2*np.pi*freq/SR; alpha=np.sin(w)/(2*q); c=np.cos(w)
     b0=1+alpha*A; b1=-2*c; b2=1-alpha*A; a0=1+alpha/A; a1=-2*c; a2=1-alpha/A
     sos=np.array([[b0/a0,b1/a0,b2/a0,1,a1/a0,a2/a0]],dtype=np.float64)
    zi=np.zeros((sos.shape[0],2,2),dtype=np.float64); store[key]=(sos,zi)
   except Exception: return x
  sos,zi=store[key]
  try:
   y,z=sosfilt(sos,x,axis=0,zi=zi); store[key]=(sos,z); return y.astype(np.float32)
  except Exception: return x
 def process_track(self,t,frames):
  out=np.zeros((frames,2),np.float32)
  if t.audio is None: return out
  end=min(t.pos+frames,len(t.audio)); count=max(0,end-t.pos)
  if count: out[:count]=t.audio[t.pos:end]
  t.pos += frames
  if t.pos>=len(t.audio):
   if self.loop: t.pos=0
   else: t.pos=len(t.audio)
  if t.phase: out *= -1
  if t.hpf>20: out=self.biquad(out,"high",t.hpf,.707,0,t.filters,"hpf")
  # Four parametric bands: low shelf, low-mid peak, high-mid peak, high shelf approximated by shelf-like peaking filters
  for j in range(4):
   if abs(t.eq[j])>.01:
    typ="peak"
    out=self.biquad(out,typ,t.eq_freq[j],t.eq_q[j],t.eq[j],t.eq_filters,f"eq{j}")
  rms=float(np.sqrt(np.mean(out*out)+1e-12)); db=self.db(rms)
  if t.gate_on:
   target=1.0 if db>=t.gate_thr else max(.03,10**((db-t.gate_thr)*max(.1,t.gate_ratio-1)/20))
   t.gate_env=.75*t.gate_env+.25*target; out*=t.gate_env
  if t.comp_on:
   peak=float(np.max(np.abs(out))+1e-9); pdb=self.db(peak)
   if pdb>t.comp_thr:
    target_db=t.comp_thr+(pdb-t.comp_thr)/max(1,t.comp_ratio)
    target=self.gain_db(target_db-pdb)
    t.env=.8*t.env+.2*target; out*=t.env
   else: t.env=.8*t.env+.2
  # dca
  out*=t.gain*self.dca_gains[t.dca]
  if self.dca_mutes[t.dca]: out*=0
  if t.mute or (t.mute_group and self.mute_groups[t.mute_group-1]): out*=0
  if any(x.solo and x.audio is not None for x in self.tracks) and not t.solo: out*=0
  l=math.cos((t.pan+1)*math.pi/4); r=math.sin((t.pan+1)*math.pi/4)
  out[:,0]*=l; out[:,1]*=r
  return out
 def process_fx(self,fx,send):
  if not fx.enabled or not np.any(send): return np.zeros_like(send)
  out=np.zeros_like(send); delay=int(np.clip(fx.time,0.01,1.8)*SR)
  delay=max(1,min(delay,len(fx.buf)-1))
  if fx.kind=="utility": return send*fx.mix
  for i in range(len(send)):
   read=(fx.p-delay)%len(fx.buf); delayed=fx.buf[read].copy()
   if fx.kind=="reverb":
    # lightweight multi-tap feedback reverb approximation
    fb=min(.88,max(.1,fx.feedback))
    fx.buf[fx.p]=send[i]+delayed*fb
    out[i]=delayed*fx.mix
   elif fx.kind=="delay":
    fx.buf[fx.p]=send[i]+delayed*min(.85,max(0,fx.feedback))
    out[i]=delayed*fx.mix
   elif fx.kind in ("chorus","flanger"):
    mod=int((.015 if fx.kind=="chorus" else .003)*SR*(1+math.sin(fx.phase)))
    rd=(fx.p-delay-mod)%len(fx.buf); d=fx.buf[rd].copy()
    fx.buf[fx.p]=send[i]+d*fx.feedback; out[i]=d*fx.mix; fx.phase+=.0004
   fx.p=(fx.p+1)%len(fx.buf)
  return out
 def callback(self,outdata,frames,time_info,status):
  mix=np.zeros((frames,2),np.float32); bus_inputs=[np.zeros_like(mix) for _ in self.buses]
  fx_inputs=[np.zeros_like(mix) for _ in self.fx]
  for t in self.tracks:
   x=self.process_track(t,frames)
   for bi,b in enumerate(self.buses):
    if t.sends[bi]>0:
     source=x if not t.send_pre[bi] else self.pre_send_source(t,frames)
     bus_inputs[bi]+=source*t.sends[bi]
   for fi in range(8):
    if t.sends[fi]>0: fx_inputs[fi]+=x*t.sends[fi]
   mix+=x
  for i,b in enumerate(self.buses):
   bx=bus_inputs[i]
   if b.mute: continue
   bx*=b.gain
   if b.comp_on:
    pk=np.max(np.abs(bx))+1e-9; db=self.db(pk)
    if db>b.comp_thr: bx*=self.gain_db(b.comp_thr+(db-b.comp_thr)/max(1,b.comp_ratio)-db)
   mix+=bx
  for i,fx in enumerate(self.fx): mix+=self.process_fx(fx,fx_inputs[i])
  if self.main_comp:
   pk=np.max(np.abs(mix))+1e-9; db=self.db(pk)
   if db>self.main_comp_thr: mix*=self.gain_db(self.main_comp_thr+(db-self.main_comp_thr)/max(1,self.main_comp_ratio)-db)
  mix*=self.master
  if self.master_mute: mix*=0
  peak=float(np.max(np.abs(mix))+1e-9); self.last_peak=peak
  if peak>.98: mix*=.98/peak
  outdata[:]=np.clip(mix,-1,1)
 def pre_send_source(self,t,frames):
  # Pre-fader approximation uses the track's original source segment.
  if t.audio is None: return np.zeros((frames,2),np.float32)
  start=max(0,t.pos-frames); end=min(start+frames,len(t.audio))
  a=np.zeros((frames,2),np.float32); n=end-start
  if n>0: a[:n]=t.audio[start:end]
  return a
 def start(self):
  if self.stream is None: self.stream=sd.OutputStream(samplerate=SR,channels=2,blocksize=BLOCK,dtype="float32",callback=self.callback)
  if not self.stream.active: self.stream.start()
  self.playing=True
 def stop(self):
  if self.stream and self.stream.active: self.stream.stop()
  self.playing=False
 def close(self):
  if self.stream: self.stream.close(); self.stream=None

class Strip(QGroupBox):
 def __init__(self,eng,i,win):
  super().__init__(); self.e=eng; self.i=i; self.w=win; self.t=eng.tracks[i]
  self.setObjectName("channelStrip"); self.setMinimumWidth(150); self.setMaximumWidth(230)
  self.setTitle(""); v=QVBoxLayout(self); v.setContentsMargins(8,9,8,8); v.setSpacing(6)
  self.chlabel=QLabel(f"CH {i+1:02d}"); self.chlabel.setObjectName("channelNumber"); self.chlabel.setAlignment(Qt.AlignmentFlag.AlignCenter); v.addWidget(self.chlabel)
  self.name=QLineEdit(self.t.name); self.name.setObjectName("channelName"); self.name.setPlaceholderText("ตั้งชื่อช่อง..."); self.name.setAlignment(Qt.AlignmentFlag.AlignCenter); self.name.editingFinished.connect(self.rename); v.addWidget(self.name)
  self.file=QLabel("NO AUDIO"); self.file.setObjectName("trackStatus"); self.file.setAlignment(Qt.AlignmentFlag.AlignCenter); self.file.setWordWrap(True); self.file.setMinimumHeight(26); v.addWidget(self.file)
  b=QPushButton("＋ LOAD AUDIO"); b.clicked.connect(self.load); v.addWidget(b)
  frow=QHBoxLayout(); frow.setSpacing(6)
  self.db_label=QLabel("−1.0 dB"); self.db_label.setObjectName("faderValue"); self.db_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
  self.fader=QSlider(Qt.Orientation.Vertical); self.fader.setRange(-600,100); self.fader.setValue(-19); self.fader.setMinimumHeight(175); self.fader.setObjectName("channelFader"); self.fader.valueChanged.connect(self.set_gain)
  frow.addStretch(1); frow.addWidget(self.fader); frow.addWidget(self.db_label); frow.addStretch(1); v.addLayout(frow,1)
  panrow=QHBoxLayout(); panrow.addWidget(QLabel("L")); self.pan=QSlider(Qt.Orientation.Horizontal); self.pan.setRange(-100,100); self.pan.valueChanged.connect(lambda x:setattr(self.t,"pan",x/100)); panrow.addWidget(self.pan,1); panrow.addWidget(QLabel("R")); v.addLayout(panrow)
  row=QHBoxLayout(); self.mute=QPushButton("MUTE"); self.mute.setCheckable(True); self.mute.setObjectName("muteButton"); self.mute.toggled.connect(lambda x:setattr(self.t,"mute",x)); self.solo=QPushButton("SOLO"); self.solo.setCheckable(True); self.solo.setObjectName("soloButton"); self.solo.toggled.connect(lambda x:setattr(self.t,"solo",x)); row.addWidget(self.mute); row.addWidget(self.solo); v.addLayout(row)
  self.phase=QCheckBox("Ø Polarity"); self.phase.toggled.connect(lambda x:setattr(self.t,"phase",x)); v.addWidget(self.phase)
  self.detail_btn=QPushButton("OPEN CHANNEL DETAIL  ›"); self.detail_btn.setObjectName("detailButton"); self.detail_btn.clicked.connect(lambda:self.w.open_channel_detail(self.i)); v.addWidget(self.detail_btn)
  # Detailed processing page: only shown after opening a channel.
  self.detail_widget=QWidget(); dv=QVBoxLayout(self.detail_widget); dv.setContentsMargins(12,12,12,12); dv.setSpacing(10)
  head=QHBoxLayout(); back=QPushButton("‹  BACK TO CHANNELS"); back.clicked.connect(lambda:self.w.switch_view(0)); head.addWidget(back)
  title=QLabel(f"CHANNEL {i+1:02d}  /  {self.t.name}"); title.setObjectName("detailTitle"); head.addWidget(title,1); dv.addLayout(head); self.detail_title=title
  self.detail_tabs=QTabWidget(); self.detail_tabs.setObjectName("detailTabs"); dv.addWidget(self.detail_tabs,1)
  gatepage=QWidget(); gl=QVBoxLayout(gatepage); gatebox=QGroupBox("HIGH-PASS FILTER"); gf=QFormLayout(gatebox)
  self.hpf=QSpinBox(); self.hpf.setRange(20,1000); self.hpf.setValue(35); self.hpf.setSuffix(" Hz"); self.hpf.valueChanged.connect(lambda x:setattr(self.t,"hpf",float(x))); gf.addRow("Low cut / HPF",self.hpf); gl.addWidget(gatebox)
  gatebox2=QGroupBox("NOISE GATE"); g2=QFormLayout(gatebox2); self.gate=QCheckBox("Enable Gate"); self.gate.toggled.connect(lambda x:setattr(self.t,"gate_on",x)); self.gate_thr=QDoubleSpinBox(); self.gate_thr.setRange(-90,0); self.gate_thr.setValue(-55); self.gate_thr.setSuffix(" dB"); self.gate_thr.valueChanged.connect(lambda x:setattr(self.t,"gate_thr",x)); self.gate_ratio=QDoubleSpinBox(); self.gate_ratio.setRange(1,20); self.gate_ratio.setValue(4); self.gate_ratio.valueChanged.connect(lambda x:setattr(self.t,"gate_ratio",x)); g2.addRow(self.gate); g2.addRow("Threshold",self.gate_thr); g2.addRow("Range / ratio",self.gate_ratio); gl.addWidget(gatebox2); gl.addStretch(1); self.detail_tabs.addTab(gatepage,"GATE / HPF")
  compage=QWidget(); cl=QVBoxLayout(compage); cb=QGroupBox("DYNAMIC PROCESSOR / COMPRESSOR"); cf=QFormLayout(cb); self.comp=QCheckBox("Enable Compressor"); self.comp.toggled.connect(lambda x:setattr(self.t,"comp_on",x)); self.comp_thr=QDoubleSpinBox(); self.comp_thr.setRange(-60,0); self.comp_thr.setValue(-18); self.comp_thr.setSuffix(" dB"); self.comp_thr.valueChanged.connect(lambda x:setattr(self.t,"comp_thr",x)); self.comp_ratio=QDoubleSpinBox(); self.comp_ratio.setRange(1,20); self.comp_ratio.setValue(3); self.comp_ratio.setSuffix(":1"); self.comp_ratio.valueChanged.connect(lambda x:setattr(self.t,"comp_ratio",x)); self.comp_attack=QDoubleSpinBox(); self.comp_attack.setRange(.1,200); self.comp_attack.setValue(10); self.comp_attack.setSuffix(" ms"); self.comp_attack.valueChanged.connect(lambda x:setattr(self.t,"comp_attack",x)); self.comp_release=QDoubleSpinBox(); self.comp_release.setRange(5,2000); self.comp_release.setValue(100); self.comp_release.setSuffix(" ms"); self.comp_release.valueChanged.connect(lambda x:setattr(self.t,"comp_release",x)); cf.addRow(self.comp); cf.addRow("Threshold",self.comp_thr); cf.addRow("Ratio",self.comp_ratio); cf.addRow("Attack",self.comp_attack); cf.addRow("Release",self.comp_release); cl.addWidget(cb); cl.addStretch(1); self.detail_tabs.addTab(compage,"DYN / COMP")
  eqpage=QWidget(); el=QVBoxLayout(eqpage); eqintro=QLabel("4-BAND PARAMETRIC EQ  •  Adjust gain and center frequency for each band"); eqintro.setObjectName("sectionHint"); el.addWidget(eqintro); eqgrid=QGridLayout(); self.eqsp=[]; self.eqfreq=[]
  for j,(label,default) in enumerate(zip(("LOW","LOW MID","HIGH MID","HIGH"),self.t.eq)):
   box=QGroupBox(label); bl=QFormLayout(box); gain=QDoubleSpinBox(); gain.setRange(-15,15); gain.setSingleStep(.5); gain.setValue(default); gain.setSuffix(" dB"); gain.valueChanged.connect(lambda x,ix=j:self.set_eq(ix,x)); freq=QDoubleSpinBox(); freq.setRange(20,20000); freq.setValue(self.t.eq_freq[j]); freq.setSuffix(" Hz"); freq.valueChanged.connect(lambda x,ix=j:self.set_eqfreq(ix,x)); q=QDoubleSpinBox(); q.setRange(.1,10); q.setSingleStep(.1); q.setValue(self.t.eq_q[j]); q.valueChanged.connect(lambda x,ix=j:self.set_eqq(ix,x)); self.eqsp.append(gain); self.eqfreq.append(freq); bl.addRow("Gain",gain); bl.addRow("Frequency",freq); bl.addRow("Q",q); eqgrid.addWidget(box,0,j)
  el.addLayout(eqgrid); el.addStretch(1); self.detail_tabs.addTab(eqpage,"EQ")
  sendpage=QWidget(); sl=QVBoxLayout(sendpage); sendintro=QLabel("BUS SENDS  •  Set the send level for this channel"); sendintro.setObjectName("sectionHint"); sl.addWidget(sendintro); self.sendsp=[]; sg=QGridLayout()
  for bi in range(8):
   box=QGroupBox(f"BUS {bi+1:02d}"); bl=QVBoxLayout(box); value=QLabel("−∞ dB"); value.setAlignment(Qt.AlignmentFlag.AlignCenter); s=QSlider(Qt.Orientation.Vertical); s.setMinimumHeight(150); s.setRange(0,100); s.valueChanged.connect(lambda x,j=bi,lab=value:self.set_send(j,x,lab)); bl.addWidget(value); bl.addWidget(s,alignment=Qt.AlignmentFlag.AlignHCenter); self.sendsp.append(s); sg.addWidget(box,0,bi)
  sl.addLayout(sg); route=QHBoxLayout(); self.dca=QComboBox(); self.dca.addItems(["No DCA"]+[f"DCA {x}" for x in range(1,9)]); self.dca.currentIndexChanged.connect(lambda x:setattr(self.t,"dca",max(0,x-1))); self.mg=QComboBox(); self.mg.addItems(["No Mute Group"]+[f"Mute Group {x}" for x in range(1,9)]); self.mg.currentIndexChanged.connect(lambda x:setattr(self.t,"mute_group",x)); route.addWidget(QLabel("DCA")); route.addWidget(self.dca); route.addWidget(QLabel("Mute Group")); route.addWidget(self.mg); route.addStretch(1); sl.addLayout(route); sl.addStretch(1); self.detail_tabs.addTab(sendpage,"SENDS / ROUTING")
 def rename(self):
  self.t.name=self.name.text().strip() or f"CH {self.i+1:02d}"; self.name.setText(self.t.name); self.detail_title.setText(f"CHANNEL {self.i+1:02d}  /  {self.t.name}")
 def set_gain(self,x):
  self.t.gain=10**(x/200); self.db_label.setText(f"{x/10:.1f} dB" if x>-600 else "−∞ dB")
 def set_eq(self,i,x): self.t.eq[i]=x; self.t.eq_filters={}
 def set_eqfreq(self,i,x): self.t.eq_freq[i]=x; self.t.eq_filters={}
 def set_eqq(self,i,x): self.t.eq_q[i]=x; self.t.eq_filters={}
 def set_send(self,i,x,label=None):
  self.t.sends[i]=x/100
  if label is not None: label.setText("−∞ dB" if x==0 else f"{20*math.log10(x/100):.1f} dB")
 def load(self):
  p,_=QFileDialog.getOpenFileName(self,"Load audio","","Audio (*.wav *.flac *.aiff *.aif *.ogg)")
  if p:
   try: self.e.load(self.i,p); self.name.setText(self.t.name); self.rename(); self.file.setText(os.path.basename(p)); self.file.setObjectName("trackLoaded")
   except Exception as ex: QMessageBox.critical(self,"Load error",str(ex))
 def refresh(self):
  t=self.t; self.name.setText(t.name); self.detail_title.setText(f"CHANNEL {self.i+1:02d}  /  {t.name}"); self.file.setText(os.path.basename(t.path) if t.path else "NO AUDIO"); self.file.setObjectName("trackLoaded" if t.path else "trackStatus"); self.file.style().unpolish(self.file); self.file.style().polish(self.file)
  self.fader.setValue(int(200*math.log10(max(t.gain,1e-6)))); self.pan.setValue(int(t.pan*100)); self.mute.setChecked(t.mute); self.solo.setChecked(t.solo); self.phase.setChecked(t.phase)
  self.hpf.setValue(int(t.hpf)); self.gate.setChecked(t.gate_on); self.gate_thr.setValue(t.gate_thr); self.gate_ratio.setValue(t.gate_ratio); self.comp.setChecked(t.comp_on); self.comp_thr.setValue(t.comp_thr); self.comp_ratio.setValue(t.comp_ratio); self.comp_attack.setValue(t.comp_attack); self.comp_release.setValue(t.comp_release)
  for s,x in zip(self.eqsp,t.eq): s.setValue(x)
  for s,x in zip(self.eqfreq,t.eq_freq): s.setValue(x)
  for i,s in enumerate(self.sendsp): s.setValue(int(t.sends[i]*100))
  self.dca.setCurrentIndex(t.dca+1); self.mg.setCurrentIndex(t.mute_group)

class Main(QMainWindow):
 def __init__(self):
  super().__init__(); self.e=Engine(); self.setWindowTitle("Mixer32 Studio Practice — 32ch Multitrack"); self.resize(1600,1000)
  root=QWidget(); v=QVBoxLayout(root); v.setContentsMargins(10,8,10,8); v.setSpacing(8)
  # M32/X32-inspired navigation header
  header=QHBoxLayout(); header.setSpacing(5)
  logo=QLabel("M32")
  logo.setObjectName("brandLogo"); header.addWidget(logo)
  sub=QLabel("STUDIO  /  MULTITRACK MIXER")
  sub.setObjectName("brandSub"); header.addWidget(sub)
  header.addStretch(1)
  self.nav_buttons=[]
  for label,idx in [("HOME",0),("DETAIL",0),("EFFECTS",2),("SCENES",0),("METERS",0),("RECORDER",0),("ROUTING",1),("MONITOR",3),("SETUP",3)]:
   nb=QPushButton(label); nb.setObjectName("navButton"); nb.setMinimumHeight(36)
   nb.clicked.connect(lambda checked=False,j=idx:self.tabs.setCurrentIndex(j) if hasattr(self,"tabs") else None)
   header.addWidget(nb); self.nav_buttons.append(nb)
  v.addLayout(header)
  toolbar=QHBoxLayout(); toolbar.setSpacing(6)
  buttons=[("▶ PLAY",self.play),("■ STOP",self.stop),("＋ LOAD MULTITRACK",self.folder),("SAVE SCENE",self.save_scene),("LOAD SCENE",self.load_scene),("SAVE PROJECT",self.save_proj),("OPEN PROJECT",self.open_proj),("EXPORT WAV",self.export)]
  for label,fn in buttons:
   b=QPushButton(label); b.clicked.connect(fn); toolbar.addWidget(b)
  v.addLayout(toolbar)
  top=QHBoxLayout()
  self.master=QSlider(Qt.Orientation.Vertical); self.master.setRange(-600,0); self.master.setValue(-14); self.master.valueChanged.connect(lambda x:setattr(self.e,"master",10**(x/200)))
  top.addWidget(QLabel("MAIN")); top.addWidget(self.master)
  self.mastermute=QCheckBox("MAIN MUTE"); self.mastermute.toggled.connect(lambda x:setattr(self.e,"master_mute",x)); top.addWidget(self.mastermute)
  self.maincomp=QCheckBox("Main Compressor"); self.maincomp.toggled.connect(lambda x:setattr(self.e,"main_comp",x)); top.addWidget(self.maincomp)
  self.mainthr=QDoubleSpinBox(); self.mainthr.setRange(-60,0); self.mainthr.setValue(-18); self.mainthr.setSuffix(" dB Main Thresh"); self.mainthr.valueChanged.connect(lambda x:setattr(self.e,"main_comp_thr",x)); top.addWidget(self.mainthr)
  self.peak=QLabel("Peak: --"); top.addWidget(self.peak)
  self.bpm=QSpinBox(); self.bpm.setRange(40,240); self.bpm.setValue(120); self.bpm.valueChanged.connect(lambda x:setattr(self.e,"bpm",x)); top.addWidget(QLabel("BPM")); top.addWidget(self.bpm)
  self.loop=QCheckBox("Loop"); self.loop.toggled.connect(lambda x:setattr(self.e,"loop",x)); top.addWidget(self.loop)
  v.addLayout(top)
  self.tabs=QComboBox(); self.tabs.addItems(["Channels 1–32","Bus Masters","FX Rack","DCA / Mute Groups"]); self.tabs.currentIndexChanged.connect(self.switch_view); v.addWidget(self.tabs)
  self.scroll=QScrollArea(); self.scroll.setWidgetResizable(True); v.addWidget(self.scroll,1)
  self.channel_widget=QWidget(); grid=QGridLayout(self.channel_widget); grid.setSpacing(10); grid.setContentsMargins(10,10,10,10); self.strips=[]
  for i in range(32):
   s=Strip(self.e,i,self); self.strips.append(s); grid.addWidget(s,i//8,i%8)
  self.scroll.setWidget(self.channel_widget)
  self.bus_widget=self.make_buses(); self.fx_widget=self.make_fx(); self.group_widget=self.make_groups()
  self.setCentralWidget(root)
  self.setStyleSheet("""
  QMainWindow, QWidget { background-color: #1b1d21; color: #e8eaf0; font-family: 'Segoe UI'; font-size: 10pt; }
  QStatusBar { background: #111317; color: #aeb5c0; border-top: 1px solid #343941; }
  QLabel#brandLogo { color: #ffffff; background: #111317; border: 1px solid #4d5662; border-radius: 7px; font-size: 25pt; font-weight: 900; padding: 1px 13px; }
  QLabel#brandSub { color: #9ca6b4; font-size: 9pt; font-weight: 700; padding-left: 8px; }
  QLabel#channelNumber { background: #303640; color: #a8d8ff; border-radius: 4px; padding: 5px; font-weight: 900; letter-spacing: 1px; }
  QLabel#trackStatus { color: #8e98a6; background: #20242a; border: 1px solid #343a43; border-radius: 4px; padding: 5px; font-size: 8pt; }
  QLabel#trackLoaded { color: #a8f0a0; background: #1c3025; border: 1px solid #39724b; border-radius: 4px; padding: 5px; font-size: 8pt; }
  QLabel#faderValue { color: #f0c66b; font-weight: 800; min-width: 42px; }
  QLabel#detailTitle { color: #ffffff; font-size: 18pt; font-weight: 900; padding: 8px; }
  QLabel#sectionHint { color: #9fa9b7; padding: 8px; }
  QPushButton { background: #303640; border: 1px solid #4a5360; border-radius: 5px; padding: 7px 10px; color: #edf1f6; font-weight: 700; }
  QPushButton:hover { border: 1px solid #7fc7ff; background: #394553; }
  QPushButton:pressed { background: #171a1f; }
  QPushButton#navButton { min-width: 68px; background: #262b32; border: 1px solid #3b424c; border-radius: 4px; color: #c8d0db; font-size: 9pt; font-weight: 800; }
  QPushButton#navButton:hover { background: #36414c; color: white; border-bottom: 3px solid #5eb9ff; }
  QPushButton#detailButton { background: #21445a; border: 1px solid #34759a; color: #dff4ff; }
  QPushButton#muteButton:checked { background: #9c3038; border-color: #e95d67; color: white; }
  QPushButton#soloButton:checked { background: #b58a25; border-color: #e5c15b; color: #171717; }
  QGroupBox { background: #242930; border: 1px solid #414955; border-radius: 7px; margin-top: 12px; padding: 8px 5px 5px 5px; font-weight: 700; color: #e2e8f0; }
  QGroupBox#channelStrip { background: #242930; border: 1px solid #3c4652; border-radius: 8px; }
  QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 5px; color: #d9e1eb; }
  QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #171a1f; color: #f2f5f9; border: 1px solid #424b57; border-radius: 4px; padding: 5px; selection-background-color: #286c96; }
  QSlider::groove:vertical { background: #101216; border: 1px solid #454d58; width: 10px; border-radius: 4px; }
  QSlider::handle:vertical { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #9ba4af, stop:0.5 #f4f6f8, stop:1 #77818d); border: 1px solid #111317; height: 20px; margin: 0 -7px; border-radius: 3px; }
  QSlider::groove:horizontal { background: #101216; border: 1px solid #454d58; height: 8px; border-radius: 4px; }
  QSlider::handle:horizontal { background: #d4dce5; border: 1px solid #111317; width: 14px; margin: -5px 0; border-radius: 3px; }
  QCheckBox { spacing: 5px; color: #dce3ec; }
  QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #697483; background: #171a1f; border-radius: 3px; }
  QCheckBox::indicator:checked { background: #58b978; border: 1px solid #8ce7a5; }
  QScrollArea { border: 1px solid #363e49; border-radius: 7px; background: #1b1d21; }
  QTabWidget::pane { border: 1px solid #424b57; border-radius: 6px; top: -1px; background: #20242a; }
  QTabBar::tab { background: #2b3139; color: #b8c2cf; padding: 10px 24px; margin-right: 4px; border-top-left-radius: 5px; border-top-right-radius: 5px; font-weight: 800; }
  QTabBar::tab:selected { background: #245574; color: #ffffff; border-bottom: 3px solid #63c1ff; }
  """)
  self.timer=QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(250)
  self.statusBar().showMessage("READY  •  Load multitrack files, mix, and save a project/scene.")
 def make_buses(self):
  w=QWidget(); grid=QGridLayout(w); self.bus_controls=[]
  for i,b in enumerate(self.e.buses):
   box=QGroupBox(f"Bus {i+1}"); l=QVBoxLayout(box); name=QLineEdit(b.name); name.textChanged.connect(lambda x,j=i:setattr(self.e.buses[j],"name",x)); l.addWidget(name)
   s=QSlider(Qt.Orientation.Vertical); s.setRange(-600,0); s.setValue(-20); s.valueChanged.connect(lambda x,j=i:setattr(self.e.buses[j],"gain",10**(x/200))); l.addWidget(s,alignment=Qt.AlignmentFlag.AlignHCenter)
   m=QCheckBox("Mute"); m.toggled.connect(lambda x,j=i:setattr(self.e.buses[j],"mute",x)); l.addWidget(m)
   c=QCheckBox("Compressor"); c.toggled.connect(lambda x,j=i:setattr(self.e.buses[j],"comp_on",x)); l.addWidget(c)
   thr=QDoubleSpinBox(); thr.setRange(-60,0); thr.setValue(-18); thr.valueChanged.connect(lambda x,j=i:setattr(self.e.buses[j],"comp_thr",x)); l.addWidget(thr)
   grid.addWidget(box,i//8,i%8); self.bus_controls.append((name,s,m,c,thr))
  return w
 def make_fx(self):
  w=QWidget(); grid=QGridLayout(w); self.fx_controls=[]
  for i,fx in enumerate(self.e.fx):
   box=QGroupBox(f"FX {i+1}: {fx.name}"); l=QVBoxLayout(box)
   en=QCheckBox("Enabled"); en.setChecked(True); en.toggled.connect(lambda x,j=i:setattr(self.e.fx[j],"enabled",x)); l.addWidget(en)
   mix=QSlider(Qt.Orientation.Horizontal); mix.setRange(0,100); mix.setValue(20); mix.valueChanged.connect(lambda x,j=i:setattr(self.e.fx[j],"mix",x/100)); l.addWidget(QLabel("Wet Mix")); l.addWidget(mix)
   tm=QDoubleSpinBox(); tm.setRange(.01,1.8); tm.setSingleStep(.05); tm.setValue(.35); tm.valueChanged.connect(lambda x,j=i:setattr(self.e.fx[j],"time",x)); l.addWidget(QLabel("Time / Decay")); l.addWidget(tm)
   fb=QDoubleSpinBox(); fb.setRange(0,.88); fb.setSingleStep(.05); fb.setValue(.3); fb.valueChanged.connect(lambda x,j=i:setattr(self.e.fx[j],"feedback",x)); l.addWidget(QLabel("Feedback")); l.addWidget(fb)
   grid.addWidget(box,i//4,i%4); self.fx_controls.append((en,mix,tm,fb))
  return w
 def make_groups(self):
  w=QWidget(); grid=QGridLayout(w)
  for i in range(8):
   box=QGroupBox(f"DCA {i+1} / Mute Group {i+1}"); l=QVBoxLayout(box)
   s=QSlider(Qt.Orientation.Horizontal); s.setRange(-600,100); s.setValue(0); s.valueChanged.connect(lambda x,j=i:setattr(self.e.dca_gains,j,10**(x/200))); l.addWidget(QLabel("DCA Level")); l.addWidget(s)
   m=QCheckBox("DCA Mute"); m.toggled.connect(lambda x,j=i:setattr(self.e.dca_mutes,j,x)); l.addWidget(m)
   mg=QCheckBox("Mute Group Active"); mg.toggled.connect(lambda x,j=i:self.set_mute_group(j,x)); l.addWidget(mg)
   grid.addWidget(box,i//4,i%4)
  return w
 def set_mute_group(self,i,x): self.e.mute_groups[i]=x
 def open_channel_detail(self,i):
  self.strips[i].refresh(); self.scroll.takeWidget(); self.scroll.setWidget(self.strips[i].detail_widget); self.strips[i].detail_tabs.setCurrentIndex(0); self.statusBar().showMessage(f"CHANNEL {i+1:02d}  •  Gate / Compressor / EQ / Sends")
 def switch_view(self,i):
  self.scroll.takeWidget(); self.scroll.setWidget([self.channel_widget,self.bus_widget,self.fx_widget,self.group_widget][i])
  if i==0: self.statusBar().showMessage("CHANNEL OVERVIEW  •  Select OPEN CHANNEL DETAIL to edit Gate, Comp, EQ and Sends")
 def play(self):
  try: self.e.start(); self.statusBar().showMessage("Playback running")
  except Exception as ex: QMessageBox.critical(self,"Audio device error",str(ex))
 def stop(self): self.e.stop(); self.statusBar().showMessage("Stopped")
 def tick(self): self.peak.setText(f"Peak: {self.e.last_peak:.3f}"+("  CLIP" if self.e.last_peak>.97 else ""))
 def folder(self):
  d=QFileDialog.getExistingDirectory(self,"Select multitrack folder")
  if not d:return
  files=sorted(os.path.join(d,x) for x in os.listdir(d) if x.lower().endswith(EXTS))
  if not files: QMessageBox.information(self,"No tracks","No supported audio files found."); return
  try:
   for i,p in enumerate(files[:32]):
    self.e.load(i,p); self.strips[i].refresh()
   QMessageBox.information(self,"Loaded",f"Loaded {min(32,len(files))} tracks.")
  except Exception as ex: QMessageBox.critical(self,"Load error",str(ex))
 def state(self):
  ts=[]
  for t in self.e.tracks:
   ts.append({k:getattr(t,k) for k in ("n","name","path","gain","pan","mute","solo","phase","hpf","gate_on","gate_thr","gate_ratio","comp_on","comp_thr","comp_ratio","comp_attack","comp_release","eq","eq_freq","eq_q","sends","send_pre","dca","mute_group")})
  bs=[{k:getattr(b,k) for k in ("name","gain","pan","mute","comp_on","comp_thr","comp_ratio","eq")} for b in self.e.buses]
  fs=[{k:getattr(f,k) for k in ("name","kind","mix","time","feedback","damping","enabled")} for f in self.e.fx]
  return {"format":"Mixer32StudioPractice","version":2,"sample_rate":SR,"master":self.e.master,"master_mute":self.e.master_mute,"main_comp":self.e.main_comp,"main_comp_thr":self.e.main_comp_thr,"main_comp_ratio":self.e.main_comp_ratio,"tracks":ts,"buses":bs,"fx":fs,"dca_gains":self.e.dca_gains,"dca_mutes":self.e.dca_mutes,"mute_groups":self.e.mute_groups,"bpm":self.e.bpm,"loop":self.e.loop}
 def apply(self,d):
  self.e.master=d.get("master",.85); self.e.master_mute=d.get("master_mute",False); self.e.main_comp=d.get("main_comp",False); self.e.main_comp_thr=d.get("main_comp_thr",-18); self.e.main_comp_ratio=d.get("main_comp_ratio",2)
  for i,x in enumerate(d.get("tracks",[])[:32]):
   t=self.e.tracks[i]
   for k,v in x.items():
    if hasattr(t,k) and k not in ("audio","pos","filters","eq_filters","env","gate_env"): setattr(t,k,v)
   if t.path and os.path.isfile(t.path):
    try:self.e.load(i,t.path); 
    except:pass
   t.eq_filters={}; t.filters={}
   self.strips[i].refresh()
  for i,x in enumerate(d.get("buses",[])[:16]):
   b=self.e.buses[i]
   for k,v in x.items():
    if hasattr(b,k):setattr(b,k,v)
   if i<len(self.bus_controls):
    name,s,m,c,thr=self.bus_controls[i]; name.setText(b.name); s.setValue(int(200*math.log10(max(b.gain,1e-6)))); m.setChecked(b.mute); c.setChecked(b.comp_on); thr.setValue(b.comp_thr)
  for i,x in enumerate(d.get("fx",[])[:8]):
   f=self.e.fx[i]
   for k,v in x.items():
    if hasattr(f,k):setattr(f,k,v)
   en,mix,tm,fb=self.fx_controls[i]; en.setChecked(f.enabled); mix.setValue(int(f.mix*100)); tm.setValue(f.time); fb.setValue(f.feedback)
  self.e.dca_gains=d.get("dca_gains",[1.]*8); self.e.dca_mutes=d.get("dca_mutes",[False]*8); self.e.mute_groups=d.get("mute_groups",[False]*8)
  self.e.bpm=d.get("bpm",120); self.bpm.setValue(self.e.bpm); self.e.loop=d.get("loop",False); self.loop.setChecked(self.e.loop)
  self.master.setValue(int(200*math.log10(max(self.e.master,1e-6)))); self.mastermute.setChecked(self.e.master_mute); self.maincomp.setChecked(self.e.main_comp); self.mainthr.setValue(self.e.main_comp_thr)
 def save_json(self,p):
  with open(p,"w",encoding="utf-8") as f:json.dump(self.state(),f,ensure_ascii=False,indent=2)
 def save_scene(self):
  p,_=QFileDialog.getSaveFileName(self,"Save Scene","Scene01.json","Scene (*.json)")
  if p:self.save_json(p); self.statusBar().showMessage("Scene saved")
 def load_scene(self):
  p,_=QFileDialog.getOpenFileName(self,"Load Scene","","Scene (*.json)")
  if p:
   try:
    with open(p,encoding="utf-8") as f:self.apply(json.load(f))
    self.statusBar().showMessage("Scene loaded")
   except Exception as ex:QMessageBox.critical(self,"Scene error",str(ex))
 def save_proj(self):
  p,_=QFileDialog.getSaveFileName(self,"Save Project","PracticeShow.m32p","Mixer Project (*.m32p *.json)")
  if p:self.save_json(p); self.statusBar().showMessage("Project saved. Keep source files available at their saved paths.")
 def open_proj(self):
  p,_=QFileDialog.getOpenFileName(self,"Open Project","","Mixer Project (*.m32p *.json)")
  if p:
   try:
    with open(p,encoding="utf-8") as f:self.apply(json.load(f))
    self.statusBar().showMessage("Project loaded")
   except Exception as ex:QMessageBox.critical(self,"Project error",str(ex))
 def export(self):
  p,_=QFileDialog.getSaveFileName(self,"Export Mix","PracticeMix.wav","WAV (*.wav)")
  if not p:return
  ts=[t for t in self.e.tracks if t.audio is not None]
  if not ts:QMessageBox.information(self,"No tracks","Load tracks first.");return
  n=max(len(t.audio) for t in ts); out=np.zeros((n,2),np.float32)
  solo=any(t.solo and t.audio is not None for t in self.e.tracks)
  for t in ts:
   if t.mute or (solo and not t.solo) or (t.mute_group and self.e.mute_groups[t.mute_group-1]) or self.e.dca_mutes[t.dca]:continue
   a=t.audio.copy()
   if t.phase:a*=-1
   if t.hpf>20:
    try:a=sosfilt(butter(2,t.hpf,btype="highpass",fs=SR,output="sos"),a,axis=0)
    except:pass
   for j in range(4):
    if abs(t.eq[j])>.01:
     try:
      f=np.clip(t.eq_freq[j],20,SR*.45); A=10**(t.eq[j]/40); w=2*np.pi*f/SR; alpha=np.sin(w)/(2*t.eq_q[j]); c=np.cos(w)
      sos=np.array([[(1+alpha*A)/(1+alpha/A),(-2*c)/(1+alpha/A),(1-alpha*A)/(1+alpha/A),1,(-2*c)/(1+alpha/A),(1-alpha/A)/(1+alpha/A)]])
      a=sosfilt(sos,a,axis=0)
     except:pass
   a*=t.gain*self.e.dca_gains[t.dca]
   if t.comp_on:
    pk=np.max(np.abs(a),axis=1,keepdims=True)+1e-9; db=20*np.log10(pk)
    gain=np.where(db>t.comp_thr,10**((t.comp_thr+(db-t.comp_thr)/max(1,t.comp_ratio)-db)/20),1)
    a*=gain
   if t.gate_on:
    rms=np.sqrt(np.mean(a*a,axis=1,keepdims=True)+1e-12); db=20*np.log10(rms)
    a*=np.where(db<t.gate_thr,.08,1)
   a[:,0]*=math.cos((t.pan+1)*math.pi/4); a[:,1]*=math.sin((t.pan+1)*math.pi/4)
   out[:len(a)]+=a
  out*=self.e.master
  pk=np.max(np.abs(out))+1e-9
  if pk>.98:out*=.98/pk
  try:sf.write(p,out,SR,subtype="PCM_24");QMessageBox.information(self,"Exported",f"Saved:\n{p}")
  except Exception as ex:QMessageBox.critical(self,"Export error",str(ex))
 def closeEvent(self,event):self.e.close();event.accept()

def main():
 app=QApplication(sys.argv); app.setStyle("Fusion"); w=Main();w.show();sys.exit(app.exec())
if __name__=="__main__":main()
