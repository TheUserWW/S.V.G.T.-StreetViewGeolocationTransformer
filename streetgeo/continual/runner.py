"""Foreground VM session: native play, then CPU training between untimed rounds."""
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from dataclasses import asdict

from .config import PROJECT,atomic_json
from .storage import RunLock,Store,now
from .resources import budget,lower_priority
from .learning import prepare_snapshot,prepare_recent_batch

class StopRequested(Exception): pass

def controls(root):
    try: return json.loads((root/'control.json').read_text(encoding='utf-8'))
    except (OSError,ValueError): return {}
def set_controls(root,**changes): atomic_json(root/'control.json',{**controls(root),**changes})

class TrainingRunner:
    store_class=Store
    worker_module=None

    def __init__(self,config,*,require_model=True):
        self.config=config.validate(require_model); self.root=config.runtime; self.store=self.store_class(self.root)
        self.active=''; self.child=None
        if require_model:
            self.active=str(config.model_path)
            stamp=config.model_path.stat(); selected=[self.active,stamp.st_mtime_ns,stamp.st_size]
            if self.store.get('selected_base')!=selected:
                self.store.set('selected_base',selected); self.store.set('active_checkpoint',self.active); self.store.set('pending_train',None)
            self.active=self.store.get('active_checkpoint',self.active)
            if not Path(self.active).is_file(): raise ValueError('Active model was moved; reselect a model')
        policy=[config.learning.validation_fraction,config.learning.spatial_block_degrees]
        if self.store.get('split_policy',policy)!=policy: raise ValueError('Existing sample split policy cannot change')
        self.store.set('split_policy',policy)
        self.log=logging.getLogger(f'svgt.vm.{id(self)}'); self.log.setLevel(logging.INFO)
        handler=RotatingFileHandler(self.root/'trainer.log',maxBytes=2*1024**2,backupCount=2,encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s')); self.log.addHandler(handler)
        self.last=None; self.resume_focus=lambda:None
    def status(self,state,detail):
        counts=self.store.counts(); cfg=self.config.learning; cursor=self.store.get('last_training_count',0)
        atomic_json(self.root/'status.json',{'state':state,'detail':detail,'pid':os.getpid(),'updated_at':now(),
                    'counts':counts,'active_checkpoint':self.active,
                    'learning':{'enabled':cfg.enabled,'mode':cfg.sample_mode,'every':cfg.train_every,'processed':cursor,
                        'queued':max(0,counts['completed']-cursor),'note':self.store.get('learning_note','')}})
        if self.last!=(state,detail): self.log.info('%s: %s',state,detail); print(detail,flush=True); self.last=(state,detail)
    def check(self):
        paused=False
        while True:
            c=controls(self.root)
            if c.get('stop'): raise StopRequested()
            if not c.get('paused'):
                if paused: self.resume_focus()
                return
            paused=True
            self.status('paused','已暂停 / Paused'); time.sleep(.2)
    def gate(self):
        while True:
            self.check(); reason=budget(self.config)
            if reason is None: return
            self.status('waiting',reason); time.sleep(1)
    def sleep(self,seconds):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline: self.check(); time.sleep(min(.2,max(0,deadline-time.monotonic())))
    def job(self,request):
        if self.worker_module is None: raise NotImplementedError('Select a source-specific model worker')
        self.gate(); identity=uuid.uuid4().hex; folder=self.root/'jobs'; folder.mkdir(exist_ok=True)
        cancel=folder/(identity+'.cancel'); response=folder/(identity+'.result.json'); path=folder/(identity+'.request.json')
        request={**request,'cpu_threads':self.config.cpu_threads,'device':self.config.device,'cancel':str(cancel),'response':str(response)}
        atomic_json(path,request); cancelled=None
        with (folder/(identity+'.log')).open('w',encoding='utf-8') as output:
            self.child=subprocess.Popen([sys.executable,'-m',self.worker_module,'worker',str(path)],cwd=PROJECT,
                env={**os.environ,'PYTHONUTF8':'1','OMP_NUM_THREADS':str(self.config.cpu_threads),'MKL_NUM_THREADS':str(self.config.cpu_threads)},
                stdout=output,stderr=output)
            lower_priority(self.child.pid)
            try:
                while self.child.poll() is None:
                    c=controls(self.root)
                    if c.get('stop') or c.get('paused') or budget(self.config):
                        cancel.touch(); cancelled=cancelled or time.monotonic()
                        if time.monotonic()-cancelled>15: self.child.terminate(); break
                    time.sleep(.2)
            finally:
                if self.child.poll() is None:
                    cancel.touch()
                    try: self.child.wait(timeout=15)
                    except subprocess.TimeoutExpired: self.child.kill()
                self.child.wait(); self.child=None
        self.check()
        if not response.exists():
            if cancelled is not None: return {'status':'paused'}
            raise RuntimeError(f'Model worker failed: {folder/(identity+".log")}')
        result=json.loads(response.read_text(encoding='utf-8'))
        if result['status']=='error': raise RuntimeError(result['error'])
        return result
    def predict(self,row):
        while True:
            self.status('predicting','模型预测 / Predicting')
            result=self.job({'kind':'predict','checkpoint':self.active,'images':row['images'],'headings':row['headings']})
            if result['status']!='paused': return result
    def train(self):
        cfg=self.config.learning
        if not cfg.enabled:
            self.store.set('learning_note','未启用自动训练 / Automatic learning disabled')
            self.status('learning_disabled','未启用自动训练 / Automatic learning disabled'); return
        while True:
            self.check()
            count=self.store.counts()['completed']; pending=self.store.get('pending_train')
            if not pending:
                batch=None
                if cfg.sample_mode=='recent':
                    batch=prepare_recent_batch(self.store,cfg)
                    if batch['state']=='waiting':
                        self.store.set('learning_note',batch['detail']); self.status('learning_wait',batch['detail']); return
                    if batch['state']=='skipped':
                        self.store.finish_training(uuid.uuid4().hex,{'status':'skipped',**batch},batch['end'])
                        self.store.set('learning_note',batch['detail']); self.status('learning_skipped',batch['detail']); continue
                    count=batch['end']; snapshot={k:batch[k] for k in ('train','validation')}
                else:
                    if count-self.store.get('last_training_count',0)<cfg.train_every: return
                    snapshot=prepare_snapshot(self.store,cfg)
                    if snapshot is None:
                        self.status('learning_wait','等待历史训练/验证样本 / Waiting for replay samples'); return
                identity=uuid.uuid4().hex
                pending={'id':identity,'count':count,'kind':'train','checkpoint':self.active,'learning':asdict(cfg),
                    'batch':{k:batch[k] for k in ('start','end','round_ids')} if batch else None,
                    'candidate':str(self.root/'models'/f'personal_{identity}.pt'),'resume':str(self.root/'models'/f'training_{identity}.resume.pt'),**snapshot}
                self.store.set('pending_train',pending)
            batch=pending.get('batch')
            detail=(f"学习第 {batch['start']}–{batch['end']} 轮：训练 {len(pending['train'])}，独立验证 {len(pending['validation'])} / Training batch"
                    if batch else '回合间训练 / Training between rounds')
            self.store.set('learning_note',detail); self.status('training',detail)
            report=self.job(pending)
            if report['status']=='paused':
                self.store.update(pending['id'],report); self.status('training_paused','训练断点已保存 / Training checkpoint saved'); return
            report={**report,'batch':batch}
            self.store.finish_training(pending['id'],report,pending['count'])
            self.active=self.store.get('active_checkpoint',self.active)
            detail='候选模型通过验证 / Model accepted' if report['status']=='accepted' else '候选未通过验证，保留当前模型 / Candidate rejected; keep current model'
            self.store.set('learning_note',detail); self.status(report['status'],detail)
    def close(self):
        self.store.close()
        for h in list(self.log.handlers): h.close(); self.log.removeHandler(h)
