"""BAI301 Option B: self-contained reproducible RNN/ESN forecasting experiment engine.
Run with python experiment_engine.py --help. This is emitted inside the Kaggle notebook.
"""
from __future__ import annotations
import argparse, contextlib, csv, hashlib, io, itertools, json, math, os, random, re, resource, sys, time, traceback, warnings, zipfile
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from urllib.request import urlopen, Request

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigs
from scipy.linalg import solve as scipy_solve
import torch
from torch import nn
import psutil

SEEDS = [42, 123, 2026]
DATA_URL = 'https://zenodo.org/records/4656140/files/electricity_hourly_dataset.zip?download=1'
EXPECTED_MD5 = '18096614662b02640d265ad2a6a416bd'


def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with open(tmp, 'w', encoding='utf-8') as f: json.dump(obj, f, indent=2, default=str, allow_nan=False)
    os.replace(tmp, path)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def file_md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda:f.read(1<<20), b''): h.update(block)
    return h.hexdigest()


def seed_all(seed, device='cpu'):
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)


def standard_config(preset='smoke', out_root='/kaggle/working/bai301_results'):
    cfg = dict(
        PRESET=preset, OUTPUT_ROOT=str(out_root), CONTEXT=100, HORIZON=100,
        DATASETS=['mg','electricity'] if preset=='full' else ['mg'],
        EPOCHS=30 if preset in ('full','minimum') else 2,
        BATCH_SIZE=64 if preset in ('full','minimum') else 8,
        RNN_HIDDEN=64 if preset in ('full','minimum') else 16,
        TRAIN_STRIDE=25, EVAL_STRIDE=100,
        MAX_TRAIN_ORIGINS=512 if preset in ('full','minimum') else 16,
        MAX_EVAL_ORIGINS=48 if preset in ('full','minimum') else 4,
        VALIDATE_EVERY=10 if preset in ('full','minimum') else 1,
        MG_N=50000 if preset in ('full','minimum') else 2500,
        MG_TRANSIENT=2000 if preset in ('full','minimum') else 500,
        MG_DT=0.1, MG_DELAY=17.0,
        LR_CANDIDATES=[0.0001,0.0003,0.001,0.003,0.01],
        TUNE_EPOCHS=3, TUNE_TRAIN_ORIGINS=128, TUNE_EVAL_ORIGINS=12,
        ESN_FIT_STATES=6000 if preset=='full' else 800,
        ESN_TUNE_EVAL_ORIGINS=12, ESN_WASHOUT=100,
        ESN_DENSITY=0.01, ESN_INPUT_SCALE=0.5,
        ESN_CANDIDATES=[
          {'rho':0.7,'leak':0.3,'ridge':1e-4},
          {'rho':0.9,'leak':0.5,'ridge':1e-4},
          {'rho':0.9,'leak':0.8,'ridge':1e-3},
          {'rho':1.05,'leak':0.3,'ridge':1e-3},
          {'rho':0.7,'leak':0.8,'ridge':1e-2}],
        ELECTRICITY_ZIP=None,
        RUN_TUNING=preset in ('full','minimum'),
    )
    return cfg


def run_directory(cfg):
    signature = {k:v for k,v in cfg.items() if k not in ('OUTPUT_ROOT','ELECTRICITY_ZIP')}
    return Path(cfg['OUTPUT_ROOT']) / ('exp_' + digest(signature))


def generate_mackey_glass(n=50000, transient=2000, dt=0.1, tau=17., beta=.2, gamma=.1, power=10, x0=1.2):
    """Forward Euler integration of x' = beta*x(t-tau)/(1+x(t-tau)^n) - gamma*x(t).
    Output sampled every 1 time unit. This is a documented numeric approximation, NOT ReservoirPy.
    """
    stride=int(round(1/dt)); delay=int(round(tau/dt))
    assert abs(stride*dt-1)<1e-8 and abs(delay*dt-tau)<1e-8
    total=(n+transient)*stride
    states=np.full(total+delay+1, x0, dtype=np.float64)
    for i in range(delay, delay+total):
        old=states[i-delay]; now=states[i]
        states[i+1]=now+dt*(beta*old/(1+old**power)-gamma*now)
    return states[delay+1+transient*stride:delay+1+total:stride].astype('float32')[:n]


def download_electricity_zip(dest, manual=None):
    dest=Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if manual:
        manual=Path(manual)
        if manual.is_file(): return manual
    if dest.exists() and zipfile.is_zipfile(dest): return dest
    # Kaggle Add Input -> inspect user-attached datasets first (internet may be disabled).
    for root in ('/kaggle/input',):
        if Path(root).exists():
            for p in Path(root).rglob('*electricity*hourly*.zip'):
                if zipfile.is_zipfile(p): return p
            for p in Path(root).rglob('*.tsf'):
                if 'electricity' in str(p).lower(): return p
    req=Request(DATA_URL, headers={'User-Agent':'BAI301 research experiment (student educational project)'})
    try:
        with urlopen(req,timeout=60) as resp, open(dest,'wb') as f:
            while True:
                chunk=resp.read(1024*1024)
                if not chunk: break
                f.write(chunk)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError('Electricity download failed. In Kaggle turn Internet ON, or upload '
                           'electricity_hourly_dataset.zip using Add Input and rerun. '
                           f'Original error: {exc}') from exc
    if not zipfile.is_zipfile(dest):
        dest.unlink(missing_ok=True); raise RuntimeError('The electricity file was not a valid ZIP.')
    actual=file_md5(dest)
    if actual != EXPECTED_MD5:
        warnings.warn(f'Electricity archive MD5 differs from Zenodo version 3: {actual}; inspect source.')
    return dest


def extract_tsf_text(path):
    path=Path(path)
    if path.suffix.lower()=='.tsf': return path.read_text(encoding='utf-8-sig')
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.lower().endswith('.tsf'):
                return z.read(name).decode('utf-8-sig')
        for name in z.namelist():
            if name.lower().endswith('.zip'):
                with zipfile.ZipFile(io.BytesIO(z.read(name))) as sub:
                    for name2 in sub.namelist():
                        if name2.lower().endswith('.tsf'):
                            return sub.read(name2).decode('utf-8-sig')
    raise RuntimeError('Archive does not contain a .tsf file. Inspect its contents and source.')


def parse_tsf_records(text):
    records=[]; data_started=False
    for line in text.splitlines():
        line=line.strip()
        if not line or line.startswith('#'): continue
        if line.lower()=='@data': data_started=True; continue
        if not data_started or line.startswith('@'): continue
        # TSF format: metadata items separated by ':' then last item is comma-separated numbers.
        pieces=line.split(':')
        if len(pieces)<2: continue
        numbers=pieces[-1]
        vals=np.array([np.nan if x.strip() in ('?','') else float(x) for x in numbers.split(',')],dtype=np.float32)
        meta=pieces[:-1]
        stamp=None
        for item in meta:
            try:
                stamp=datetime.strptime(item.strip(),'%Y-%m-%d %H-%M-%S')
                break
            except ValueError: pass
        records.append((stamp, vals, meta[0] if meta else 'unknown'))
    if not records: raise RuntimeError('No time series found in the TSF file.')
    return records


def aggregate_electricity(records):
    # Align by start time if available; do not assume all records begin simultaneously.
    starts=[r[0] for r in records]
    if all(s is not None for s in starts):
        start=min(starts)
        offsets=[int(round((s-start).total_seconds()/3600)) for s in starts]
        length=max(o+len(r[1]) for o,r in zip(offsets,records))
    else:
        if len({len(r[1]) for r in records}) != 1:
            raise ValueError('Cannot align unequal-length electricity series without timestamps.')
        if len({str(s) for s in starts}) != 1:
            raise ValueError('Some electricity start times are missing; cannot safely aggregate.')
        start=None; offsets=[0]*len(records); length=len(records[0][1])
    agg=np.zeros(length, dtype=np.float64)
    coverage=np.zeros(length,dtype=np.int32)
    for (stamp, vals, name),o in zip(records,offsets):
        # Causal missing-value treatment: forward fill once the series has begun,
        # and leading missing values become zero (documented, never backfill from future).
        cleaned=pd.Series(vals).ffill().fillna(0).to_numpy(dtype=np.float64)
        agg[o:o+len(vals)] += cleaned
        coverage[o:o+len(vals)] += 1
    if np.min(coverage)==0: raise RuntimeError('Aggregate series contains hours with no observed clients.')
    meta=dict(n_clients=len(records),hours=length,start_timestamp=str(start),
              client_coverage_min=int(coverage.min()),client_coverage_max=int(coverage.max()),
              client_coverage_changes=int(np.sum(np.diff(coverage)!=0)),
              missing_policy='Within client forward-fill, leading missing->0; not backfill')
    return agg.astype(np.float32),meta


def prepare_data(cfg):
    root=run_directory(cfg); data=root/'data'; data.mkdir(parents=True,exist_ok=True)
    meta_path=data/'metadata.json'
    metadata=json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for dataset in cfg['DATASETS']:
        output=data/(dataset+'.npy')
        if output.exists():
            values=np.load(output)
            metadata[dataset].update(n=len(values),sha256=hashlib.sha256(output.read_bytes()).hexdigest()) if dataset in metadata else metadata.update({dataset:dict(n=len(values),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),cached=True)})
            continue
        if dataset=='mg':
            values=generate_mackey_glass(cfg['MG_N'],cfg['MG_TRANSIENT'],cfg['MG_DT'],cfg['MG_DELAY'])
            source_meta=dict(source='Mackey & Glass equation, documented forward Euler method',
                             dt=cfg['MG_DT'],delay=cfg['MG_DELAY'],beta=.2,gamma=.1,exponent=10,
                             transient_time_units=cfg['MG_TRANSIENT'])
        elif dataset=='electricity':
            archive=download_electricity_zip(data/'electricity_hourly_dataset.zip',cfg.get('ELECTRICITY_ZIP'))
            records=parse_tsf_records(extract_tsf_text(archive))
            values,source_meta=aggregate_electricity(records)
            source_meta.update(source='Zenodo 10.5281/zenodo.4656140, electricity hourly v3',
                               source_archive=str(archive),source_md5=file_md5(archive))
        else: raise ValueError(dataset)
        if not np.isfinite(values).all() or np.std(values)<1e-7:
            raise RuntimeError(f'{dataset}: invalid, missing, or nearly constant generated data')
        np.save(output,values)
        metadata[dataset]=dict(n=len(values),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),**source_meta)
    save_json(meta_path,metadata)
    return metadata


class SeriesWindows:
    def __init__(self, values,cfg,split, max_origins=None,eval_mode=False):
        self.values=np.asarray(values, dtype=np.float32)
        n=len(self.values); boundaries=[0,int(n*.7),int(n*.85),n]
        a,b={'train':(boundaries[0],boundaries[1]),'val':(boundaries[1],boundaries[2]),
             'test':(boundaries[2],boundaries[3])}[split]
        self.a=a; self.b=b
        self.mean=float(np.mean(self.values[:boundaries[1]],dtype=np.float64))
        self.std=float(np.std(self.values[:boundaries[1]],dtype=np.float64))
        if self.std<=1e-9: raise ValueError('Near-zero training standard deviation')
        self.scaled=(self.values-self.mean)/self.std
        C,H=cfg['CONTEXT'],cfg['HORIZON']; stride=cfg['EVAL_STRIDE'] if eval_mode else cfg['TRAIN_STRIDE']
        # Context and target both inside the named chronological split. No split crossing.
        self.origins=np.arange(a,b-(C+H)+1,max(1,stride),dtype=np.int64)
        if len(self.origins)==0: raise RuntimeError(f'{split}: insufficient observations for C={C}, H={H}')
        cap=max_origins
        if cap is not None and len(self.origins)>cap:
            idx=np.linspace(0,len(self.origins)-1,cap,dtype=np.int64)
            self.origins=self.origins[idx]
        self.C=C;self.H=H

    def batches(self,batch_size,seed=42,shuffle=False):
        origins=self.origins.copy()
        if shuffle: np.random.default_rng(seed).shuffle(origins)
        offsets=np.arange(self.C+self.H,dtype=np.int64)
        for i in range(0,len(origins),batch_size):
            indices=origins[i:i+batch_size,None]+offsets[None,:]
            block=torch.from_numpy(np.ascontiguousarray(self.scaled[indices]))
            yield block[:,:self.C].unsqueeze(-1), block[:,self.C:].unsqueeze(-1)


def windows_for(cfg,dataset):
    series=np.load(run_directory(cfg)/'data'/(dataset+'.npy'))
    tr=SeriesWindows(series,cfg,'train',cfg['MAX_TRAIN_ORIGINS'])
    va=SeriesWindows(series,cfg,'val',cfg['MAX_EVAL_ORIGINS'],eval_mode=True)
    te=SeriesWindows(series,cfg,'test',cfg['MAX_EVAL_ORIGINS'],eval_mode=True)
    return tr,va,te


class ForecastNet(nn.Module):
    def __init__(self,architecture,hidden_size):
        super().__init__(); layer={'RNN':nn.RNN,'LSTM':nn.LSTM,'GRU':nn.GRU}[architecture]
        self.architecture=architecture
        self.recurrent=layer(1,hidden_size,num_layers=1,batch_first=True)
        self.output_layer=nn.Linear(hidden_size,1)
        self.hidden_size=hidden_size
    def forward_inputs(self,inputs,state=None):
        hidden_seq, state=self.recurrent(inputs,state)
        return self.output_layer(hidden_seq[:,-1,:]),state
    def forward(self,context,targets=None,forcing='never',scheduled_prob=0.,rng=None):
        pred,state=self.forward_inputs(context)
        if targets is None:return pred,state
        predictions=[pred]; prev=pred
        for t in range(1,targets.shape[1]):
            if forcing=='always': feed=targets[:,t-1,:]
            elif forcing=='never': feed=prev
            else:
                mask=(torch.rand(prev.shape,device=prev.device,generator=rng)<scheduled_prob)
                feed=torch.where(mask,targets[:,t-1,:],prev)
            pred,state=self.forward_inputs(feed.unsqueeze(1),state)
            predictions.append(pred);prev=pred
        return torch.stack(predictions,dim=1),state


def params(model): return sum(x.numel() for x in model.parameters() if x.requires_grad)


def match_hidden_sizes(rnn_hidden):
    target=params(ForecastNet('RNN',rnn_hidden))
    options={'RNN':rnn_hidden}
    for arch in ('LSTM','GRU'):
        options[arch]=min(range(4,160),key=lambda h:abs(params(ForecastNet(arch,h))-target))
    return options


def detach_state(state):
    if state is None:return None
    return tuple(s.detach() for s in state) if isinstance(state,tuple) else state.detach()


def tbptt_batch(model,context,target,opt,chunk,clipping,forcing,teacher_probability,device,
                collect_grads=False):
    """Correct truncated BPTT: detach recurrent state *and predicted feedback* at
    chunk boundaries, use one optimizer step per entire C+H-1 example/minibatch.
    The context and rollout are in one time-line. All H losses are weighted 1/H.
    """
    C=context.shape[1]; H=target.shape[1]
    total_steps=C+H-1
    chunk=total_steps if chunk=='full' else int(chunk)
    state=None; last_pred=None; total_loss=0.
    opt.zero_grad(set_to_none=True)
    for start in range(0,total_steps,chunk):
        end=min(start+chunk,total_steps)
        losses=[]
        # Entirely loss-free context blocks need no gradient graph.
        no_losses=(end<C)
        with (torch.no_grad() if no_losses else contextlib.nullcontext()):
            t=start
            # Vectorize the observed context, stopping at t=C; no prediction targets
            # except the one at the final observed time t=C-1.
            if t<C:
                stop=min(end,C)
                inputs=context[:,t:stop,:]
                last_pred,state=model.forward_inputs(inputs,state)
                if stop==C:
                    losses.append(torch.square(last_pred-target[:,0,:]).mean()/H)
                t=stop
            while t<end:
                # t=C corresponds to feeding target[0] or predicted target[0],
                # then forecasting target[1].
                prev_true=target[:,t-C,:]
                if forcing=='always': feed=prev_true
                elif forcing=='never': feed=last_pred
                elif forcing=='scheduled':
                    mask=(torch.rand(last_pred.shape,device=device)<teacher_probability)
                    feed=torch.where(mask,prev_true,last_pred)
                else: raise ValueError(forcing)
                last_pred,state=model.forward_inputs(feed.unsqueeze(1),state)
                losses.append(torch.square(last_pred-target[:,t-C+1,:]).mean()/H)
                t+=1
        if losses:
            subloss=torch.stack(losses).sum()
            if not torch.isfinite(subloss):raise FloatingPointError('non-finite loss in training')
            subloss.backward()
            total_loss += float(subloss.detach().cpu())
        state=detach_state(state)
        if last_pred is not None: last_pred=last_pred.detach()
    # Gradient norms are recorded BEFORE clipping.
    preclip=float(torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf')).item())
    if not math.isfinite(preclip):raise FloatingPointError('non-finite gradient norm')
    if clipping:
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    opt.step()
    return total_loss,preclip


def pred_rollout(model,context,H):
    pred,state=model.forward_inputs(context)
    out=[pred]
    for _ in range(1,H):
        pred,state=model.forward_inputs(pred.unsqueeze(1),state)
        out.append(pred)
    return torch.stack(out,dim=1)


def gpu_sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize(device)


def metrics_from_arrays(yhat, y, std):
    if not np.isfinite(yhat).all():raise FloatingPointError('non-finite forecast')
    delta=(yhat-y).astype(np.float64)
    H=delta.shape[1]
    r={'rmse_rollout':float(np.sqrt(np.mean(delta*delta))*std),
       'rmse_rollout_norm':float(np.sqrt(np.mean(delta*delta)))}
    for h in (1,20,50,100):
        if h<=H:r[f'rmse_{h}']=float(np.sqrt(np.mean(delta[:,h-1]**2))*std)
    return r

@torch.no_grad()
def eval_neural(model,ds,device,cfg,limit=None):
    model.eval(); targets=[]; predictions=[]
    for context,target in ds.batches(cfg['BATCH_SIZE'],seed=0,shuffle=False):
        context=context.to(device); target=target.to(device)
        p=pred_rollout(model,context,ds.H)
        targets.append(target.squeeze(-1).cpu().numpy())
        predictions.append(p.squeeze(-1).cpu().numpy())
    return metrics_from_arrays(np.concatenate(predictions),np.concatenate(targets),ds.std)


def persistence(ds):
    predictions=[];targets=[]
    for context,target in ds.batches(64):
        pred=context[:,-1:,0].numpy().repeat(ds.H,axis=1)
        predictions.append(pred);targets.append(target[:,:,0].numpy())
    return metrics_from_arrays(np.concatenate(predictions),np.concatenate(targets),ds.std)


def get_teacher_prob(forcing,epoch,epochs):
    if forcing=='always':return 1.
    if forcing=='never':return 0.
    return max(0.,1.-epoch/max(1,epochs-1))


def train_neural(cfg,info,device='cpu',tuning=False,lr=None):
    seed_all(info['seed'],device)
    tr,va,te=windows_for(cfg,info['dataset'])
    if tuning:
        tr.origins=tr.origins[np.linspace(0,len(tr.origins)-1,min(len(tr.origins),cfg['TUNE_TRAIN_ORIGINS']),dtype=np.int64)]
        va.origins=va.origins[:cfg['TUNE_EVAL_ORIGINS']]
    hidden=match_hidden_sizes(cfg['RNN_HIDDEN'])[info['architecture']]
    model=ForecastNet(info['architecture'],hidden).to(device)
    lr=float(lr if lr is not None else 0.001)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    n_epochs=cfg['TUNE_EPOCHS'] if tuning else cfg['EPOCHS']
    every=n_epochs if tuning else cfg['VALIDATE_EVERY']
    grad_norms=[]; history=[]; best_val=float('inf');best_state=None;best_epoch=None
    training_seconds=0.; wall_start=time.perf_counter(); n_updates=0
    time_to_train_mse_0_1=None
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
    try:
        for ep in range(n_epochs):
            model.train(); epoch_losses=[]; this_norms=[]
            gpu_sync(device);start=time.perf_counter()
            for ctx,target in tr.batches(cfg['BATCH_SIZE'],seed=info['seed']+ep,shuffle=True):
                ctx,target=ctx.to(device),target.to(device)
                train_loss,gn=tbptt_batch(model,ctx,target,opt,info['tbptt'],info['clip'],
                                          info['forcing'],get_teacher_prob(info['forcing'],ep,n_epochs),device)
                epoch_losses.append(train_loss);this_norms.append(gn);n_updates+=1
            gpu_sync(device);epoch_seconds=time.perf_counter()-start
            training_seconds+=epoch_seconds
            grad_norms.extend(this_norms)
            rec={'epoch':ep+1,'train_epoch_seconds':float(epoch_seconds),
                 'mean_train_loss':float(np.mean(epoch_losses)),
                 'grad_norm_mean_preclip':float(np.mean(this_norms)),
                 'optimizer_updates_so_far':n_updates}
            if ((ep+1)%every==0) or ep==0 or ep+1==n_epochs:
                val=eval_neural(model,va,device,cfg)
                rec['val_rmse_rollout_norm']=val['rmse_rollout_norm']
                if val['rmse_rollout_norm']<best_val:
                    best_val=val['rmse_rollout_norm'];best_epoch=ep+1
                    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            history.append(rec)
            if time_to_train_mse_0_1 is None and rec['mean_train_loss'] <= 0.1:
                time_to_train_mse_0_1=float(training_seconds)
        model.load_state_dict(best_state)
        val=eval_neural(model,va,device,cfg)
        result=dict(info,architecture=info['architecture'],seed=info['seed'],learning_rate=lr,
            hidden_size=hidden,trainable_parameters=params(model),best_epoch=best_epoch,
            train_origins=len(tr.origins),val_origins=len(va.origins),test_origins=len(te.origins),
            training_seconds=float(training_seconds),
            total_wall_seconds=float(time.perf_counter()-wall_start),
            gradient_norm_mean=float(np.mean(grad_norms)),
            gradient_norm_p95=float(np.percentile(grad_norms,95)),
            time_to_train_mse_0_1=time_to_train_mse_0_1,
            optimizer_updates=int(n_updates),diverged=False,
            peak_gpu_memory_mb=(float(torch.cuda.max_memory_allocated(device)/2**20)
                               if str(device).startswith('cuda') else None),
            peak_cpu_rss_mb=None,status='ok')
        result.update({'val_'+k:v for k,v in val.items()})
        if not tuning:
            test=eval_neural(model,te,device,cfg)
            result.update({'test_'+k:v for k,v in test.items()})
        return result,history
    finally:
        del model
        if str(device).startswith('cuda'): torch.cuda.empty_cache()


def esn_spectral_rescale(W,rho):
    # Radius approx via ARPACK (nonsymmetric sparse W). Fallback uses power iteration
    # only if ARPACK cannot converge; record fallback in metadata if needed.
    try:
        ev=eigs(W,k=1,which='LM',return_eigenvectors=False,maxiter=600,tol=.02)
        radius=float(np.max(np.abs(ev)))
    except Exception:
        v=np.random.default_rng(0).normal(size=W.shape[0])
        for _ in range(75):
            v=W@v
            v=v/(np.linalg.norm(v)+1e-12)
        radius=float(np.linalg.norm(W@v))
    if radius<1e-9:raise RuntimeError('Reservoir spectral radius nearly zero')
    return (W*(rho/radius)).tocsr()


class EchoState:
    def __init__(self,size,seed,params,cfg):
        self.size=int(size); self.leak=float(params['leak']);self.ridge=float(params['ridge'])
        rng=np.random.default_rng(seed)
        self.win=rng.uniform(-cfg['ESN_INPUT_SCALE'],cfg['ESN_INPUT_SCALE'],size=(self.size,)).astype(np.float64)
        self.bias=rng.uniform(-.1,.1,size=(self.size,)).astype(np.float64)
        density=float(cfg['ESN_DENSITY'])
        self.W=sparse.random(self.size,self.size,density=density,format='csr',random_state=seed,
            data_rvs=lambda s:rng.uniform(-1,1,size=s))
        self.W=esn_spectral_rescale(self.W,float(params['rho']))
        self.beta=None
    def step(self,x,state):
        return (1-self.leak)*state+self.leak*np.tanh(self.W@state+self.win*float(x)+self.bias)
    def features(self,x,state):
        return np.concatenate(([1.,float(x)],state))
    def fit(self,series_scaled,train_end,max_states,washout):
        n=int(train_end); idx=np.arange(max(washout,1),n-1,dtype=np.int64)
        if len(idx)>max_states:
            idx=idx[np.linspace(0,len(idx)-1,max_states,dtype=np.int64)]
        feat=np.empty((len(idx),self.size+2),dtype=np.float64)
        y=np.empty((len(idx),),dtype=np.float64)
        r=np.zeros(self.size,dtype=np.float64); p=0
        for t in range(n-1):
            x=float(series_scaled[t]);r=self.step(x,r)
            if p<len(idx) and t==idx[p]:
                feat[p]=self.features(x,r);y[p]=series_scaled[t+1];p+=1
        assert p==len(idx)
        # Normal equations + regularization (stable solve with conditioning check).
        gram=feat.T@feat
        gram.flat[::gram.shape[0]+1] += self.ridge
        self.beta=scipy_solve(gram,feat.T@y,assume_a='pos',check_finite=False)
        return len(idx)
    def predict_window(self,context,H):
        state=np.zeros(self.size,dtype=np.float64)
        for value in context:state=self.step(value,state)
        last=float(context[-1]);out=np.empty(H,dtype=np.float64)
        for j in range(H):
            guess=float(self.features(last,state)@self.beta)
            if not math.isfinite(guess):raise FloatingPointError('ESN predictions non-finite')
            out[j]=guess
            last=guess
            if j<H-1:state=self.step(last,state)
        return out


def eval_esn(model,ds):
    preds=[];targets=[]
    for context,target in ds.batches(16):
        cx=context.squeeze(-1).numpy();y=target.squeeze(-1).numpy()
        p=np.stack([model.predict_window(row,ds.H) for row in cx])
        preds.append(p);targets.append(y)
    return metrics_from_arrays(np.concatenate(preds),np.concatenate(targets),ds.std)


def train_esn(cfg,info,tuning=False,params=None):
    seed_all(info['seed'])
    tr,va,te=windows_for(cfg,info['dataset'])
    if tuning:va.origins=va.origins[:cfg['ESN_TUNE_EVAL_ORIGINS']]
    series=tr.scaled
    if params is None:params=cfg['ESN_CANDIDATES'][1]
    wall=time.perf_counter()
    model=EchoState(info['esn_size'],info['seed'],params,cfg)
    train_start=time.perf_counter()
    nstates=model.fit(series,tr.b,min(cfg['ESN_FIT_STATES'],tr.b-2),cfg['ESN_WASHOUT'])
    train_seconds=time.perf_counter()-train_start
    val=eval_esn(model,va)
    result=dict(info,status='ok',diverged=False,esn_fit_states=nstates,
                esn_spectral_radius_target=params['rho'],esn_leak=params['leak'],
                esn_ridge=params['ridge'],trainable_parameters=info['esn_size']+2,
                total_wall_seconds=time.perf_counter()-wall,
                training_seconds=train_seconds,
                cpu_rss_postfit_mb=float(psutil.Process().memory_info().rss/2**20),
                peak_cpu_rss_mb=float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0),
                peak_gpu_memory_mb=None,gradient_norm_mean=None,gradient_norm_p95=None,
                optimizer_updates=1,train_origins=nstates,val_origins=len(va.origins),
                test_origins=len(te.origins))
    result.update({'val_'+k:v for k,v in val.items()})
    if not tuning:
        test=eval_esn(model,te)
        result.update({'test_'+k:v for k,v in test.items()})
    return result,[]


def experiment_grid(cfg):
    preset=cfg['PRESET']
    if preset=='smoke':
        specs=[dict(dataset='mg',architecture=a,tbptt=20,clip=True,forcing='always',seed=42)
               for a in ('RNN','LSTM','GRU')]
        specs.append(dict(dataset='mg',architecture='ESN',esn_size=200,seed=42))
        return specs
    if preset=='minimum':
        datasets=['mg'];chunks=[20,50];forcing=['always','scheduled']
    else:
        datasets=cfg['DATASETS'];chunks=[20,50,'full'];forcing=['always','scheduled','never']
    specs=[dict(dataset=d,architecture=a,tbptt=k,clip=bool(c),forcing=f,seed=s)
           for d,a,k,c,f,s in itertools.product(datasets,['RNN','LSTM','GRU'],chunks,[0,1],forcing,SEEDS)]
    specs += [dict(dataset=d,architecture='ESN',esn_size=z,seed=s)
              for d,z,s in itertools.product(datasets,[200,500,1000],SEEDS)]
    return specs


def tuning_grid(cfg):
    if cfg['PRESET']=='smoke' or not cfg['RUN_TUNING']:return []
    return [(d,a) for d,a in itertools.product(cfg['DATASETS'],['RNN','LSTM','GRU','ESN'])]


def tuning_path(cfg,dataset,arch):return run_directory(cfg)/'tuning'/f'{dataset}_{arch}.json'


def do_tune(cfg,dataset,arch,device):
    path=tuning_path(cfg,dataset,arch)
    if path.exists(): return
    if arch=='ESN':
        trials=[]
        for p in cfg['ESN_CANDIDATES']:
            try:
                obj,_=train_esn(cfg,dict(dataset=dataset,architecture='ESN',esn_size=200,seed=42),True,p)
                trials.append(dict(params=p,score=obj['val_rmse_rollout_norm'],seconds=obj['total_wall_seconds']))
            except Exception as e:trials.append(dict(params=p,score=None,error=str(e)))
        good=[r for r in trials if r['score'] is not None]
        if not good:raise RuntimeError(f'ESN tuning failed: {trials}')
        best=min(good,key=lambda r:r['score'])
        save_json(path,dict(dataset=dataset,arch=arch,selection='validation free-running RMSE',
                            trials=trials,chosen_params=best['params']))
    else:
        trials=[]
        for lr in cfg['LR_CANDIDATES']:
            try:
                info=dict(dataset=dataset,architecture=arch,tbptt=50,clip=True,forcing='always',seed=42)
                obj,_=train_neural(cfg,info,device,True,lr)
                trials.append(dict(lr=lr,score=obj['val_rmse_rollout_norm'],seconds=obj['total_wall_seconds']))
            except Exception as e:trials.append(dict(lr=lr,score=None,error=str(e)))
        good=[r for r in trials if r['score'] is not None]
        if not good:raise RuntimeError(f'LR tuning failed: {trials}')
        best=min(good,key=lambda r:r['score'])
        save_json(path,dict(dataset=dataset,arch=arch,selection='validation free-running RMSE',
                            trials=trials,chosen_lr=best['lr']))


def execute_worker(cfg,worker_id,n_workers,device,mode):
    root=run_directory(cfg)
    if mode=='tune':
        tasks=tuning_grid(cfg)
        for i, (d,a) in enumerate(tasks):
            if i%n_workers!=worker_id:continue
            print(f'[worker {worker_id}] tuning {d}/{a}',flush=True)
            do_tune(cfg,d,a,device)
        return
    tasks=experiment_grid(cfg)
    for i,info in enumerate(tasks):
        if i%n_workers!=worker_id:continue
        key=digest(info)
        out=root/'runs'/(key+'.json')
        if out.exists():
            try:
                existing=json.loads(out.read_text())
                if existing.get('status')=='ok':continue
            except Exception:pass
        print(f'[worker {worker_id}] {i+1}/{len(tasks)} {info}',flush=True)
        wall=time.perf_counter(); hist=[]
        try:
            path=tuning_path(cfg,info['dataset'],info['architecture'])
            hyper=json.loads(path.read_text()) if path.exists() else {}
            if info['architecture']=='ESN':
                result,hist=train_esn(cfg,info,False,hyper.get('chosen_params'))
            else:
                result,hist=train_neural(cfg,info,device,False,hyper.get('chosen_lr',.001))
            result['run_key']=key
        except Exception as e:
            result={**info,'status':'failed','diverged':True,'run_key':key,
                    'total_wall_seconds':time.perf_counter()-wall,
                    'error':repr(e),'traceback':traceback.format_exc(limit=8)}
        save_json(out,result)
        if hist:
            save_json(root/'epochs'/(key+'.json'),hist)
        if result['status']!='ok':print(f"[worker {worker_id}] FAILED: {result['error']}",flush=True)
    print(f'[worker {worker_id}] done',flush=True)


def self_tests():
    # Shapes, matching first step, full-vs-long-chunk, TBPTT gradient detachment,
    # equal optimizer update counts, and forecast independence of future targets.
    seed_all(42)
    torch.set_num_threads(1)
    C,H=6,5; x=torch.randn(3,C,1);y=torch.randn(3,H,1)
    for arch in ['RNN','LSTM','GRU']:
        m=ForecastNet(arch,8)
        pred,st=m.forward_inputs(x)
        assert pred.shape==(3,1)
        if arch=='LSTM':assert isinstance(st,tuple) and len(st)==2
        with torch.no_grad():
            teacher,_=m(x,y,'always'); free,_=m(x,y,'never')
        assert torch.allclose(teacher[:,0],free[:,0],atol=1e-7)
        assert not torch.isnan(teacher).any()
        # first prediction is unaffected by any future target perturbation
        with torch.no_grad():
            teacher2,_=m(x,y*10000,'always')
        assert torch.allclose(teacher[:,0],teacher2[:,0],atol=1e-7)
        # Full BPTT equals non-truncated chunk >= graph length.
        m1=ForecastNet(arch,8);m2=ForecastNet(arch,8);m2.load_state_dict(m1.state_dict())
        o1=torch.optim.SGD(m1.parameters(),lr=.01);o2=torch.optim.SGD(m2.parameters(),lr=.01)
        l1,g1=tbptt_batch(m1,x,y,o1,'full',False,'always',1.,'cpu')
        l2,g2=tbptt_batch(m2,x,y,o2,C+H+10,False,'always',1.,'cpu')
        assert abs(l1-l2)<1e-6 and abs(g1-g2)<1e-5
        assert all(torch.allclose(p,q,atol=1e-6) for p,q in zip(m1.parameters(),m2.parameters()))
        # With a short chunk, one step still occurs per minibatch (not per chunk).
        m3=ForecastNet(arch,8);m3.load_state_dict(m1.state_dict())
        o3=torch.optim.SGD(m3.parameters(),lr=.01)
        l3,g3=tbptt_batch(m3,x,y,o3,2,True,'scheduled',.5,'cpu')
        assert math.isfinite(l3)
    # Synthetic MG should be nonlinear, reproducible, and finite.
    a=generate_mackey_glass(400,transient=100)
    b=generate_mackey_glass(400,transient=100)
    assert len(a)==400 and np.isfinite(a).all() and np.array_equal(a,b) and np.std(a)>1e-3
    # Test a miniature TSF parser with staggered starts and missing observations.
    txt='@attribute series_name string\n@attribute start_timestamp date\n@data\nA:2012-01-01 00-00-00:1,2,3\nB:2012-01-01 01-00-00:2,?,4'
    ag,meta=aggregate_electricity(parse_tsf_records(txt))
    assert len(ag)==4 and np.allclose(ag,[1,4,5,4])
    return 'PASS: RNN/LSTM/GRU tensor shapes, leakage, full-BPTT identity, TBPTT updates, MG and TSF parsing'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--mode',choices=['prepare','tune','run','test'],default='run')
    p.add_argument('--worker-id',type=int,default=0)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    if args.mode=='test': print(self_tests());return
    if args.mode=='prepare':print('DATA:',prepare_data(cfg));return
    execute_worker(cfg,args.worker_id,args.workers,args.device,args.mode)

if __name__=='__main__':main()
