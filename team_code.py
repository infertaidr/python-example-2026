#!/usr/bin/env python
import joblib
import json
import numpy as np
import os
import warnings
import mne
import pyedflib

mne.set_log_level('ERROR')
warnings.filterwarnings('ignore')

# MG1: montage-gated A / PE
FEATURE_COLS_A = [
    'log_ma_index', 'log_arousal_index', 'pct_rem', 'pct_nrem', 'pct_n2',
    'log_ilm_index', 'stage_transition_rate', 'waso_min', 'sleep_ineff',
    'n3_front_loading_ratio',
    'eeg_delta_rel_all', 'eeg_theta_rel_all', 'eeg_theta_delta_ratio'
]
FEATURE_COLS_PE = FEATURE_COLS_A + ['alpha_peak_frequency_occipital', 'stage_entropy_std']
INVALID_STAGES = [0, 9]

THRESHOLD = 0.63
EEG_N_WINDOWS = 16
EEG_WINDOW_SEC = 30

# occipital alpha_peak (v5b windowed 파이프라인과 동일 파라미터)
OCC_TARGET_FS = 100
OCC_EPOCH_SEC = 30
OCC_MIN_EP    = 5
OCC_K_STAGE   = 10
OCC_U_UNIFORM = 30

def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == '':
            return default
        return float(x)
    except:
        return default

def normalize_session_id(session_id):
    try:
        if isinstance(session_id, float) and session_id.is_integer():
            return str(int(session_id))
        s = str(session_id).strip()
        if s.endswith('.0'):
            s = s[:-2]
        return s
    except:
        return str(session_id)

def get_native_sfreq(raw, ch_name):
    try:
        ch_idx  = raw.ch_names.index(ch_name)
        n_samps = raw._raw_extras[0]['n_samps']
        rec_len = raw._raw_extras[0]['record_length']
        return float(n_samps[ch_idx]) / float(rec_len)
    except:
        return raw.info['sfreq']

def find_channel(raw, keywords, exclude=None):
    exclude = exclude or []
    for ch in raw.ch_names:
        chl = ch.lower()
        if all(k in chl for k in keywords) and not any(e in chl for e in exclude):
            return ch
    return None

def count_events(arr):
    padded = np.concatenate([[0], (arr > 0.5).astype(int)])
    return int(np.sum(np.diff(padded) == 1))

def get_stage_epochs(raw, ch_name):
    native_sfreq = get_native_sfreq(raw, ch_name)
    common_sfreq = raw.info['sfreq']
    raw_data     = raw.get_data(picks=ch_name)[0]
    if abs(common_sfreq - native_sfreq) > 0.01:
        step   = max(int(round(common_sfreq / native_sfreq)), 1)
        epochs = np.round(raw_data[::step]).astype(int)
    else:
        if abs(native_sfreq - 1.0/30.0) < 0.01:
            epochs = np.round(raw_data).astype(int)
        elif 0.9 <= native_sfreq <= 1.1:
            epochs = np.round(raw_data[::30]).astype(int)
        else:
            step   = max(1, int(round(native_sfreq * 30)))
            epochs = np.round(raw_data[::step]).astype(int)
    return epochs

def get_event_data(raw, ch_name):
    native_sfreq = get_native_sfreq(raw, ch_name)
    common_sfreq = raw.info['sfreq']
    raw_data     = raw.get_data(picks=ch_name)[0]
    if abs(common_sfreq - native_sfreq) > 0.01 and native_sfreq > 0:
        step = max(int(round(common_sfreq / native_sfreq)), 1)
        data = np.round(raw_data[::step]).astype(int)
    else:
        data = np.round(raw_data).astype(int)
    return data, native_sfreq

def trapz_compat(y, x, axis=-1):
    try:
        return np.trapezoid(y, x, axis=axis)
    except AttributeError:
        return np.trapz(y, x, axis=axis)

def find_annot_file_by_bids(bids_folder, session_id, site_id, data_folder):
    annot_base = os.path.join(data_folder, 'algorithmic_annotations')
    if not os.path.isdir(annot_base):
        return None
    bids_folder = str(bids_folder).strip() if bids_folder is not None else ''
    session_id  = normalize_session_id(session_id)
    site_id     = str(site_id).strip() if site_id is not None else ''
    if not bids_folder or not session_id:
        return None
    target_name = f"{bids_folder}_ses-{session_id}_caisr_annotations.edf"
    if site_id:
        candidate = os.path.join(annot_base, site_id, target_name)
        if os.path.exists(candidate):
            return candidate
    for site_folder in os.listdir(annot_base):
        site_path = os.path.join(annot_base, site_folder)
        if not os.path.isdir(site_path):
            continue
        candidate = os.path.join(site_path, target_name)
        if os.path.exists(candidate):
            return candidate
    for root, _, files in os.walk(annot_base):
        for fname in files:
            if (fname.endswith('.edf') and bids_folder in fname
                    and f"ses-{session_id}" in fname and "caisr" in fname.lower()):
                return os.path.join(root, fname)
    return None

def find_annot_file_train(patient_id, data_folder):
    annot_base = os.path.join(data_folder, 'algorithmic_annotations')
    if not os.path.exists(annot_base):
        return None
    pid_str = str(patient_id).strip()
    if not pid_str:
        return None
    for site_folder in os.listdir(annot_base):
        site_path = os.path.join(annot_base, site_folder)
        if not os.path.isdir(site_path):
            continue
        for fname in os.listdir(site_path):
            if fname.endswith('.edf') and pid_str in fname:
                return os.path.join(site_path, fname)
    return None

def find_phys_file_by_bids(bids_folder, session_id, site_id, data_folder):
    phys_base = os.path.join(data_folder, 'physiological_data')
    if not os.path.isdir(phys_base):
        return None
    bids_folder = str(bids_folder).strip() if bids_folder is not None else ''
    session_id  = normalize_session_id(session_id)
    site_id     = str(site_id).strip() if site_id is not None else ''
    if not bids_folder or not session_id:
        return None
    target_name = f"{bids_folder}_ses-{session_id}.edf"
    if site_id:
        candidate = os.path.join(phys_base, site_id, target_name)
        if os.path.exists(candidate):
            return candidate
    for site_folder in os.listdir(phys_base):
        site_path = os.path.join(phys_base, site_folder)
        if not os.path.isdir(site_path):
            continue
        candidate = os.path.join(site_path, target_name)
        if os.path.exists(candidate):
            return candidate
    for root, _, files in os.walk(phys_base):
        for fname in files:
            if (fname.endswith('.edf') and bids_folder in fname
                    and f"ses-{session_id}" in fname):
                return os.path.join(root, fname)
    return None

def find_phys_file_train(patient_id, data_folder):
    phys_base = os.path.join(data_folder, 'physiological_data')
    if not os.path.exists(phys_base):
        return None
    pid_str = str(patient_id).strip()
    if not pid_str:
        return None
    for site_folder in os.listdir(phys_base):
        site_path = os.path.join(phys_base, site_folder)
        if not os.path.isdir(site_path):
            continue
        for fname in os.listdir(site_path):
            if fname.endswith('.edf') and pid_str in fname:
                return os.path.join(site_path, fname)
    return None

def extract_caisr_features(edf_path):
    if not edf_path or not os.path.exists(edf_path):
        return {}
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except:
        return {}

    f = {}
    valid_epochs = None

    stage_ch = find_channel(raw, ['stage','caisr'], exclude=['prob'])
    if stage_ch:
        epochs = get_stage_epochs(raw, stage_ch)
        valid  = epochs[(epochs >= 1) & (epochs <= 5)]
        n_ep   = len(valid)
        if n_ep > 0:
            n_sleep = int(np.sum(valid != 5))
            n_wake  = int(np.sum(valid == 5))
            half    = n_ep // 2
            f['tst_min']     = round(n_sleep * 30/60, 2)
            f['tib_min']     = round(n_ep * 30/60, 2)
            f['sleep_eff']   = round(n_sleep/n_ep*100, 2)
            f['waso_min']    = round(n_wake * 30/60, 2)
            f['sleep_ineff'] = round(100.0 - f['sleep_eff'], 2)
            f['waso_pct']    = round(f['waso_min']/(f['tib_min']+1e-4)*100, 2)
            if n_sleep > 0:
                f['pct_n3']  = round(np.sum(valid==1)/n_sleep*100, 3)
                f['pct_n2']  = round(np.sum(valid==2)/n_sleep*100, 3)
                f['pct_n1']  = round(np.sum(valid==3)/n_sleep*100, 3)
                f['pct_rem'] = round(np.sum(valid==4)/n_sleep*100, 3)
                f['pct_nrem']= round((np.sum(valid==1)+np.sum(valid==2)+
                                      np.sum(valid==3))/n_sleep*100, 3)
                valid_epochs = valid
                f['stage_transition_rate'] = round(
                    float(np.sum(np.diff(valid)!=0))/n_ep, 4)
                f['n3_front_loading_ratio'] = round(
                    (np.sum(valid[:half]==1)+1e-4)/
                    (np.sum(valid[half:]==1)+1e-4), 4)
            else:
                for k in ['pct_n3','pct_n2','pct_n1','pct_rem','pct_nrem',
                          'stage_transition_rate','n3_front_loading_ratio']:
                    f[k] = np.nan
        else:
            for k in ['tst_min','tib_min','sleep_eff','waso_min','sleep_ineff',
                      'waso_pct','pct_n3','pct_n2','pct_n1','pct_rem','pct_nrem',
                      'stage_transition_rate','n3_front_loading_ratio']:
                f[k] = np.nan

    resp_ch = find_channel(raw, ['resp','caisr'], exclude=['prob'])
    if resp_ch and not np.isnan(f.get('tst_min', np.nan)):
        resp_data, _ = get_event_data(raw, resp_ch)
        def cnt(vals):
            return count_events(np.isin(resp_data, vals).astype(int))
        f['n_ma'] = cnt([3])
    else:
        f['n_ma'] = np.nan

    arou_ch = find_channel(raw, ['arousal','caisr'], exclude=['prob'])
    if arou_ch and not np.isnan(f.get('tst_min', np.nan)):
        arou_data, _ = get_event_data(raw, arou_ch)
        f['n_arousals'] = count_events(arou_data)
    else:
        f['n_arousals'] = np.nan

    limb_ch = find_channel(raw, ['limb','caisr'], exclude=['prob'])
    if limb_ch and not np.isnan(f.get('tst_min', np.nan)):
        limb_data, _ = get_event_data(raw, limb_ch)
        f['n_ilm'] = count_events((limb_data==1).astype(int))
    else:
        f['n_ilm'] = np.nan

    tst_hours = max(f.get('tst_min', np.nan)/60.0, 1e-6) \
        if not np.isnan(f.get('tst_min', np.nan)) else np.nan

    if not np.isnan(tst_hours) and tst_hours > 0.1:
        ma_idx      = f.get('n_ma', np.nan)/tst_hours if not np.isnan(f.get('n_ma', np.nan)) else np.nan
        arousal_idx = f.get('n_arousals', np.nan)/tst_hours if not np.isnan(f.get('n_arousals', np.nan)) else np.nan
        ilm_idx     = f.get('n_ilm', np.nan)/tst_hours if not np.isnan(f.get('n_ilm', np.nan)) else np.nan
    else:
        ma_idx = arousal_idx = ilm_idx = np.nan

    f['log_ma_index']      = round(float(np.log1p(ma_idx)), 4) if not np.isnan(ma_idx) else np.nan
    f['log_arousal_index'] = round(float(np.log1p(arousal_idx)), 4) if not np.isnan(arousal_idx) else np.nan
    f['log_ilm_index']     = round(float(np.log1p(ilm_idx)), 4) if not np.isnan(ilm_idx) else np.nan

    return f

def pick_eeg_channel_pyedf(labels):
    labels_lower = [l.lower() for l in labels]
    priorities = [['c3-m2'],['c3-a2'],['c3'],
                  ['f3-m2'],['f3-a2'],['f3'],
                  ['c4-m1'],['c4-a1'],['c4'],
                  ['o1-m2'],['o1-a2'],['o1']]
    for kws in priorities:
        for i, l in enumerate(labels_lower):
            if all(k in l for k in kws):
                return i, labels[i]
    return None, None

def extract_eeg_relative_sparse(phys_path, n_windows=EEG_N_WINDOWS, window_sec=EEG_WINDOW_SEC):
    keys = ['eeg_delta_rel_all','eeg_theta_rel_all','eeg_theta_delta_ratio']
    default = {k: np.nan for k in keys}

    if not phys_path or not os.path.exists(phys_path):
        return default
    try:
        f = pyedflib.EdfReader(phys_path)
        labels = f.getSignalLabels()
        idx, ch_name = pick_eeg_channel_pyedf(labels)
        if idx is None:
            f.close()
            return default
        fs_orig = f.getSampleFrequency(idx)
        n_samples_total = f.getNSamples()[idx]

        win_samples = int(window_sec * fs_orig)
        positions = np.linspace(0.1, 0.9, n_windows)

        segments = []
        for p in positions:
            start_sample = int(p * n_samples_total)
            start_sample = min(start_sample, max(n_samples_total - win_samples, 0))
            start_sample = max(start_sample, 0)
            seg = f.readSignal(idx, start=start_sample, n=win_samples)
            segments.append(seg)
        f.close()
    except:
        try: f.close()
        except: pass
        return default

    try:
        from scipy.signal import welch, butter, filtfilt, resample

        eeg_data = np.concatenate(segments)

        if np.nanstd(eeg_data) < 0.01:
            eeg_data = eeg_data * 1e6

        fs = fs_orig
        if fs_orig != 100:
            n_new = int(len(eeg_data) * 100 / fs_orig)
            eeg_data = resample(eeg_data, n_new)
            fs = 100.0

        b, a = butter(4, [0.5/(fs/2), 30/(fs/2)], btype='band')
        eeg_f = filtfilt(b, a, eeg_data)

        seg_len = int(fs * window_sec)
        n_segs  = len(eeg_f) // seg_len
        if n_segs < 3:
            return default
        segs_2d = eeg_f[:n_segs*seg_len].reshape(n_segs, seg_len)

        amp = np.ptp(segs_2d, axis=1)
        std = np.std(segs_2d, axis=1)
        amp_thresh = np.nanpercentile(amp, 95) if n_segs > 5 else amp.max()+1
        good_mask = (amp < amp_thresh) & (std > 1e-4)
        good_segs = segs_2d[good_mask]

        if len(good_segs) < 3:
            return default

        nperseg = min(int(fs*4), good_segs.shape[1])
        freqs, psd = welch(good_segs, fs=fs, nperseg=nperseg, axis=1)

        delta_mask = (freqs >= 0.5) & (freqs < 4)
        theta_mask = (freqs >= 4) & (freqs < 8)
        total_mask = (freqs >= 0.5) & (freqs < 30)

        delta_power = trapz_compat(psd[:, delta_mask], freqs[delta_mask], axis=1)
        theta_power = trapz_compat(psd[:, theta_mask], freqs[theta_mask], axis=1)
        total_power = trapz_compat(psd[:, total_mask], freqs[total_mask], axis=1)

        delta_rel = delta_power / (total_power + 1e-10)
        theta_rel = theta_power / (total_power + 1e-10)

        result = {}
        result['eeg_delta_rel_all'] = round(float(np.nanmean(delta_rel)), 6)
        result['eeg_theta_rel_all'] = round(float(np.nanmean(theta_rel)), 6)
        result['eeg_theta_delta_ratio'] = round(
            float(np.nanmean(theta_power) / max(np.nanmean(delta_power), 1e-8)), 6)
        return result
    except:
        return default

# ============================================================
# occipital alpha_peak 추출 (v5b windowed 이식 = parity)
# ============================================================
def _occ_clean(s):
    return s.lower().replace('eeg','').replace(':','-').replace('/','-').replace('_','').strip()

_OCC_DERIV = [('o1', ['m2','a2']), ('o2', ['m1','a1'])]

def _resolve_occ(labels):
    cl = [_occ_clean(l) for l in labels]
    for prim, refs in _OCC_DERIV:
        for i, c in enumerate(cl):
            if prim in c and any(r in c for r in refs):
                return ('direct', i, None)
        pi = next((i for i, c in enumerate(cl) if c == prim), None)
        if pi is not None:
            for r in refs:
                ri = next((i for i, c in enumerate(cl) if c == r), None)
                if ri is not None:
                    return ('derive', pi, ri)
    return (None, None, None)

def _occ_decode_stage(raw, fs):
    spe = fs * OCC_EPOCH_SEC
    if spe < 1.5:
        return np.round(raw).astype(int)
    return np.round(raw[::int(round(spe))]).astype(int)

def _occ_load_stage(annot_path):
    try:
        f = pyedflib.EdfReader(annot_path)
        labels = f.getSignalLabels()
        idx = next((i for i, l in enumerate(labels)
                    if 'stage' in l.lower() and 'caisr' in l.lower() and 'prob' not in l.lower()), None)
        if idx is None:
            f.close(); return None
        raw = f.readSignal(idx); fs = f.getSampleFrequency(idx); f.close()
        return _occ_decode_stage(raw, fs)
    except:
        try: f.close()
        except: pass
        return None

def _occ_select_epochs(stage):
    sel = set()
    for s in [1, 2, 3, 4, 5]:
        idx = np.where(stage == s)[0]
        if len(idx) == 0:
            continue
        pick = idx if len(idx) <= OCC_K_STAGE else idx[np.linspace(0, len(idx)-1, OCC_K_STAGE).astype(int)]
        sel.update(pick.tolist())
    n = len(stage)
    sel.update(set(np.linspace(0, n-1, min(OCC_U_UNIFORM, n)).astype(int).tolist()))
    return sorted(sel)

def _occ_read_win(f, mode, pi, ri, fs, e, nsamp):
    L = int(fs * OCC_EPOCH_SEC); start = int(e * L)
    if start + L > nsamp[pi]:
        start = max(nsamp[pi] - L, 0)
    try:
        a = f.readSignal(pi, start=start, n=L)
        if mode == 'derive':
            b = f.readSignal(ri, start=start, n=L); a = a - b[:len(a)]
        return np.asarray(a, float)
    except:
        return None

def _occ_win_alpha_peak(win, fs):
    if win is None or len(win) < fs * 5:
        return None
    from scipy.signal import welch, butter, filtfilt, resample_poly
    if abs(fs - OCC_TARGET_FS) > 0.1:
        up, down = OCC_TARGET_FS, int(round(fs)); g = np.gcd(up, down)
        win = resample_poly(win, up // g, down // g)
    fsr = float(OCC_TARGET_FS)
    if not np.isfinite(win).all() or np.std(win) < 1e-6:
        return None
    try:
        b, a = butter(4, [0.5/(fsr/2), 35/(fsr/2)], btype='band'); win = filtfilt(b, a, win)
    except:
        return None
    fr, psd = welch(win, fs=fsr, nperseg=min(int(fsr*4), len(win)))
    amk = (fr >= 7) & (fr <= 13)
    return fr[amk][np.argmax(psd[amk])]

def extract_occipital_alpha_peak(phys_path, annot_path):
    default = {'alpha_peak_frequency_occipital': np.nan}
    if not phys_path or not os.path.exists(phys_path):
        return default
    stage = _occ_load_stage(annot_path) if annot_path else None
    if stage is None:
        return default
    try:
        f = pyedflib.EdfReader(phys_path)
        labels = f.getSignalLabels(); nsamp = f.getNSamples()
        mode, pi, ri = _resolve_occ(labels)
        if mode is None:
            f.close(); return default
        fs = f.getSampleFrequency(pi)
        sel = _occ_select_epochs(stage)
        max_ep = nsamp[pi] // int(fs * OCC_EPOCH_SEC)
        sel = [e for e in sel if e < max_ep and e < len(stage)]
        peaks = []
        for e in sel:
            if int(stage[e]) in (4, 5):   # REM + Wake
                peaks.append(_occ_win_alpha_peak(_occ_read_win(f, mode, pi, ri, fs, e, nsamp), fs))
        f.close()
        v = np.array([x for x in peaks if x is not None and np.isfinite(x)])
        return {'alpha_peak_frequency_occipital': float(np.median(v)) if len(v) >= OCC_MIN_EP else np.nan}
    except:
        try: f.close()
        except: pass
        return default

def extract_stage_entropy_std(annot_path):
    """CAISR stage-probability(N3,N2,N1,REM,W) entropy의 환자 내 std.
       arousal_extract.py와 동일 로직 (parity)."""
    default = {'stage_entropy_std': np.nan}
    if not annot_path or not os.path.exists(annot_path):
        return default
    try:
        f = pyedflib.EdfReader(annot_path)
        labels = f.getSignalLabels()
        si = next((i for i, l in enumerate(labels)
                   if 'stage' in l.lower() and 'caisr' in l.lower() and 'prob' not in l.lower()), None)
        prob_idx = {}
        for key in ['n3', 'n2', 'n1', 'r', 'w']:
            j = next((i for i, l in enumerate(labels)
                      if l.lower() == f'caisr_prob_{key}' or l.lower().endswith(f'prob_{key}')), None)
            if j is not None:
                prob_idx[key] = j
        if si is None or len(prob_idx) != 5:
            f.close(); return default
        sfs = f.getSampleFrequency(si)
        stage = _occ_decode_stage(f.readSignal(si), sfs)
        P = []
        for key in ['n3', 'n2', 'n1', 'r', 'w']:
            pr = f.readSignal(prob_idx[key]); pfs = f.getSampleFrequency(prob_idx[key])
            pe = pr if pfs * 30 < 1.5 else pr[::int(round(pfs * 30))]
            P.append(pe)
        f.close()
        m = min(len(p) for p in P)
        P = np.vstack([p[:m] for p in P]).T
        stg = stage[:m] if len(stage) >= m else np.concatenate([stage, [9] * (m - len(stage))])
        valid = ~np.isin(stg, INVALID_STAGES)
        Pv = P[valid]
        if len(Pv) < 10:
            return default
        s = Pv.sum(axis=1, keepdims=True)
        Pn = np.clip(Pv / np.where(s > 0, s, 1), 1e-12, 1)
        ent = -np.sum(Pn * np.log(Pn), axis=1)
        return {'stage_entropy_std': float(np.std(ent))}
    except:
        try: f.close()
        except: pass
        return default

def resolve_occipital_mode(phys_path):
    """occipital 채널 mode: 'direct'/'derive'/'missing'. alpha 추출 resolver와 동일 소스."""
    if not phys_path or not os.path.exists(phys_path):
        return 'missing'
    try:
        f = pyedflib.EdfReader(phys_path)
        labels = f.getSignalLabels(); f.close()
        mode, pi, ri = _resolve_occ(labels)
        return mode if mode is not None else 'missing'
    except:
        try: f.close()
        except: pass
        return 'missing'

def train_model(data_folder, model_folder, verbose=False):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    os.makedirs(model_folder, exist_ok=True)
    demo = pd.read_csv(os.path.join(data_folder, 'demographics.csv'))
    if demo['Cognitive_Impairment'].dtype == object:
        demo['Cognitive_Impairment'] = demo['Cognitive_Impairment'].map(
            {'TRUE':True,'True':True,'true':True,
             'FALSE':False,'False':False,'false':False})

    records = []
    for i, (_, row) in enumerate(demo.iterrows()):
        pid = row['BDSPPatientID']
        annot_path = find_annot_file_train(pid, data_folder)
        caisr_feat = extract_caisr_features(annot_path)
        phys_path = find_phys_file_train(pid, data_folder)
        eeg_feat = extract_eeg_relative_sparse(phys_path)
        occ_feat = extract_occipital_alpha_peak(phys_path, annot_path)
        ent_feat = extract_stage_entropy_std(annot_path)
        feats = {**caisr_feat, **eeg_feat, **occ_feat, **ent_feat}
        feats['BDSPPatientID'] = pid
        records.append(feats)
        if verbose and (i+1) % 200 == 0:
            print(f'  [{i+1}/{len(demo)}]')

    caisr_df = pd.DataFrame(records)
    df = demo[['BDSPPatientID','Cognitive_Impairment']].merge(
        caisr_df, on='BDSPPatientID', how='left')
    y = df['Cognitive_Impairment'].astype(int).values

    def fit_one(cols):
        impute = {}
        Xdf = df[cols].copy()
        for c in cols:
            med = Xdf[c].median()
            if not np.isfinite(med): med = 0.0
            impute[c] = float(med)
            Xdf[c] = Xdf[c].fillna(med)
        model = Pipeline([('scaler', StandardScaler()),
                          ('clf', LogisticRegression(class_weight='balanced',
                                                     max_iter=1000, random_state=42))])
        model.fit(Xdf[cols].values.astype(float), y)
        return model, impute

    model_A,  impute_A  = fit_one(FEATURE_COLS_A)
    model_PE, impute_PE = fit_one(FEATURE_COLS_PE)

    joblib.dump(model_A,  os.path.join(model_folder, 'model_A.pkl'))
    joblib.dump(model_PE, os.path.join(model_folder, 'model_PE.pkl'))
    with open(os.path.join(model_folder, 'meta.json'), 'w') as fj:
        json.dump({'features_A': FEATURE_COLS_A, 'features_PE': FEATURE_COLS_PE,
                   'impute_A': impute_A, 'impute_PE': impute_PE,
                   'threshold': THRESHOLD}, fj)
    if verbose:
        print('MG1 models saved (A + PE).')

def load_model(model_folder, verbose=False):
    model_A  = joblib.load(os.path.join(model_folder, 'model_A.pkl'))
    model_PE = joblib.load(os.path.join(model_folder, 'model_PE.pkl'))
    with open(os.path.join(model_folder, 'meta.json')) as f:
        meta = json.load(f)
    if verbose:
        print('MG1 models loaded. threshold=', meta['threshold'])
    return {'model_A': model_A, 'model_PE': model_PE, 'meta': meta}

def run_model(model_dict, record, data_folder, verbose=False):
    try:
        meta = model_dict['meta']
        threshold = meta.get('threshold', THRESHOLD)
        bids_folder = record.get('BidsFolder', '')
        site_id     = record.get('SiteID', '')
        session_id  = record.get('SessionID', '')

        annot_path = find_annot_file_by_bids(bids_folder, session_id, site_id, data_folder)
        phys_path  = find_phys_file_by_bids(bids_folder, session_id, site_id, data_folder)

        caisr_feat = extract_caisr_features(annot_path) if annot_path else {}
        eeg_feat   = extract_eeg_relative_sparse(phys_path) if phys_path else {}
        occ_feat   = extract_occipital_alpha_peak(phys_path, annot_path) if phys_path else {}
        ent_feat   = extract_stage_entropy_std(annot_path) if annot_path else {}
        feat = {**caisr_feat, **eeg_feat, **occ_feat, **ent_feat}

        # montage-gated hard routing: direct -> PE, else(derive/missing) -> A
        mode = resolve_occipital_mode(phys_path)
        if mode == 'direct':
            model = model_dict['model_PE']; cols = meta['features_PE']; impute = meta['impute_PE']
        else:
            model = model_dict['model_A'];  cols = meta['features_A'];  impute = meta['impute_A']

        vec = []
        for col in cols:
            v = feat.get(col, np.nan)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = impute.get(col, 0.0)
            vec.append(float(v))
        prob = float(model.predict_proba([vec])[0][1])
        binary = int(prob >= threshold)
        return binary, prob
    except Exception as e:
        if verbose:
            print(f'Error: {e}')
        return 0, 0.5
