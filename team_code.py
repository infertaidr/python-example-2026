#!/usr/bin/env python

# ============================================================
# PhysioNet Challenge 2026 | Team: Momochi-SleepAI
#
# Model    : LogisticRegression + StandardScaler
# Features : 13개 (CAISR rate-normalized 10개 + sparse EEG 3개)
# Strategy : v2(rate-normalized CAISR)에 sparse-window EEG spectral
#            feature를 추가. raw EEG 전체를 읽지 않고 deterministic
#            sparse window(16 x 30초, 10~90% 위치)만 읽어 runtime을
#            CAISR 수준으로 유지하면서 EEG 정보를 반영.
# Threshold: 0.63 (prevalence-based reward 최적화, plateau 확인됨)
#
# Version history:
#   v1   : run_model()에서 BDSPPatientID로 환자 식별 -> hidden record에
#          BDSPPatientID가 없어 매칭 실패 버그. Age-cond AUROC=0.500
#   v1.1 : BidsFolder+SiteID+SessionID 기반 매칭으로 수정.
#          12개 raw-count CAISR feature. Age-cond=0.584, Reward=0.021 (hidden)
#   v2   : raw event count -> TST 기준 per-hour rate -> log1p 변환.
#          11개 feature, threshold=0.58. Age-cond=0.574, Reward=0.050 (hidden)
#   v3   : v2에 sparse-window EEG spectral feature(delta/theta relative
#          power, theta/delta ratio) 추가. CAISR feature는 drop-one
#          ablation으로 plm_index, stage_entropy 제거(redundant 확인),
#          n3_front_loading_ratio 추가(EEG와 시너지 확인).
#          Local 검증(S0001->I0006/I0002 + LOSO stress test):
#            worst age-cond: 0.7345(v2) -> 0.7694 (+0.0349)
#            site gap: 0.0001 -> 0.0014 (균형 유지)
#            stress test: 0.5856 -> 0.5937 (개선)
#            worst reward proxy: 0.3148(v2) -> 0.3563 (+0.0415)
#          전체 EEG 원본 읽기는 runtime 9.57초/명(p95 20.71초, max 48.71초)
#          으로 위험했으나, sparse-window 방식은 같은 성능을 유지하면서
#          runtime을 0.02~0.17초/명 수준으로 낮춤 (full-read와 r=0.83~0.94
#          상관관계로 신호 보존 확인).
# ============================================================

import joblib
import json
import numpy as np
import os
import warnings
import mne
import pyedflib

mne.set_log_level('ERROR')
warnings.filterwarnings('ignore')

# ============================================================
# 확정 feature 목록 (순서 고정 - 변경 금지)
# ============================================================
FEATURE_COLS = [
    'log_ma_index', 'log_arousal_index', 'pct_rem', 'pct_nrem', 'pct_n2',
    'log_ilm_index', 'stage_transition_rate', 'waso_min', 'sleep_ineff',
    'n3_front_loading_ratio',
    'eeg_delta_rel_all', 'eeg_theta_rel_all', 'eeg_theta_delta_ratio'
]

THRESHOLD = 0.63

# EEG sparse-window 설정
EEG_N_WINDOWS = 16
EEG_WINDOW_SEC = 30

# ============================================================
# 유틸 함수
# ============================================================
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
    """1, 1.0, '1', '1.0' 등 다양한 형태를 '1'로 통일"""
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

# ============================================================
# 파일 탐색 (BidsFolder + SessionID + SiteID 기반)
# ============================================================
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
    """train_model()용. demographics.csv에는 BDSPPatientID가 항상 존재."""
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
    """train_model()용. demographics.csv에는 BDSPPatientID가 항상 존재."""
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

# ============================================================
# CAISR feature 추출
# ============================================================
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

    # Respiratory (n_ma만 필요)
    resp_ch = find_channel(raw, ['resp','caisr'], exclude=['prob'])
    if resp_ch and not np.isnan(f.get('tst_min', np.nan)):
        resp_data, _ = get_event_data(raw, resp_ch)
        def cnt(vals):
            return count_events(np.isin(resp_data, vals).astype(int))
        f['n_ma'] = cnt([3])
    else:
        f['n_ma'] = np.nan

    # Arousal
    arou_ch = find_channel(raw, ['arousal','caisr'], exclude=['prob'])
    if arou_ch and not np.isnan(f.get('tst_min', np.nan)):
        arou_data, _ = get_event_data(raw, arou_ch)
        f['n_arousals'] = count_events(arou_data)
    else:
        f['n_arousals'] = np.nan

    # Limb (n_ilm만 필요)
    limb_ch = find_channel(raw, ['limb','caisr'], exclude=['prob'])
    if limb_ch and not np.isnan(f.get('tst_min', np.nan)):
        limb_data, _ = get_event_data(raw, limb_ch)
        f['n_ilm'] = count_events((limb_data==1).astype(int))
    else:
        f['n_ilm'] = np.nan

    # raw count -> TST 기준 per-hour rate -> log1p
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

# ============================================================
# EEG sparse-window feature 추출
# ============================================================
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
# train_model
# ============================================================
def train_model(data_folder, model_folder, verbose=False):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    os.makedirs(model_folder, exist_ok=True)
    if verbose:
        print('Loading demographics...')

    demo = pd.read_csv(os.path.join(data_folder, 'demographics.csv'))
    if demo['Cognitive_Impairment'].dtype == object:
        demo['Cognitive_Impairment'] = demo['Cognitive_Impairment'].map(
            {'TRUE':True,'True':True,'true':True,
             'FALSE':False,'False':False,'false':False})

    if verbose:
        print(f'Extracting CAISR + EEG features for {len(demo)} patients...')

    records = []
    for i, (_, row) in enumerate(demo.iterrows()):
        pid = row['BDSPPatientID']
        annot_path = find_annot_file_train(pid, data_folder)
        caisr_feat = extract_caisr_features(annot_path)

        phys_path = find_phys_file_train(pid, data_folder)
        eeg_feat = extract_eeg_relative_sparse(phys_path)

        feats = {**caisr_feat, **eeg_feat}
        feats['BDSPPatientID'] = pid
        records.append(feats)
        if verbose and (i+1) % 500 == 0:
            print(f'  [{i+1}/{len(demo)}]')

    caisr_df = pd.DataFrame(records)
    df = demo[['BDSPPatientID','Cognitive_Impairment']].merge(
        caisr_df, on='BDSPPatientID', how='left')

    # Training data median imputation (NaN fallback 포함)
    impute_vals = {}
    for col in FEATURE_COLS:
        if col in df.columns:
            med = df[col].median()
            if not np.isfinite(med):
                med = 0.0
            impute_vals[col] = float(med)
            df[col] = df[col].fillna(med)

    X = df[FEATURE_COLS].values.astype(float)
    y = df['Cognitive_Impairment'].astype(int).values

    if verbose:
        print(f'Training LR+Scaler... (n={len(X)}, CI={y.sum()})')

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42))
    ])
    model.fit(X, y)

    joblib.dump(model, os.path.join(model_folder, 'model.pkl'))
    with open(os.path.join(model_folder, 'features.json'), 'w') as fj:
        json.dump(FEATURE_COLS, fj)
    with open(os.path.join(model_folder, 'impute_vals.json'), 'w') as fj:
        json.dump(impute_vals, fj)
    with open(os.path.join(model_folder, 'threshold.json'), 'w') as fj:
        json.dump({'threshold': THRESHOLD}, fj)

    if verbose:
        print('Model saved!')

# ============================================================
# load_model
# ============================================================
def load_model(model_folder, verbose=False):
    model = joblib.load(os.path.join(model_folder, 'model.pkl'))
    with open(os.path.join(model_folder, 'features.json')) as f:
        features = json.load(f)
    with open(os.path.join(model_folder, 'impute_vals.json')) as f:
        impute_vals = json.load(f)
    threshold = THRESHOLD
    if os.path.exists(os.path.join(model_folder, 'threshold.json')):
        with open(os.path.join(model_folder, 'threshold.json')) as f:
            threshold = json.load(f)['threshold']
    if verbose:
        print(f'Model loaded! threshold={threshold}')
    return {'model': model, 'features': features,
            'impute_vals': impute_vals, 'threshold': threshold}

# ============================================================
# run_model
# ============================================================
def run_model(model_dict, record, data_folder, verbose=False):
    try:
        clf         = model_dict['model']
        features    = model_dict['features']
        impute_vals = model_dict['impute_vals']
        threshold   = model_dict.get('threshold', THRESHOLD)

        # record = {'BidsFolder':..., 'SiteID':..., 'SessionID':...}
        bids_folder = record.get('BidsFolder', '')
        site_id     = record.get('SiteID', '')
        session_id  = record.get('SessionID', '')

        annot_path = find_annot_file_by_bids(bids_folder, session_id, site_id, data_folder)
        caisr_feat = extract_caisr_features(annot_path) if annot_path else {}

        phys_path = find_phys_file_by_bids(bids_folder, session_id, site_id, data_folder)
        eeg_feat = extract_eeg_relative_sparse(phys_path) if phys_path else {}

        feat = {**caisr_feat, **eeg_feat}

        feat_vec = []
        for col in features:
            v = feat.get(col, np.nan)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = impute_vals.get(col, 0.0)
            feat_vec.append(float(v))

        prob   = float(clf.predict_proba([feat_vec])[0][1])
        binary = int(prob >= threshold)
        return binary, prob

    except Exception as e:
        if verbose:
            print(f'Error: {e}')
        return 0, 0.5
