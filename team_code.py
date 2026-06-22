#!/usr/bin/env python

# ============================================================
# PhysioNet Challenge 2026 | Team: Momochi-SleepAI
#
# Model    : LogisticRegression + StandardScaler
# Features : 12개 (CAISR sleep/event features, age 제외)
# Strategy : Age-conditioned AUROC 최적화
#            Greedy forward selection (age-conditioned worst 기준)
# Threshold: 0.60 (prevalence-based reward 최적화)
# External development (S0001 train → I0006/I0002):
#   I0006 age-cond=0.7182 / I0002 age-cond=0.7224
#   Worst age-conditioned AUROC = 0.7182
#
# v1.1 FIX: run_model()이 받는 record는 BidsFolder/SiteID/SessionID만
#           가지고 있고 BDSPPatientID는 없음을 확인. annotation file
#           매칭을 BidsFolder+SessionID 기반 정확 매칭으로 수정.
#           (v1에서 patient_id가 빈 문자열로 처리되어 모든 환자가
#            동일 annotation을 사용하는 버그가 있었음 -> age-cond AUROC 0.500)
# ============================================================

import joblib
import json
import numpy as np
import os
import warnings
import mne

mne.set_log_level('ERROR')
warnings.filterwarnings('ignore')

# ============================================================
# 확정 feature 목록 (순서 고정 - 변경 금지)
# ============================================================
FEATURE_COLS = [
    'n_ma', 'n_arousals', 'pct_rem', 'pct_nrem', 'pct_n2',
    'n_ilm', 'stage_entropy', 'n_plm', 'plm_index',
    'stage_transition_rate', 'waso_min', 'sleep_ineff'
]

THRESHOLD = 0.60

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

# ============================================================
# Annotation file 탐색 (BidsFolder + SessionID + SiteID 기반)
# ============================================================
def find_annot_file_by_bids(bids_folder, session_id, site_id, data_folder):
    """
    record = {'BidsFolder':..., 'SiteID':..., 'SessionID':...} 형태에서
    정확한 annotation EDF 파일을 찾는다.
    파일명 패턴: {BidsFolder}_ses-{SessionID}_caisr_annotations.edf
    """
    annot_base = os.path.join(data_folder, 'algorithmic_annotations')
    if not os.path.isdir(annot_base):
        return None

    bids_folder = str(bids_folder).strip() if bids_folder is not None else ''
    session_id  = normalize_session_id(session_id)
    site_id     = str(site_id).strip() if site_id is not None else ''

    if not bids_folder or not session_id:
        return None

    target_name = f"{bids_folder}_ses-{session_id}_caisr_annotations.edf"

    # 1) SiteID 폴더 직접 확인 (가장 빠르고 안전)
    if site_id:
        site_path = os.path.join(annot_base, site_id)
        candidate = os.path.join(site_path, target_name)
        if os.path.exists(candidate):
            return candidate

    # 2) 모든 site 폴더에서 정확 파일명 확인
    for site_folder in os.listdir(annot_base):
        site_path = os.path.join(annot_base, site_folder)
        if not os.path.isdir(site_path):
            continue
        candidate = os.path.join(site_path, target_name)
        if os.path.exists(candidate):
            return candidate

    # 3) fallback: bids_folder와 ses-{session_id}가 둘 다 포함된 caisr 파일
    for root, _, files in os.walk(annot_base):
        for fname in files:
            if (fname.endswith('.edf')
                    and bids_folder in fname
                    and f"ses-{session_id}" in fname
                    and "caisr" in fname.lower()):
                return os.path.join(root, fname)

    return None

def find_annot_file_train(patient_id, data_folder):
    """
    train_model()에서 demographics.csv 전체를 순회할 때 사용.
    demographics.csv에는 BDSPPatientID가 있으므로 substring 매칭이 안전하게 동작.
    """
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

    # Sleep Architecture
    stage_ch = find_channel(raw, ['stage','caisr'], exclude=['prob'])
    if stage_ch:
        epochs = get_stage_epochs(raw, stage_ch)
        valid  = epochs[(epochs >= 1) & (epochs <= 5)]
        n_ep   = len(valid)
        if n_ep > 0:
            n_sleep = int(np.sum(valid != 5))
            n_wake  = int(np.sum(valid == 5))
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
                stage_probs = [np.mean(valid==s) for s in [1,2,3,4,5]]
                f['stage_entropy'] = round(float(
                    -np.sum([p*np.log(p+1e-9) for p in stage_probs if p>0])), 4)
                rem_idx = np.where(valid==4)[0]
                f['rem_latency_min'] = round(
                    rem_idx[0]*30/60 if len(rem_idx)>0 else n_ep*30/60, 2)
            else:
                for k in ['pct_n3','pct_n2','pct_n1','pct_rem','pct_nrem',
                          'stage_transition_rate','stage_entropy','rem_latency_min']:
                    f[k] = np.nan
        else:
            for k in ['tst_min','tib_min','sleep_eff','waso_min','sleep_ineff',
                      'waso_pct','pct_n3','pct_n2','pct_n1','pct_rem','pct_nrem',
                      'stage_transition_rate','stage_entropy','rem_latency_min']:
                f[k] = np.nan

    # Respiratory
    resp_ch = find_channel(raw, ['resp','caisr'], exclude=['prob'])
    if resp_ch and not np.isnan(f.get('tst_min', np.nan)):
        resp_data, _ = get_event_data(raw, resp_ch)
        tst_h = f['tst_min'] / 60
        def cnt(vals):
            return count_events(np.isin(resp_data, vals).astype(int))
        n_oa,n_ca,n_ma,n_hy,n_rera = cnt([1]),cnt([2]),cnt([3]),cnt([4]),cnt([5])
        n_ap = n_oa+n_ca+n_ma
        f['n_oa']=n_oa; f['n_ca']=n_ca; f['n_ma']=n_ma
        f['n_hy']=n_hy; f['n_rera']=n_rera
        f['ahi'] = round((n_ap+n_hy)/tst_h,3)        if tst_h>0 else np.nan
        f['rdi'] = round((n_ap+n_hy+n_rera)/tst_h,3) if tst_h>0 else np.nan
        f['pct_oa'] = round(n_oa/n_ap*100,2) if n_ap>0 else 0.0
        f['pct_ca'] = round(n_ca/n_ap*100,2) if n_ap>0 else 0.0
    else:
        for k in ['n_oa','n_ca','n_ma','n_hy','n_rera','ahi','rdi','pct_oa','pct_ca']:
            f[k] = np.nan

    # Arousal
    arou_ch = find_channel(raw, ['arousal','caisr'], exclude=['prob'])
    if arou_ch and not np.isnan(f.get('tst_min', np.nan)):
        arou_data, arou_sfreq = get_event_data(raw, arou_ch)
        tst_h = f['tst_min'] / 60
        n_ar  = count_events(arou_data)
        f['n_arousals']    = n_ar
        f['arousal_index'] = round(n_ar/tst_h,3) if tst_h>0 else np.nan
        if valid_epochs is not None and int(np.sum(valid_epochs==4)) > 0:
            try:
                rem_mask = (valid_epochs==4)
                spe      = max(int(round(30*arou_sfreq)),1)
                n_ep2    = min(len(rem_mask), len(arou_data)//spe)
                rem_ar   = 0
                for ei in range(n_ep2):
                    seg = arou_data[ei*spe:(ei+1)*spe]
                    if ei < len(rem_mask) and rem_mask[ei]:
                        rem_ar += count_events(seg)
                f['arousal_rem_pct'] = round(rem_ar/n_ar*100,2) if n_ar>0 else 0.0
            except:
                f['arousal_rem_pct'] = np.nan
        else:
            f['arousal_rem_pct'] = 0.0
    else:
        for k in ['n_arousals','arousal_index','arousal_rem_pct']:
            f[k] = np.nan

    # Limb
    limb_ch = find_channel(raw, ['limb','caisr'], exclude=['prob'])
    if limb_ch and not np.isnan(f.get('tst_min', np.nan)):
        limb_data, _ = get_event_data(raw, limb_ch)
        tst_h = f['tst_min'] / 60
        n_plm = count_events((limb_data==2).astype(int))
        n_ilm = count_events((limb_data==1).astype(int))
        f['n_plm']     = n_plm
        f['n_ilm']     = n_ilm
        f['plm_index'] = round(n_plm/tst_h,3) if tst_h>0 else np.nan
    else:
        f['n_plm']=f['n_ilm']=f['plm_index']=np.nan

    # event_burden 맨 마지막
    f['event_burden'] = round(
        (0 if np.isnan(f.get('ahi', np.nan)) else f.get('ahi', 0)) +
        (0 if np.isnan(f.get('arousal_index', np.nan)) else f.get('arousal_index', 0)) +
        (0 if np.isnan(f.get('plm_index', np.nan)) else f.get('plm_index', 0)), 3)

    return f

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
        print(f'Extracting CAISR features for {len(demo)} patients...')

    records = []
    for i, (_, row) in enumerate(demo.iterrows()):
        edf_path = find_annot_file_train(row['BDSPPatientID'], data_folder)
        feats    = extract_caisr_features(edf_path)
        feats['BDSPPatientID'] = row['BDSPPatientID']
        records.append(feats)
        if verbose and (i+1) % 500 == 0:
            print(f'  [{i+1}/{len(demo)}]')

    caisr_df = pd.DataFrame(records)
    df = demo[['BDSPPatientID','Cognitive_Impairment']].merge(
        caisr_df, on='BDSPPatientID', how='left')

    # Winsorize
    for col, q in [('n_ilm',0.99),('n_plm',0.99),('plm_index',0.99)]:
        if col in df.columns:
            cap = df[col].quantile(q)
            df[col] = df[col].clip(upper=cap)

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
        # (BDSPPatientID는 record에 없음! BidsFolder 기반으로 식별해야 함)
        bids_folder = record.get('BidsFolder', '')
        site_id     = record.get('SiteID', '')
        session_id  = record.get('SessionID', '')

        edf_path = find_annot_file_by_bids(bids_folder, session_id, site_id, data_folder)
        feat = extract_caisr_features(edf_path) if edf_path else {}

        # Feature vector with imputation
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
