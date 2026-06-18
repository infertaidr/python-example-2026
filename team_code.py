#!/usr/bin/env python

# ============================================================
# PhysioNet Challenge 2026 | Team: Momochi-SleepAI
#
# Model    : RandomForestClassifier
# Features : 24개 (Demographics + CAISR sleep/event features)
#            sex_male 제외 (site별 방향성 불일치로 제거)
# Strategy : Site-consistent feature selection
# CV       : S0001 internal 5-fold + I0006/I0002 external
# External : I0006 AUROC=0.7815 / I0002 AUROC=0.7155
#            Worst-site AUROC=0.7155
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
    'age', 'age_group_65plus', 'ethnicity_hispanic',
    'tst_min', 'tib_min', 'sleep_eff', 'waso_min',
    'sleep_ineff', 'waso_pct',
    'pct_n3', 'pct_n2', 'pct_rem', 'pct_nrem',
    'stage_entropy', 'rem_latency_min', 'n3_front_loading_ratio',
    'n_ma', 'n_rera',
    'n_arousals', 'arousal_rem_pct',
    'n_plm', 'n_ilm', 'plm_index', 'event_burden'
]

RF_PARAMS = {
    'n_estimators':     200,
    'max_depth':        6,
    'min_samples_leaf': 5,
    'class_weight':     'balanced',
    'random_state':     42,
    'n_jobs':           -1,
}

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

def find_annot_file(patient_id, data_folder):
    annot_base = os.path.join(data_folder, 'algorithmic_annotations')
    if not os.path.exists(annot_base):
        return None
    pid_str = str(patient_id)
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
            half    = n_ep // 2
            f['tst_min']         = round(n_sleep * 30/60, 2)
            f['tib_min']         = round(n_ep * 30/60, 2)
            f['sleep_eff']       = round(n_sleep/n_ep*100, 2)
            f['waso_min']        = round(n_wake * 30/60, 2)
            f['sleep_onset_min'] = round(
                np.argmax(valid!=5)*30/60 if n_sleep>0 else n_ep*30/60, 2)
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
                f['n3_front_loading_ratio'] = round(
                    (np.sum(valid[:half]==1)+1e-4)/
                    (np.sum(valid[half:]==1)+1e-4), 4)
                f['n3_rem_ratio'] = round(
                    float(np.sum(valid==1))/(float(np.sum(valid==4))+1e-4), 4)
            else:
                for k in ['pct_n3','pct_n2','pct_n1','pct_rem','pct_nrem',
                          'stage_transition_rate','stage_entropy','rem_latency_min',
                          'n3_front_loading_ratio','n3_rem_ratio']:
                    f[k] = np.nan
        else:
            for k in ['tst_min','tib_min','sleep_eff','waso_min','sleep_onset_min',
                      'sleep_ineff','waso_pct','pct_n3','pct_n2','pct_n1','pct_rem',
                      'pct_nrem','stage_transition_rate','stage_entropy',
                      'rem_latency_min','n3_front_loading_ratio','n3_rem_ratio']:
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

    # event_burden 맨 마지막에 계산
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
    from sklearn.ensemble import RandomForestClassifier

    os.makedirs(model_folder, exist_ok=True)
    if verbose:
        print('Loading demographics...')

    demo = pd.read_csv(os.path.join(data_folder, 'demographics.csv'))
    if demo['Cognitive_Impairment'].dtype == object:
        demo['Cognitive_Impairment'] = demo['Cognitive_Impairment'].map(
            {'TRUE':True,'True':True,'true':True,
             'FALSE':False,'False':False,'false':False})

    demo['age']                = demo['Age'].apply(lambda x: safe_float(x))
    demo['age_group_65plus']   = (demo['age'] >= 65).astype(int)
    demo['ethnicity_hispanic'] = (demo['Ethnicity'].str.strip().str.lower()=='hispanic').astype(int)

    if verbose:
        print(f'Extracting CAISR features for {len(demo)} patients...')

    records = []
    for i, (_, row) in enumerate(demo.iterrows()):
        edf_path = find_annot_file(row['BDSPPatientID'], data_folder)
        feats    = extract_caisr_features(edf_path)
        feats['BDSPPatientID'] = row['BDSPPatientID']
        records.append(feats)
        if verbose and (i+1) % 500 == 0:
            print(f'  [{i+1}/{len(demo)}]')

    caisr_df = pd.DataFrame(records)
    df = demo[['BDSPPatientID','age','age_group_65plus',
               'ethnicity_hispanic','Cognitive_Impairment']].merge(
        caisr_df, on='BDSPPatientID', how='left')

    # Winsorize
    for col, q in [('ahi',0.99),('rdi',0.99),('n_ilm',0.99),
                   ('plm_index',0.99),('n_plm',0.99),('event_burden',0.99)]:
        if col in df.columns:
            cap = df[col].quantile(q)
            df[col] = df[col].clip(upper=cap)

    # Training data 기준 median imputation
    impute_vals = {}
    for col in FEATURE_COLS:
        if col in df.columns:
            med = df[col].median()
            impute_vals[col] = float(med)
            df[col] = df[col].fillna(med)

    X = df[FEATURE_COLS].values.astype(float)
    y = df['Cognitive_Impairment'].astype(int).values

    if verbose:
        print(f'Training RF... (n={len(X)}, CI={y.sum()})')

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X, y)

    joblib.dump(model, os.path.join(model_folder, 'model.pkl'))
    with open(os.path.join(model_folder, 'features.json'), 'w') as fj:
        json.dump(FEATURE_COLS, fj)
    with open(os.path.join(model_folder, 'impute_vals.json'), 'w') as fj:
        json.dump(impute_vals, fj)

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
    if verbose:
        print('Model loaded!')
    return {'model': model, 'features': features, 'impute_vals': impute_vals}

# ============================================================
# run_model
# ============================================================
def run_model(model_dict, record, data_folder, verbose=False):
    try:
        clf         = model_dict['model']
        features    = model_dict['features']
        impute_vals = model_dict['impute_vals']

        patient_id = record.get('BDSPPatientID', '')
        edf_path   = find_annot_file(patient_id, data_folder)
        feat       = extract_caisr_features(edf_path) if edf_path else {}

        # Demographics (safe_float 사용)
        age = safe_float(record.get('Age', np.nan))
        feat['age']                = age
        feat['age_group_65plus']   = 1 if (not np.isnan(age) and age >= 65) else 0
        feat['ethnicity_hispanic'] = 1 if str(record.get('Ethnicity','')).strip().lower()=='hispanic' else 0

        # Feature vector with imputation
        feat_vec = []
        for col in features:
            v = feat.get(col, np.nan)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = impute_vals.get(col, 0.0)
            feat_vec.append(float(v))

        prob   = float(clf.predict_proba([feat_vec])[0][1])
        binary = int(prob >= 0.5)
        return binary, prob

    except Exception as e:
        if verbose:
            print(f'Error: {e}')
        return 0, 0.5
