#!/usr/bin/env python

# ============================================================
# PhysioNet Challenge 2026 | Team: Momochi-SleepAI
#
# Model    : RandomForestClassifier (hybrid feature + tuned)
# Features : 31개 = 15개 absolute + 16개 relative
# Validation: Site-based leave-one-out CV
#   - S0001→I0006 AUROC: 0.6744
#   - S0001→I0002 AUROC: 0.6605
#   - worst-case:  0.6605 (이전 0.5031 대비 대폭 개선)
#
# 주요 변경 이력:
#   v1: proxy feature (착시 0.82)
#   v2: CAISR 14개 (0.67/0.49)
#   v3: arousal/resp 추가 21개 (0.68/0.58)
#   v4: RF 하이퍼파라미터 튜닝 (0.72/0.67)
#   v5: hybrid feature 31개 (0.674/0.661) ← 현재
#
# 핵심 인사이트:
#   - absolute feature → I0006 강함 (site-specific severity)
#   - relative feature → I0002 강함 (site-invariant structure)
#   - hybrid → worst-case 최적화
#   - 진짜 robust signal: ar_cluster, rem_ar_ratio, entropy_std
# ============================================================

import joblib
import json
import numpy as np
import os
import pandas as pd
import warnings
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import pyedflib

warnings.filterwarnings('ignore')

# ============================================================
# 확정 feature 목록 (순서 고정)
# ============================================================
FEATURE_COLS = [
    # absolute (15개)
    'tst_min', 'sleep_eff', 'n3_ratio', 'n2_ratio', 'rem_ratio', 'wake_ratio',
    'n3_ratio_2nd', 'rem_ratio_1st', 'transition_rate',
    'prob_n3_mean', 'prob_r_mean', 'prob_w_mean', 'stage_entropy_std',
    'arousal_index', 'ahi',
    # relative (16개)
    'wake_sleep_ratio', 'n3_rem_ratio', 'n3_n2_ratio', 'rem_nrem_ratio', 'nrem_rem_ratio',
    'n3_shift', 'rem_shift', 'wake_shift', 'n3_front_loading', 'rem_back_loading',
    'prob_w_minus_n3', 'prob_w_over_r',
    'rem_arousal_ratio', 'arousal_cluster_ratio',
    'ca_ratio', 'rem_nrem_resp_ratio',
]

RF_PARAMS = {
    'n_estimators':     300,
    'max_depth':        10,
    'min_samples_leaf': 8,
    'max_features':     0.4,
    'class_weight':     'balanced',
    'random_state':     42,
    'n_jobs':           -1,
}

# ============================================================
# 헬퍼 함수
# ============================================================
def count_events(signal, value):
    is_event = (signal == value)
    return int(np.sum(np.diff(is_event.astype(int)) == 1))

def get_bout_lengths(stage_arr, stage_val):
    is_s = (stage_arr == stage_val).astype(int)
    diff = np.diff(np.concatenate([[0], is_s, [0]]))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    return (ends - starts) * 0.5

def resp_burden_by_stage(stage_arr, resp_sig, stage_val):
    resp_in_stage = []
    for ep in range(len(stage_arr)):
        if stage_arr[ep] == stage_val:
            r_start = ep * 30
            r_end   = min(r_start + 30, len(resp_sig))
            resp_in_stage.extend(resp_sig[r_start:r_end])
    resp_in_stage = np.array(resp_in_stage)
    if len(resp_in_stage) == 0:
        return np.nan
    n_events = (count_events(resp_in_stage, 1) +
                count_events(resp_in_stage, 2) +
                count_events(resp_in_stage, 4))
    stage_hours = len(resp_in_stage) / 3600
    return float(n_events / stage_hours) if stage_hours > 0 else np.nan

def find_annot_file(patient_id, data_folder):
    annot_base = os.path.join(data_folder, 'algorithmic_annotations')
    if not os.path.exists(annot_base):
        return None
    for site_folder in os.listdir(annot_base):
        site_path = os.path.join(annot_base, site_folder)
        if not os.path.isdir(site_path):
            continue
        for fname in os.listdir(site_path):
            if str(patient_id) in fname and fname.endswith('.edf'):
                return os.path.join(site_path, fname)
    return None

# ============================================================
# Feature 추출 (absolute + relative hybrid)
# ============================================================
def extract_hybrid_features(edf_path):
    try:
        with pyedflib.EdfReader(edf_path) as f:
            labels = f.getSignalLabels()
            lbl = {l: i for i, l in enumerate(labels)}
            stage   = f.readSignal(lbl['stage_caisr'])
            prob_n3 = f.readSignal(lbl['caisr_prob_n3'])
            prob_n2 = f.readSignal(lbl['caisr_prob_n2'])
            prob_n1 = f.readSignal(lbl['caisr_prob_n1'])
            prob_r  = f.readSignal(lbl['caisr_prob_r'])
            prob_w  = f.readSignal(lbl['caisr_prob_w'])
            arousal = f.readSignal(lbl['arousal_caisr']) if 'arousal_caisr' in lbl else None
            resp    = f.readSignal(lbl['resp_caisr'])    if 'resp_caisr'    in lbl else None

        valid = stage != 9
        sv    = stage[valid]
        total = len(sv)
        tst   = int(np.sum(sv != 5))
        tst_h = tst * 30 / 3600
        if total == 0 or tst_h == 0:
            return {}

        half = total // 2
        eps  = 1e-9

        n3        = float(np.mean(sv == 1))
        n2        = float(np.mean(sv == 2))
        rem       = float(np.mean(sv == 4))
        wake      = float(np.mean(sv == 5))
        sleep_eff = tst / total

        # entropy (한 번만 계산)
        probs_stack = np.stack([prob_n3, prob_n2, prob_n1, prob_r, prob_w], axis=1)
        probs_clip  = np.clip(probs_stack, 1e-9, 1)
        entropy     = -np.sum(probs_clip * np.log(probs_clip), axis=1)

        # resp events (한 번만 계산)
        if resp is not None:
            n_oa = count_events(resp, 1)
            n_ca = count_events(resp, 2)
            n_hy = count_events(resp, 4)
            total_resp = n_oa + n_ca + n_hy

        feat = {}

        # ── Absolute features (15개) ──────────────────────
        feat['tst_min']           = tst * 0.5
        feat['sleep_eff']         = sleep_eff
        feat['n3_ratio']          = n3
        feat['n2_ratio']          = n2
        feat['rem_ratio']         = rem
        feat['wake_ratio']        = wake
        feat['n3_ratio_2nd']      = float(np.mean(sv[half:] == 1))
        feat['rem_ratio_1st']     = float(np.mean(sv[:half] == 4))
        feat['transition_rate']   = float(np.sum(np.diff(sv) != 0) / total)
        feat['prob_n3_mean']      = float(np.mean(prob_n3))
        feat['prob_r_mean']       = float(np.mean(prob_r))
        feat['prob_w_mean']       = float(np.mean(prob_w))
        feat['stage_entropy_std'] = float(np.std(entropy))
        feat['arousal_index']     = (
            count_events(arousal, 1) / tst_h if arousal is not None else np.nan
        )
        feat['ahi'] = (
            (n_oa + n_ca + n_hy) / tst_h if resp is not None else np.nan
        )

        # ── Relative features (16개) ──────────────────────
        feat['wake_sleep_ratio']  = wake / (sleep_eff + eps)
        feat['n3_rem_ratio']      = n3   / (rem + eps)
        feat['n3_n2_ratio']       = n3   / (n2 + eps)
        feat['rem_nrem_ratio']    = rem  / (n2 + n3 + eps)
        feat['nrem_rem_ratio']    = (n2 + n3) / (rem + eps)

        n3_1st   = float(np.mean(sv[:half] == 1))
        n3_2nd   = float(np.mean(sv[half:] == 1))
        rem_1st  = float(np.mean(sv[:half] == 4))
        rem_2nd  = float(np.mean(sv[half:] == 4))
        wake_1st = float(np.mean(sv[:half] == 5))
        wake_2nd = float(np.mean(sv[half:] == 5))

        feat['n3_shift']          = n3_2nd  - n3_1st
        feat['rem_shift']         = rem_2nd - rem_1st
        feat['wake_shift']        = wake_2nd - wake_1st
        feat['n3_front_loading']  = n3_1st  / (n3_2nd  + eps)
        feat['rem_back_loading']  = rem_2nd / (rem_1st + eps)
        feat['prob_w_minus_n3']   = float(np.mean(prob_w) - np.mean(prob_n3))
        feat['prob_w_over_r']     = float(np.mean(prob_w) / (np.mean(prob_r) + eps))

        if arousal is not None:
            ar_onsets = np.where(np.diff((arousal==1).astype(int)) == 1)[0]
            ar_epochs = (ar_onsets / 60).astype(int)
            ar_epochs = ar_epochs[ar_epochs < total]
            n_rem_ar  = int(np.sum(sv[ar_epochs] == 4))
            n_nrem_ar = int(np.sum(np.isin(sv[ar_epochs], [1,2,3])))
            feat['rem_arousal_ratio']     = float(n_rem_ar / (n_nrem_ar + n_rem_ar + eps))
            feat['arousal_cluster_ratio'] = (
                float(np.mean(np.diff(ar_onsets) < 120))
                if len(ar_onsets) > 2 else np.nan
            )
        else:
            feat['rem_arousal_ratio']     = np.nan
            feat['arousal_cluster_ratio'] = np.nan

        if resp is not None:
            feat['ca_ratio'] = float(n_ca / (total_resp + eps))
            rem_resp  = resp_burden_by_stage(sv, resp, 4)
            nrem_resp = resp_burden_by_stage(sv, resp, 2)
            feat['rem_nrem_resp_ratio'] = (
                float(rem_resp / (nrem_resp + eps))
                if rem_resp is not None and not np.isnan(rem_resp) else np.nan
            )
        else:
            feat['ca_ratio']            = np.nan
            feat['rem_nrem_resp_ratio'] = np.nan

        return feat

    except Exception:
        return {}

# ============================================================
# train_model
# ============================================================
def train_model(data_folder, model_folder, verbose=False):
    os.makedirs(model_folder, exist_ok=True)

    if verbose:
        print('📂 데이터 로드 중...')

    df = pd.read_csv(os.path.join(data_folder, 'demographics.csv'))

    if verbose:
        print(f'✅ 환자 수: {len(df)}명')
        print('🔄 Hybrid feature 추출 중...')

    records = []
    for _, row in df.iterrows():
        pid      = row.get('BDSPPatientID', '')
        feat     = {'BDSPPatientID': pid}
        edf_path = find_annot_file(pid, data_folder)
        if edf_path:
            feat.update(extract_hybrid_features(edf_path))
        records.append(feat)

    df_feat = pd.DataFrame(records)
    df_all  = df.merge(df_feat, on='BDSPPatientID', how='left')

    if 'Cognitive_Impairment' in df_all.columns:
        y = df_all['Cognitive_Impairment'].astype(int)
    else:
        raise ValueError('라벨 컬럼을 찾을 수 없습니다.')

    X = df_all[FEATURE_COLS].copy()

    if verbose:
        print(f'🤖 모델 학습 중... (feature: {len(FEATURE_COLS)}개, 샘플: {len(X)}명)')

    imputer = SimpleImputer(strategy='median')
    X_imp   = imputer.fit_transform(X)

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_imp, y)

    joblib.dump(model,   os.path.join(model_folder, 'momochi_model.pkl'))
    joblib.dump(imputer, os.path.join(model_folder, 'imputer.pkl'))
    with open(os.path.join(model_folder, 'features.json'), 'w') as f:
        json.dump(FEATURE_COLS, f, indent=2)

    if verbose:
        print(f'💾 모델 저장 완료 → {model_folder}')
        print('✅ 훈련 완료!')

# ============================================================
# load_model
# ============================================================
def load_model(model_folder, verbose=False):
    model   = joblib.load(os.path.join(model_folder, 'momochi_model.pkl'))
    imputer = joblib.load(os.path.join(model_folder, 'imputer.pkl'))
    with open(os.path.join(model_folder, 'features.json'), 'r') as f:
        features = json.load(f)
    if verbose:
        print('✅ 모델 로드 완료!')
    return {'model': model, 'imputer': imputer, 'features': features}

# ============================================================
# run_model
# ============================================================
def run_model(model_dict, record, data_folder, verbose=False):
    try:
        clf      = model_dict['model']
        imputer  = model_dict['imputer']
        features = model_dict['features']

        patient_id = record.get('BDSPPatientID', '')
        edf_path   = find_annot_file(patient_id, data_folder)
        feat = {}
        if edf_path:
            feat = extract_hybrid_features(edf_path)

        row    = {col: feat.get(col, np.nan) for col in features}
        X      = pd.DataFrame([row])[features]
        X_imp  = imputer.transform(X)
        prob   = float(clf.predict_proba(X_imp)[0][1])
        binary = int(prob >= 0.5)
        return binary, prob

    except Exception as e:
        if verbose:
            print(f'⚠️ 예측 실패: {e}')
        return 0, 0.5
