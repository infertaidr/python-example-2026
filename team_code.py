#!/usr/bin/env python

# ============================================================
# PhysioNet Challenge 2026 | Team: Momochi-SleepAI
#
# Model    : RandomForestClassifier (hyperparameter tuned)
# Features : 21개 CAISR-based sleep structure features
#           + 5개 인구통계 features (Age, BMI, sex, race, ethnicity)
#           = 총 26개 features
# Validation: 5-Fold CV AUROC: 0.7871 ± 0.0206
#
# v8:
#   - Age, BMI merge 충돌 수정 (demographics.csv 원본 사용)
#   - extract_demo_features에서 Age, BMI 제거
# ============================================================

import glob
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
    # CAISR 수면 피처 21개
    'sleep_eff',
    'n2_ratio',
    'wake_ratio',
    'waso_min',
    'wake_bout_max',
    'transition_entropy',
    'n3_ratio_2nd',
    'rem_ratio_1st',
    'stage_confidence_min',
    'stage_entropy_std',
    'prob_r_mean',
    'prob_w_mean',
    'arousal_index',
    'ahi',
    'arousal_burstiness',
    'arousal_cluster_ratio',
    'rem_arousal_ratio',
    'ahi_rem',
    'ahi_nrem',
    'ahi_n3',
    'ahi_rem_nrem_ratio',
    # 인구통계 피처 5개
    'Age',
    'BMI',
    'sex_num',
    'race_num',
    'ethnicity_num',
]

# CAISR annotation EDF에서 반드시 있어야 하는 필수 채널
REQUIRED_CHANNELS = {'stage_caisr', 'caisr_prob_n3', 'caisr_prob_n2',
                     'caisr_prob_n1', 'caisr_prob_r', 'caisr_prob_w'}

# ============================================================
# RF 하이퍼파라미터
# ============================================================
RF_PARAMS = {
    'n_estimators':     300,
    'max_depth':        10,
    'min_samples_leaf': 4,
    'max_features':     0.5,
    'class_weight':     'balanced',
    'random_state':     42,
    'n_jobs':           -1,
}

# ============================================================
# 경로 조립 + fallback
# ============================================================
def build_annot_path(data_folder, site_id, bids_folder, session_id):
    fname   = f"{bids_folder}-ses{session_id}.edf"
    primary = os.path.join(data_folder, 'algorithmic_annotations',
                           str(site_id), fname)
    if os.path.exists(primary):
        return primary

    pattern1 = os.path.join(data_folder, 'algorithmic_annotations',
                             str(site_id), f'*{bids_folder}*.edf')
    hits = glob.glob(pattern1)
    if hits:
        return hits[0]

    pattern2 = os.path.join(data_folder, 'algorithmic_annotations',
                             '**', f'*{bids_folder}*.edf')
    hits = glob.glob(pattern2, recursive=True)
    if hits:
        return hits[0]

    return primary


# ============================================================
# 헬퍼 함수
# ============================================================
def count_events(signal, value):
    is_event = (signal == value)
    onsets = np.sum(np.diff(is_event.astype(int)) == 1)
    return int(onsets)

def get_bout_lengths(stage_arr, stage_val):
    is_s  = (stage_arr == stage_val).astype(int)
    diff  = np.diff(np.concatenate([[0], is_s, [0]]))
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
    n_events    = (count_events(resp_in_stage, 1) +
                   count_events(resp_in_stage, 2) +
                   count_events(resp_in_stage, 4))
    stage_hours = len(resp_in_stage) / 3600
    return float(n_events / stage_hours) if stage_hours > 0 else np.nan


# ============================================================
# 인구통계 피처 추출 (Age, BMI 제외 - demographics.csv 원본 사용)
# ============================================================
def extract_demo_features(row):
    race_map = {'White':0,'Black':1,'Asian':2,'Hispanic':3,'Others':4,'Unavailable':4}
    eth_map  = {'Not Hispanic':0,'Hispanic':1,'Unknown':2}
    return {
        'sex_num':       1 if str(row.get('Sex', '')).strip() == 'Male' else 0,
        'race_num':      race_map.get(str(row.get('Race', 'Unavailable')).strip(), 4),
        'ethnicity_num': eth_map.get(str(row.get('Ethnicity', 'Unknown')).strip(), 2),
    }


# ============================================================
# CAISR feature 추출
# ============================================================
def extract_caisr_features(edf_path):
    try:
        with pyedflib.EdfReader(edf_path) as f:
            raw_labels   = f.getSignalLabels()
            label_to_idx = {l.lower().strip(): i for i, l in enumerate(raw_labels)}

            missing = REQUIRED_CHANNELS - set(label_to_idx.keys())
            if missing:
                return {}

            stage   = f.readSignal(label_to_idx['stage_caisr'])
            prob_n3 = f.readSignal(label_to_idx['caisr_prob_n3'])
            prob_n2 = f.readSignal(label_to_idx['caisr_prob_n2'])
            prob_n1 = f.readSignal(label_to_idx['caisr_prob_n1'])
            prob_r  = f.readSignal(label_to_idx['caisr_prob_r'])
            prob_w  = f.readSignal(label_to_idx['caisr_prob_w'])

            arousal = f.readSignal(label_to_idx['arousal_caisr']) \
                      if 'arousal_caisr' in label_to_idx else None
            resp    = f.readSignal(label_to_idx['resp_caisr']) \
                      if 'resp_caisr' in label_to_idx else None

        valid        = stage != 9
        stage_v      = stage[valid]
        total_epochs = len(stage_v)
        tst_epochs   = np.sum(stage_v != 5)
        tst_hours    = tst_epochs * 30 / 3600

        if total_epochs == 0 or tst_hours == 0:
            return {}

        feat = {}

        feat['sleep_eff']  = float(tst_epochs / total_epochs)
        feat['n2_ratio']   = float(np.mean(stage_v == 2))
        feat['wake_ratio'] = float(np.mean(stage_v == 5))
        feat['waso_min']   = float(np.sum(stage_v == 5) * 0.5)

        wake_bouts = get_bout_lengths(stage_v, 5)
        feat['wake_bout_max'] = float(np.max(wake_bouts)) if len(wake_bouts) > 0 else 0.0

        transitions  = list(zip(stage_v[:-1], stage_v[1:]))
        trans_counts = Counter(transitions)
        total_trans  = sum(trans_counts.values())
        trans_probs  = np.clip(
            np.array([v / total_trans for v in trans_counts.values()]), 1e-9, 1)
        feat['transition_entropy'] = float(-np.sum(trans_probs * np.log(trans_probs)))

        half = total_epochs // 2
        feat['n3_ratio_2nd']  = float(np.mean(stage_v[half:] == 1))
        feat['rem_ratio_1st'] = float(np.mean(stage_v[:half] == 4))

        probs      = np.stack([prob_n3, prob_n2, prob_n1, prob_r, prob_w], axis=1)
        prob_max   = np.max(probs, axis=1)
        feat['stage_confidence_min'] = float(np.min(prob_max))
        probs_clip = np.clip(probs, 1e-9, 1)
        entropy    = -np.sum(probs_clip * np.log(probs_clip), axis=1)
        feat['stage_entropy_std'] = float(np.std(entropy))
        feat['prob_r_mean'] = float(np.mean(prob_r))
        feat['prob_w_mean'] = float(np.mean(prob_w))

        if arousal is not None:
            n_ar = count_events(arousal, 1)
            feat['arousal_index'] = float(n_ar / tst_hours)
            ar_onsets = np.where(np.diff((arousal == 1).astype(int)) == 1)[0]
            if len(ar_onsets) > 2:
                intervals = np.diff(ar_onsets)
                mu, sigma = np.mean(intervals), np.std(intervals)
                feat['arousal_burstiness']    = float((sigma - mu) / (sigma + mu + 1e-9))
                feat['arousal_cluster_ratio'] = float(np.mean(intervals < 120))
                ar_epochs = (ar_onsets / 60).astype(int)
                ar_epochs = ar_epochs[ar_epochs < total_epochs]
                n_nrem_ar = int(np.sum(np.isin(stage_v[ar_epochs], [1, 2, 3])))
                n_rem_ar  = int(np.sum(stage_v[ar_epochs] == 4))
                feat['rem_arousal_ratio'] = float(
                    n_rem_ar / (n_nrem_ar + n_rem_ar + 1e-9))
            else:
                feat['arousal_burstiness']    = np.nan
                feat['arousal_cluster_ratio'] = np.nan
                feat['rem_arousal_ratio']     = np.nan
        else:
            feat['arousal_index']         = np.nan
            feat['arousal_burstiness']    = np.nan
            feat['arousal_cluster_ratio'] = np.nan
            feat['rem_arousal_ratio']     = np.nan

        if resp is not None:
            n_oa = count_events(resp, 1)
            n_ca = count_events(resp, 2)
            n_hy = count_events(resp, 4)
            feat['ahi']      = float((n_oa + n_ca + n_hy) / tst_hours)
            feat['ahi_rem']  = resp_burden_by_stage(stage_v, resp, 4)
            feat['ahi_nrem'] = resp_burden_by_stage(stage_v, resp, 2)
            feat['ahi_n3']   = resp_burden_by_stage(stage_v, resp, 1)
            ahi_rem  = feat['ahi_rem']
            ahi_nrem = feat['ahi_nrem']
            if ahi_rem is not None and not np.isnan(ahi_rem):
                feat['ahi_rem_nrem_ratio'] = float(ahi_rem / (ahi_nrem + 0.01))
            else:
                feat['ahi_rem_nrem_ratio'] = np.nan
        else:
            feat['ahi']                = np.nan
            feat['ahi_rem']            = np.nan
            feat['ahi_nrem']           = np.nan
            feat['ahi_n3']             = np.nan
            feat['ahi_rem_nrem_ratio'] = np.nan

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

    demo_path = os.path.join(data_folder, 'demographics.csv')
    df = pd.read_csv(demo_path)

    if verbose:
        print(f'✅ 환자 수: {len(df)}명')
        print('🔄 CAISR feature 추출 중...')

    records = []
    n_found = 0
    for _, row in df.iterrows():
        bids_folder = str(row.get('BidsFolder', ''))
        site_id     = str(row.get('SiteID', ''))
        session_id  = row.get('SessionID', '')

        feat = {'BidsFolder': bids_folder}

        # 인구통계 피처 (sex, race, ethnicity만 — Age/BMI는 df 원본 사용)
        feat.update(extract_demo_features(row))

        # CAISR 피처
        annot_path = build_annot_path(data_folder, site_id, bids_folder, session_id)
        if os.path.exists(annot_path):
            n_found += 1
            feat.update(extract_caisr_features(annot_path))
        elif verbose:
            print(f'  ⚠️ annotation 없음: {annot_path}')

        records.append(feat)

    if verbose:
        print(f'  📁 annotation 찾은 환자: {n_found}/{len(df)}명')

    df_feat = pd.DataFrame(records)

    # BidsFolder 기준으로 merge (Age, BMI는 df 원본에서 가져옴)
    df_all = df.merge(df_feat, on='BidsFolder', how='left')

    # Age, BMI 결측치 처리
    df_all['Age'] = df_all['Age'].fillna(df_all['Age'].median())
    df_all['BMI'] = df_all['BMI'].fillna(df_all['BMI'].median())

    if 'Cognitive_Impairment' not in df_all.columns:
        raise ValueError('라벨 컬럼(Cognitive_Impairment)을 찾을 수 없습니다.')

    y = df_all['Cognitive_Impairment'].apply(
        lambda v: 1 if str(v).strip().upper() == 'TRUE' or v is True else 0
    )

    X = df_all[FEATURE_COLS].copy()

    if verbose:
        nan_rate = X.isna().mean().mean()
        print(f'  📊 feature NaN 비율: {nan_rate:.1%}')
        print(f'  🏷️  라벨 분포: {dict(y.value_counts())}')
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

        bids_folder = str(record.get('BidsFolder', ''))
        site_id     = str(record.get('SiteID', ''))
        session_id  = record.get('SessionID', '')

        # CAISR 피처
        annot_path = build_annot_path(data_folder, site_id, bids_folder, session_id)
        feat = {}
        if os.path.exists(annot_path):
            feat = extract_caisr_features(annot_path)
        elif verbose:
            print(f'  ⚠️ annotation 없음: {annot_path} → median imputation')

        # 인구통계 피처 (sex, race, ethnicity)
        feat.update(extract_demo_features(record))

        # Age, BMI는 record에서 직접
        try:
            feat['Age'] = float(record.get('Age', np.nan))
        except Exception:
            feat['Age'] = np.nan
        try:
            feat['BMI'] = float(record.get('BMI', np.nan))
        except Exception:
            feat['BMI'] = np.nan

        row = {col: feat.get(col, np.nan) for col in features}
        X   = pd.DataFrame([row])[features]

        X_imp  = imputer.transform(X)
        prob   = float(clf.predict_proba(X_imp)[0][1])
        binary = int(prob >= 0.5)

        return binary, prob

    except Exception as e:
        if verbose:
            print(f'  ⚠️ 예측 실패: {e}')
        return 0, 0.5
