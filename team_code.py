#!/usr/bin/env python

# ============================================================
# PhysioNet Challenge 2026 | Team: Momochi-SleepAI
#
# Model : VotingClassifier (RandomForest + GradientBoosting)
# Features : 21개 CAISR-based sleep structure features
# + 5개 인구통계 features (Age, BMI, sex, race, ethnicity)
# = 총 26개 features
#
# v9 (improvement):
# - 과적합 방지를 위해 RF 정규화 강화
#   max_depth: 10→5, min_samples_leaf: 4→12, max_features: 0.5→sqrt
# - GradientBoostingClassifier 추가 (max_depth=3, lr=0.05)
# - VotingClassifier(soft) 앙상블로 일반화 성능 향상
# - 피처 클리핑으로 outlier/분포 이동 대응
# ============================================================

import glob
import joblib
import json
import numpy as np
import os
import pandas as pd
import warnings
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
import pyedflib

warnings.filterwarnings('ignore')

FEATURE_COLS = [
    'sleep_eff', 'n2_ratio', 'wake_ratio', 'waso_min', 'wake_bout_max',
    'transition_entropy', 'n3_ratio_2nd', 'rem_ratio_1st',
    'stage_confidence_min', 'stage_entropy_std', 'prob_r_mean',
    'prob_w_mean', 'arousal_index', 'ahi', 'arousal_burstiness',
    'arousal_cluster_ratio', 'rem_arousal_ratio', 'ahi_rem', 'ahi_nrem',
    'ahi_n3', 'ahi_rem_nrem_ratio', 'Age', 'BMI', 'sex_num', 'race_num',
    'ethnicity_num',
]

# 피처별 클리핑 범위 (이상치 대응 — 훈련/테스트 분포 차이 완화)
CLIP_RANGES = {
    'sleep_eff': (0.0, 1.0),
    'n2_ratio': (0.0, 1.0),
    'wake_ratio': (0.0, 1.0),
    'waso_min': (0.0, 600.0),
    'wake_bout_max': (0.0, 300.0),
    'transition_entropy': (0.0, 5.0),
    'n3_ratio_2nd': (0.0, 1.0),
    'rem_ratio_1st': (0.0, 1.0),
    'stage_confidence_min': (0.0, 1.0),
    'stage_entropy_std': (0.0, 2.0),
    'prob_r_mean': (0.0, 1.0),
    'prob_w_mean': (0.0, 1.0),
    'arousal_index': (0.0, 150.0),
    'ahi': (0.0, 150.0),
    'arousal_burstiness': (-1.0, 1.0),
    'arousal_cluster_ratio': (0.0, 1.0),
    'rem_arousal_ratio': (0.0, 1.0),
    'ahi_rem': (0.0, 150.0),
    'ahi_nrem': (0.0, 150.0),
    'ahi_n3': (0.0, 150.0),
    'ahi_rem_nrem_ratio': (0.0, 20.0),
    'Age': (18.0, 100.0),
    'BMI': (10.0, 70.0),
}

REQUIRED_CHANNELS = {'stage_caisr', 'caisr_prob_n3', 'caisr_prob_n2',
                     'caisr_prob_n1', 'caisr_prob_r', 'caisr_prob_w'}

# ── RandomForest: 정규화 강화 ──────────────────────────────
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,           # 10 → 5 (과적합 방지)
    'min_samples_leaf': 12,   # 4  → 12
    'min_samples_split': 20,  # 신규
    'max_features': 'sqrt',   # 0.5 → sqrt
    'max_samples': 0.8,       # 신규: bootstrap 80%
    'class_weight': 'balanced_subsample',  # bootstrap에 적합
    'random_state': 42,
    'n_jobs': -1,
}

# ── GradientBoosting: 얕고 느리게 학습 ───────────────────
GB_PARAMS = {
    'n_estimators': 150,
    'max_depth': 3,
    'min_samples_leaf': 10,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'max_features': 'sqrt',
    'random_state': 42,
}


def build_annot_path(data_folder, site_id, bids_folder, session_id):
    # 실제 파일명 형식: {BidsFolder}_ses-{SessionID}_caisr_annotations.edf
    fname = f"{bids_folder}_ses-{session_id}_caisr_annotations.edf"
    primary = os.path.join(data_folder, 'algorithmic_annotations',
                           str(site_id), fname)
    if os.path.exists(primary):
        return primary

    # fallback: glob으로 탐색
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


def count_events(signal, value):
    is_event = (signal == value)
    onsets = np.sum(np.diff(is_event.astype(int)) == 1)
    return int(onsets)


def get_bout_lengths(stage_arr, stage_val):
    is_s = (stage_arr == stage_val).astype(int)
    diff = np.diff(np.concatenate([[0], is_s, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return (ends - starts) * 0.5


def resp_burden_by_stage(stage_arr, resp_sig, stage_val, fs=1):
    """fs: resp 신호의 실제 샘플링 레이트(Hz). epoch당 샘플 수 = 30 * fs."""
    epoch_samples = int(30 * fs)
    resp_in_stage = []
    for ep in range(len(stage_arr)):
        if stage_arr[ep] == stage_val:
            r_start = ep * epoch_samples
            r_end = min(r_start + epoch_samples, len(resp_sig))
            resp_in_stage.extend(resp_sig[r_start:r_end])
    resp_in_stage = np.array(resp_in_stage)
    if len(resp_in_stage) == 0:
        return np.nan
    n_events = (count_events(resp_in_stage, 1) +
                count_events(resp_in_stage, 2) +
                count_events(resp_in_stage, 4))
    stage_hours = len(resp_in_stage) / (fs * 3600)
    return float(n_events / stage_hours) if stage_hours > 0 else np.nan


def extract_demo_features(row):
    race_map = {'White': 0, 'Black': 1, 'Asian': 2, 'Hispanic': 3,
                'Others': 4, 'Unavailable': 4}
    eth_map = {'Not Hispanic': 0, 'Hispanic': 1, 'Unknown': 2}
    return {
        'sex_num': 1 if str(row.get('Sex', '')).strip() == 'Male' else 0,
        'race_num': race_map.get(str(row.get('Race', 'Unavailable')).strip(), 4),
        'ethnicity_num': eth_map.get(str(row.get('Ethnicity', 'Unknown')).strip(), 2),
    }


def extract_caisr_features(edf_path):
    try:
        with pyedflib.EdfReader(edf_path) as f:
            raw_labels = f.getSignalLabels()
            label_to_idx = {l.lower().strip(): i for i, l in enumerate(raw_labels)}

            missing = REQUIRED_CHANNELS - set(label_to_idx.keys())
            if missing:
                return {}

            stage = f.readSignal(label_to_idx['stage_caisr'])
            prob_n3 = f.readSignal(label_to_idx['caisr_prob_n3'])
            prob_n2 = f.readSignal(label_to_idx['caisr_prob_n2'])
            prob_n1 = f.readSignal(label_to_idx['caisr_prob_n1'])
            prob_r = f.readSignal(label_to_idx['caisr_prob_r'])
            prob_w = f.readSignal(label_to_idx['caisr_prob_w'])

            arousal = f.readSignal(label_to_idx['arousal_caisr']) \
                if 'arousal_caisr' in label_to_idx else None
            resp = None
            resp_fs = 1
            if 'resp_caisr' in label_to_idx:
                resp_idx = label_to_idx['resp_caisr']
                resp = f.readSignal(resp_idx)
                resp_fs = int(f.getSampleFrequency(resp_idx)) or 1

            valid = stage != 9
            stage_v = stage[valid]
            total_epochs = len(stage_v)
            tst_epochs = np.sum(stage_v != 5)
            tst_hours = tst_epochs * 30 / 3600

            if total_epochs == 0 or tst_hours == 0:
                return {}

            feat = {}

            feat['sleep_eff'] = float(tst_epochs / total_epochs)
            feat['n2_ratio'] = float(np.mean(stage_v == 2))
            feat['wake_ratio'] = float(np.mean(stage_v == 5))
            feat['waso_min'] = float(np.sum(stage_v == 5) * 0.5)

            wake_bouts = get_bout_lengths(stage_v, 5)
            feat['wake_bout_max'] = float(np.max(wake_bouts)) if len(wake_bouts) > 0 else 0.0

            transitions = list(zip(stage_v[:-1], stage_v[1:]))
            trans_counts = Counter(transitions)
            total_trans = sum(trans_counts.values())
            trans_probs = np.clip(
                np.array([v / total_trans for v in trans_counts.values()]), 1e-9, 1)
            feat['transition_entropy'] = float(-np.sum(trans_probs * np.log(trans_probs)))

            half = total_epochs // 2
            feat['n3_ratio_2nd'] = float(np.mean(stage_v[half:] == 1))
            feat['rem_ratio_1st'] = float(np.mean(stage_v[:half] == 4))

            probs = np.stack([prob_n3, prob_n2, prob_n1, prob_r, prob_w], axis=1)
            prob_max = np.max(probs, axis=1)
            feat['stage_confidence_min'] = float(np.min(prob_max))
            probs_clip = np.clip(probs, 1e-9, 1)
            entropy = -np.sum(probs_clip * np.log(probs_clip), axis=1)
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
                    feat['arousal_burstiness'] = float((sigma - mu) / (sigma + mu + 1e-9))
                    feat['arousal_cluster_ratio'] = float(np.mean(intervals < 120))
                    ar_epochs = (ar_onsets / 60).astype(int)
                    ar_epochs = ar_epochs[ar_epochs < total_epochs]
                    n_nrem_ar = int(np.sum(np.isin(stage_v[ar_epochs], [1, 2, 3])))
                    n_rem_ar = int(np.sum(stage_v[ar_epochs] == 4))
                    feat['rem_arousal_ratio'] = float(
                        n_rem_ar / (n_nrem_ar + n_rem_ar + 1e-9))
                else:
                    feat['arousal_burstiness'] = np.nan
                    feat['arousal_cluster_ratio'] = np.nan
                    feat['rem_arousal_ratio'] = np.nan
            else:
                feat['arousal_index'] = np.nan
                feat['arousal_burstiness'] = np.nan
                feat['arousal_cluster_ratio'] = np.nan
                feat['rem_arousal_ratio'] = np.nan

            if resp is not None:
                n_oa = count_events(resp, 1)
                n_ca = count_events(resp, 2)
                n_hy = count_events(resp, 4)
                feat['ahi'] = float((n_oa + n_ca + n_hy) / tst_hours)
                feat['ahi_rem'] = resp_burden_by_stage(stage_v, resp, 4, fs=resp_fs)
                feat['ahi_nrem'] = resp_burden_by_stage(stage_v, resp, 2, fs=resp_fs)
                feat['ahi_n3'] = resp_burden_by_stage(stage_v, resp, 1, fs=resp_fs)
                ahi_rem = feat['ahi_rem']
                ahi_nrem = feat['ahi_nrem']
                if ahi_rem is not None and not np.isnan(ahi_rem):
                    feat['ahi_rem_nrem_ratio'] = float(ahi_rem / (ahi_nrem + 0.01))
                else:
                    feat['ahi_rem_nrem_ratio'] = np.nan
            else:
                feat['ahi'] = np.nan
                feat['ahi_rem'] = np.nan
                feat['ahi_nrem'] = np.nan
                feat['ahi_n3'] = np.nan
                feat['ahi_rem_nrem_ratio'] = np.nan

            return feat

    except Exception:
        return {}


def clip_features(X: pd.DataFrame) -> pd.DataFrame:
    """이상치 클리핑 — 훈련/테스트 분포 차이 완화."""
    X = X.copy()
    for col, (lo, hi) in CLIP_RANGES.items():
        if col in X.columns:
            X[col] = X[col].clip(lower=lo, upper=hi)
    return X


def train_model(data_folder, model_folder, verbose=False):
    os.makedirs(model_folder, exist_ok=True)

    if verbose:
        print('데이터 로드 중...')

    demo_path = os.path.join(data_folder, 'demographics.csv')
    df = pd.read_csv(demo_path)

    if verbose:
        print(f'환자 수: {len(df)}명')
        print('CAISR feature 추출 중...')

    records = []
    n_found = 0
    for _, row in df.iterrows():
        bids_folder = str(row.get('BidsFolder', ''))
        site_id = str(row.get('SiteID', ''))
        session_id = row.get('SessionID', '')

        feat = {'BidsFolder': bids_folder}
        feat.update(extract_demo_features(row))

        annot_path = build_annot_path(data_folder, site_id, bids_folder, session_id)
        if os.path.exists(annot_path):
            n_found += 1
            feat.update(extract_caisr_features(annot_path))
        elif verbose:
            print(f'  annotation 없음: {annot_path}')

        records.append(feat)

    if verbose:
        print(f'  annotation 찾은 환자: {n_found}/{len(df)}명')

    df_feat = pd.DataFrame(records)
    df_all = df.merge(df_feat, on='BidsFolder', how='left')

    df_all['Age'] = df_all['Age'].fillna(df_all['Age'].median())
    df_all['BMI'] = df_all['BMI'].fillna(df_all['BMI'].median())

    if 'Cognitive_Impairment' not in df_all.columns:
        raise ValueError('라벨 컬럼(Cognitive_Impairment)을 찾을 수 없습니다.')

    y = df_all['Cognitive_Impairment'].apply(
        lambda v: 1 if str(v).strip().upper() == 'TRUE' or v is True else 0
    )

    X = df_all[FEATURE_COLS].copy()
    X = clip_features(X)

    if verbose:
        nan_rate = X.isna().mean().mean()
        print(f'  feature NaN 비율: {nan_rate:.1%}')
        print(f'  라벨 분포: {dict(y.value_counts())}')
        print(f'모델 학습 중... (feature: {len(FEATURE_COLS)}개, 샘플: {len(X)}명)')

    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)

    rf = RandomForestClassifier(**RF_PARAMS)
    gb = GradientBoostingClassifier(**GB_PARAMS)

    # RF는 class_weight='balanced_subsample' 이미 설정됨
    # GB는 class_weight 미지원 → scale_pos_weight 방식으로 대신 처리
    # → VotingClassifier.fit()에 sample_weight 없이 단순 fit
    model = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft',
        weights=[1, 1],
    )
    model.fit(X_imp, y)

    # prior probability 저장 (예측 실패 시 fallback용)
    prior = float(np.mean(y))

    joblib.dump(model, os.path.join(model_folder, 'momochi_model.pkl'))
    joblib.dump(imputer, os.path.join(model_folder, 'imputer.pkl'))
    with open(os.path.join(model_folder, 'features.json'), 'w') as f:
        json.dump(FEATURE_COLS, f, indent=2)
    with open(os.path.join(model_folder, 'prior.json'), 'w') as f:
        json.dump({'prior': prior}, f)

    if verbose:
        print(f'  prior probability: {prior:.4f}')
        print(f'모델 저장 완료 → {model_folder}')
        print('훈련 완료!')


def load_model(model_folder, verbose=False):
    model = joblib.load(os.path.join(model_folder, 'momochi_model.pkl'))
    imputer = joblib.load(os.path.join(model_folder, 'imputer.pkl'))
    with open(os.path.join(model_folder, 'features.json'), 'r') as f:
        features = json.load(f)
    prior_path = os.path.join(model_folder, 'prior.json')
    prior = 0.5
    if os.path.exists(prior_path):
        with open(prior_path, 'r') as f:
            prior = json.load(f).get('prior', 0.5)
    if verbose:
        print(f'모델 로드 완료! (prior={prior:.4f})')
    return {'model': model, 'imputer': imputer, 'features': features, 'prior': prior}


def run_model(model_dict, record, data_folder, verbose=False):
    prior = float(model_dict.get('prior', 0.5))
    bids_folder = str(record.get('BidsFolder', ''))

    try:
        clf = model_dict['model']
        imputer = model_dict['imputer']
        features = model_dict['features']

        site_id = str(record.get('SiteID', ''))
        session_id = record.get('SessionID', '')

        annot_path = build_annot_path(data_folder, site_id, bids_folder, session_id)
        feat = {}
        if os.path.exists(annot_path):
            feat = extract_caisr_features(annot_path)
            if not feat and verbose:
                print(f'  [경고] {bids_folder}: annotation 파싱 실패 → CAISR feature 없음')
        else:
            if verbose:
                print(f'  [경고] {bids_folder}: annotation 파일 없음 → median imputation')

        feat.update(extract_demo_features(record))

        try:
            feat['Age'] = float(record.get('Age', np.nan))
        except Exception:
            feat['Age'] = np.nan
        try:
            feat['BMI'] = float(record.get('BMI', np.nan))
        except Exception:
            feat['BMI'] = np.nan

        nan_features = [col for col in features if np.isnan(feat.get(col, np.nan))]
        if verbose and nan_features:
            print(f'  [NaN] {bids_folder}: {len(nan_features)}/{len(features)}개 feature NaN')

        row = {col: feat.get(col, np.nan) for col in features}
        X = pd.DataFrame([row])[features]
        X = clip_features(X)

        X_imp = imputer.transform(X)
        prob = float(clf.predict_proba(X_imp)[0][1])
        binary = int(prob >= 0.5)

        return binary, prob

    except Exception as e:
        # 0.5 고정 대신 훈련 데이터 prior 반환 → AUROC 손실 최소화
        if verbose:
            print(f'  [오류] {bids_folder}: 예측 실패({e}) → prior({prior:.4f}) 반환')
        binary = int(prior >= 0.5)
        return binary, prior
