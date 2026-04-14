 #!/usr/bin/env python

# ============================================================
# PhysioNet Challenge 2026 | Team: Momochi-SleepAI
#
# Model : VotingClassifier (RandomForest + GradientBoosting)
# Features : 26개 CAISR features + 6개 SaO2 features = 32개
#
# v10:
# - raw EDF SaO2 feature 추가 (6개)
#   → annotation 없는 검증셋/테스트셋 대응
# - find_phys_file: physiological_data 경로 탐색
# - extract_raw_features: SaO2 기반 6개 feature (속도 최적화)
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
from sklearn.impute import SimpleImputer, KNNImputer
import pyedflib

warnings.filterwarnings('ignore')

# 전체 모델: CAISR 26개 + SaO2 6개 = 32개
FEATURE_COLS = [
    # CAISR features (21개)
    'sleep_eff', 'n2_ratio', 'wake_ratio', 'waso_min', 'wake_bout_max',
    'transition_entropy', 'n3_ratio_2nd', 'rem_ratio_1st',
    'stage_confidence_min', 'stage_entropy_std', 'prob_r_mean',
    'prob_w_mean', 'arousal_index', 'ahi', 'arousal_burstiness',
    'arousal_cluster_ratio', 'rem_arousal_ratio', 'ahi_rem', 'ahi_nrem',
    'ahi_n3', 'ahi_rem_nrem_ratio',
    # demographics (5개)
    'Age', 'BMI', 'sex_num', 'race_num', 'ethnicity_num',
    # raw EDF features (6개) → SaO2 기반, annotation 없어도 추출 가능
    'sao2_mean', 'sao2_min', 'sao2_std',
    'sao2_below90', 'sao2_below85', 'odi4',
]

# 슬림 모델: annotation 없을 때 사용 (SaO2 6개 + demographics 5개 = 11개)
FEATURE_COLS_SLIM = [
    'Age', 'BMI', 'sex_num', 'race_num', 'ethnicity_num',
    'sao2_mean', 'sao2_min', 'sao2_std',
    'sao2_below90', 'sao2_below85', 'odi4',
]

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
    'sao2_mean': (50.0, 100.0),
    'sao2_min': (50.0, 100.0),
    'sao2_std': (0.0, 10.0),
    'sao2_below90': (0.0, 1.0),
    'sao2_below85': (0.0, 1.0),
    'odi4': (0.0, 500.0),
}

REQUIRED_CHANNELS = {'stage_caisr', 'caisr_prob_n3', 'caisr_prob_n2',
                     'caisr_prob_n1', 'caisr_prob_r', 'caisr_prob_w'}

RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,
    'min_samples_leaf': 12,
    'min_samples_split': 20,
    'max_features': 'sqrt',
    'max_samples': 0.8,
    'class_weight': 'balanced_subsample',
    'random_state': 42,
    'n_jobs': -1,
}

GB_PARAMS = {
    'n_estimators': 150,
    'max_depth': 3,
    'min_samples_leaf': 10,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'max_features': 'sqrt',
    'random_state': 42,
}


# ── 경로 탐색 ──────────────────────────────────────────────

def build_annot_path(data_folder, site_id, bids_folder, session_id):
    fname = f"{bids_folder}_ses-{session_id}_caisr_annotations.edf"
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


def find_phys_file(data_folder, site_id, bids_folder, session_id):
    """raw physiological EDF 경로 탐색"""
    fname = f"{bids_folder}_ses-{session_id}.edf"
    primary = os.path.join(data_folder, 'physiological_data',
                           str(site_id), fname)
    if os.path.exists(primary):
        return primary
    pattern1 = os.path.join(data_folder, 'physiological_data',
                            str(site_id), f'*{bids_folder}*.edf')
    hits = glob.glob(pattern1)
    if hits:
        return hits[0]
    pattern2 = os.path.join(data_folder, 'physiological_data',
                            '**', f'*{bids_folder}*.edf')
    hits = glob.glob(pattern2, recursive=True)
    if hits:
        return hits[0]
    return None


# ── 헬퍼 함수 ─────────────────────────────────────────────


def build_human_annot_path(data_folder, site_id, bids_folder, session_id):
    fname = f"{bids_folder}_ses-{session_id}_expert_annotations.edf"
    primary = os.path.join(data_folder, 'human_annotations', str(site_id), fname)
    if os.path.exists(primary):
        return primary
    pattern = os.path.join(data_folder, 'human_annotations', '**', f'*{bids_folder}*.edf')
    hits = glob.glob(pattern, recursive=True)
    return hits[0] if hits else None


def extract_human_features(edf_path):
    try:
        with pyedflib.EdfReader(edf_path) as f:
            raw_labels = f.getSignalLabels()
            lbl = {l.lower().strip(): i for i, l in enumerate(raw_labels)}

            if 'stage_expert' not in lbl:
                return {}

            stage = f.readSignal(lbl['stage_expert'])
            arousal = f.readSignal(lbl['arousal_expert']) if 'arousal_expert' in lbl else None
            resp = None
            resp_fs = 1
            if 'resp_expert' in lbl:
                resp_idx = lbl['resp_expert']
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

            # human annotation은 prob 채널 없음 → NaN
            feat['stage_confidence_min'] = np.nan
            feat['stage_entropy_std'] = np.nan
            feat['prob_r_mean'] = np.nan
            feat['prob_w_mean'] = np.nan

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
                    feat['rem_arousal_ratio'] = float(n_rem_ar / (n_nrem_ar + n_rem_ar + 1e-9))
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

def find_channel(labels, candidates):
    """대소문자/공백 무시하고 채널 인덱스 반환"""
    labels_lower = [l.lower().strip() for l in labels]
    for c in candidates:
        if c.lower().strip() in labels_lower:
            return labels_lower.index(c.lower().strip())
    return None


def count_events(signal, value):
    is_event = (signal == value)
    return int(np.sum(np.diff(is_event.astype(int)) == 1))


def get_bout_lengths(stage_arr, stage_val):
    is_s = (stage_arr == stage_val).astype(int)
    diff = np.diff(np.concatenate([[0], is_s, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return (ends - starts) * 0.5


def resp_burden_by_stage(stage_arr, resp_sig, stage_val, fs=1):
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


# ── Feature 추출 ───────────────────────────────────────────

def extract_demo_features(row):
    race_map = {'White': 0, 'Black': 1, 'Asian': 2, 'Hispanic': 3,
                'Others': 4, 'Unavailable': 4}
    eth_map = {'Not Hispanic': 0, 'Hispanic': 1, 'Unknown': 2}
    return {
        'sex_num': 1 if str(row.get('Sex', '')).strip() == 'Male' else 0,
        'race_num': race_map.get(str(row.get('Race', 'Unavailable')).strip(), 4),
        'ethnicity_num': eth_map.get(str(row.get('Ethnicity', 'Unknown')).strip(), 2),
    }


def extract_raw_features(edf_path):
    """raw physiological EDF에서 SaO2 + Airflow 기반 feature 추출"""
    try:
        with pyedflib.EdfReader(edf_path) as f:
            labels = f.getSignalLabels()
            feat = {}

            # ── SaO2 ──────────────────────────────────────
            sao2_idx = find_channel(labels, ['SaO2', 'SpO2', 'sao2', 'SAO2'])
            if sao2_idx is not None:
                sao2 = f.readSignal(sao2_idx)
                sao2_fs = f.getSampleFrequency(sao2_idx)
                sao2 = sao2[(sao2 > 50) & (sao2 <= 100)]
                if len(sao2) > 0:
                    feat['sao2_mean'] = float(np.mean(sao2))
                    feat['sao2_min'] = float(np.min(sao2))
                    feat['sao2_std'] = float(np.std(sao2))
                    feat['sao2_below90'] = float(np.mean(sao2 < 90))
                    feat['sao2_below85'] = float(np.mean(sao2 < 85))
                    # ODI4: 4% 산소포화도 저하 횟수
                    baseline = np.percentile(sao2, 90)
                    drops = np.diff((sao2 < (baseline - 4)).astype(int))
                    feat['odi4'] = float(np.sum(drops == 1))
                else:
                    for k in ['sao2_mean', 'sao2_min', 'sao2_std',
                              'sao2_below90', 'sao2_below85', 'odi4']:
                        feat[k] = np.nan
            else:
                for k in ['sao2_mean', 'sao2_min', 'sao2_std',
                          'sao2_below90', 'sao2_below85', 'odi4']:
                    feat[k] = np.nan

            # Airflow 제거 → SaO2만 사용 (속도 최적화)

            return feat

    except Exception:
        return {}


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
    X = X.copy()
    for col, (lo, hi) in CLIP_RANGES.items():
        if col in X.columns:
            X[col] = X[col].clip(lower=lo, upper=hi)
    return X


# ── train_model ────────────────────────────────────────────

def train_model(data_folder, model_folder, verbose=False):
    os.makedirs(model_folder, exist_ok=True)

    if verbose:
        print('데이터 로드 중...')

    df = pd.read_csv(os.path.join(data_folder, 'demographics.csv'))

    if verbose:
        print(f'환자 수: {len(df)}명')
        print('feature 추출 중... (CAISR + raw EDF)')

    records = []
    n_annot = 0
    n_raw = 0
    for _, row in df.iterrows():
        bids_folder = str(row.get('BidsFolder', ''))
        site_id = str(row.get('SiteID', ''))
        session_id = row.get('SessionID', '')

        feat = {'BidsFolder': bids_folder}
        feat.update(extract_demo_features(row))

        # CAISR annotation feature 우선, 없으면 human annotation으로 대체
        annot_path = build_annot_path(data_folder, site_id, bids_folder, session_id)
        if os.path.exists(annot_path):
            n_annot += 1
            caisr_feat = extract_caisr_features(annot_path)
            if caisr_feat:
                feat.update(caisr_feat)
            else:
                # CAISR 파싱 실패 → human annotation으로 대체
                human_path = build_human_annot_path(data_folder, site_id, bids_folder, session_id)
                if human_path:
                    feat.update(extract_human_features(human_path))
        else:
            # CAISR 없으면 human annotation으로 대체
            human_path = build_human_annot_path(data_folder, site_id, bids_folder, session_id)
            if human_path:
                feat.update(extract_human_features(human_path))

        # raw EDF feature (SaO2)
        phys_path = find_phys_file(data_folder, site_id, bids_folder, session_id)
        if phys_path:
            n_raw += 1
            feat.update(extract_raw_features(phys_path))

        records.append(feat)

    if verbose:
        print(f'  annotation 찾은 환자: {n_annot}/{len(df)}명')
        print(f'  raw EDF 찾은 환자: {n_raw}/{len(df)}명')

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

    imputer = KNNImputer(n_neighbors=5)
    X_imp = imputer.fit_transform(X)

    rf = RandomForestClassifier(**RF_PARAMS)
    gb = GradientBoostingClassifier(**GB_PARAMS)

    model = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft',
        weights=[1, 1],
    )
    model.fit(X_imp, y)

    prior = float(np.mean(y))

    joblib.dump(model, os.path.join(model_folder, 'momochi_model.pkl'))
    joblib.dump(imputer, os.path.join(model_folder, 'imputer.pkl'))
    with open(os.path.join(model_folder, 'features.json'), 'w') as f:
        json.dump(FEATURE_COLS, f, indent=2)
    with open(os.path.join(model_folder, 'prior.json'), 'w') as f:
        json.dump({'prior': prior}, f)

    # ── slim 모델 학습 (annotation 없을 때 fallback) ──────────
    if verbose:
        print('slim 모델 학습 중... (SaO2 + demographics만)')
    X_slim = df_all[FEATURE_COLS_SLIM].copy()
    X_slim = clip_features(X_slim)
    imputer_slim = SimpleImputer(strategy='median')
    X_slim_imp = imputer_slim.fit_transform(X_slim)

    rf_slim = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=12,
        min_samples_split=20, max_features='sqrt', max_samples=0.8,
        class_weight='balanced_subsample', random_state=42, n_jobs=-1,
    )
    gb_slim = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, min_samples_leaf=10,
        learning_rate=0.05, subsample=0.8, max_features='sqrt', random_state=42,
    )
    model_slim = VotingClassifier(
        estimators=[('rf', rf_slim), ('gb', gb_slim)],
        voting='soft', weights=[1, 1],
    )
    model_slim.fit(X_slim_imp, y)

    joblib.dump(model_slim, os.path.join(model_folder, 'momochi_model_slim.pkl'))
    joblib.dump(imputer_slim, os.path.join(model_folder, 'imputer_slim.pkl'))
    with open(os.path.join(model_folder, 'features_slim.json'), 'w') as f:
        json.dump(FEATURE_COLS_SLIM, f, indent=2)

    if verbose:
        print(f'  prior probability: {prior:.4f}')
        print(f'모델 저장 완료 → {model_folder}')
        print('훈련 완료!')


# ── load_model ─────────────────────────────────────────────

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

    # slim 모델 로드
    model_slim = joblib.load(os.path.join(model_folder, 'momochi_model_slim.pkl'))
    imputer_slim = joblib.load(os.path.join(model_folder, 'imputer_slim.pkl'))
    with open(os.path.join(model_folder, 'features_slim.json'), 'r') as f:
        features_slim = json.load(f)

    if verbose:
        print(f'모델 로드 완료! (prior={prior:.4f})')
    return {
        'model': model, 'imputer': imputer, 'features': features,
        'model_slim': model_slim, 'imputer_slim': imputer_slim, 'features_slim': features_slim,
        'prior': prior,
    }


# ── run_model ──────────────────────────────────────────────

def run_model(model_dict, record, data_folder, verbose=False):
    prior = float(model_dict.get('prior', 0.5))
    bids_folder = str(record.get('BidsFolder', ''))

    try:
        clf = model_dict['model']
        imputer = model_dict['imputer']
        features = model_dict['features']

        site_id = str(record.get('SiteID', ''))
        session_id = record.get('SessionID', '')

        # CAISR annotation feature
        annot_path = build_annot_path(data_folder, site_id, bids_folder, session_id)
        feat = {}
        if os.path.exists(annot_path):
            feat = extract_caisr_features(annot_path)
            if not feat and verbose:
                print(f'  [경고] {bids_folder}: annotation 파싱 실패')
        else:
            if verbose:
                print(f'  [경고] {bids_folder}: annotation 없음 → raw EDF로 대체')

        # raw EDF feature (항상 시도)
        phys_path = find_phys_file(data_folder, site_id, bids_folder, session_id)
        if phys_path:
            raw_feat = extract_raw_features(phys_path)
            feat.update(raw_feat)
        elif verbose:
            print(f'  [경고] {bids_folder}: raw EDF도 없음')

        # demographics 보완
        demo_record = dict(record)
        if any(k not in demo_record for k in ['Age', 'BMI', 'Sex', 'Race', 'Ethnicity']):
            demo_path = os.path.join(data_folder, 'demographics.csv')
            if os.path.exists(demo_path):
                df_demo = pd.read_csv(demo_path)
                mask = (df_demo['BidsFolder'] == bids_folder) & \
                       (df_demo['SessionID'] == session_id)
                rows = df_demo.loc[mask]
                if not rows.empty:
                    demo_record.update(rows.iloc[0].to_dict())

        feat.update(extract_demo_features(demo_record))
        try:
            feat['Age'] = float(demo_record.get('Age', np.nan))
        except Exception:
            feat['Age'] = np.nan
        try:
            feat['BMI'] = float(demo_record.get('BMI', np.nan))
        except Exception:
            feat['BMI'] = np.nan

        # CAISR feature가 하나라도 있으면 full 모델, 없으면 slim 모델
        caisr_cols = [col for col in features if col not in
                      ['Age', 'BMI', 'sex_num', 'race_num', 'ethnicity_num',
                       'sao2_mean', 'sao2_min', 'sao2_std', 'sao2_below90', 'sao2_below85', 'odi4']]
        has_caisr = any(not np.isnan(feat.get(col, np.nan)) for col in caisr_cols)

        if has_caisr:
            # full 모델 사용
            row = {col: feat.get(col, np.nan) for col in features}
            X = pd.DataFrame([row])[features]
            X = clip_features(X)
            X_imp = imputer.transform(X)
            prob = float(clf.predict_proba(X_imp)[0][1])
        else:
            # slim 모델 사용 (annotation 없을 때)
            if verbose:
                print(f'  [slim] {bids_folder}: annotation 없음 → slim 모델 사용')
            clf_slim = model_dict.get('model_slim')
            imputer_slim = model_dict.get('imputer_slim')
            features_slim = model_dict.get('features_slim', [])
            if clf_slim is not None:
                row_slim = {col: feat.get(col, np.nan) for col in features_slim}
                X_slim = pd.DataFrame([row_slim])[features_slim]
                X_slim = clip_features(X_slim)
                X_slim_imp = imputer_slim.transform(X_slim)
                prob = float(clf_slim.predict_proba(X_slim_imp)[0][1])
            else:
                prob = prior

        binary = int(prob >= 0.5)
        return binary, prob

    except Exception as e:
        if verbose:
            print(f'  [오류] {bids_folder}: 예측 실패({e}) → prior({prior:.4f}) 반환')
        binary = int(prior >= 0.5)
        return binary, prior
