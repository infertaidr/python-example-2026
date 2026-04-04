#!/usr/bin/env python

# ============================================================
# PhysioNet Challenge 2026 | Team: Momochi-SleepAI
#
# Model    : RandomForestClassifier
# Features : 14개 CAISR-based sleep structure features
# Validation: Site-based leave-one-out CV
#   - S0001→I0006 AUROC: 0.6692
#   - S0001→I0002 AUROC: 0.4892
#
# Feature selection 근거:
#   - 3개 site 모두에서 라벨과 일관된 상관관계를 보이는 feature만 선택
#   - arousal_interval_std는 site 간 분포 차이가 너무 커서 제외
#     (I0006에서 std=2606으로 노이즈 심함 → 일반화 방해)
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
# 확정 feature 목록 (순서 고정 - train/inference 동일하게 유지)
# ============================================================
FEATURE_COLS = [
    'sleep_eff',             # 수면 효율
    'n2_ratio',              # N2 비율
    'wake_ratio',            # 각성 비율
    'waso_min',              # 수면 후 각성 시간 (분)
    'wake_bout_max',         # 최장 각성 구간 (분)
    'transition_entropy',    # 수면 단계 전환 엔트로피
    'n3_ratio_2nd',          # 후반부 N3 비율
    'rem_ratio_1st',         # 전반부 REM 비율
    'stage_confidence_min',  # CAISR 최소 신뢰도
    'stage_entropy_std',     # CAISR 엔트로피 변동성
    'prob_r_mean',           # REM 확률 평균
    'prob_w_mean',           # Wake 확률 평균
    'arousal_index',         # 각성 지수 (시간당)
    'ahi',                   # 무호흡-저호흡 지수
]

# ============================================================
# 헬퍼 함수
# ============================================================
def count_events(signal, value):
    """연속 구간을 이벤트 1개로 카운트 (onset 기준)"""
    is_event = (signal == value)
    onsets = np.sum(np.diff(is_event.astype(int)) == 1)
    return int(onsets)

def get_bout_lengths(stage_arr, stage_val):
    """특정 수면 단계의 연속 구간 길이 (분 단위)"""
    is_s = (stage_arr == stage_val).astype(int)
    diff = np.diff(np.concatenate([[0], is_s, [0]]))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    return (ends - starts) * 0.5  # 30초 epoch → 분

def find_annot_file(patient_id, data_folder):
    """환자 ID로 CAISR annotation EDF 파일 경로 찾기"""
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
# CAISR feature 추출
# ============================================================
def extract_caisr_features(edf_path):
    """
    CAISR annotation EDF에서 수면 구조 feature 추출
    반환: dict (FEATURE_COLS에 해당하는 값들)
    실패 시: 빈 dict 반환 → 나중에 median imputation
    """
    try:
        with pyedflib.EdfReader(edf_path) as f:
            labels = f.getSignalLabels()
            label_to_idx = {l: i for i, l in enumerate(labels)}

            stage   = f.readSignal(label_to_idx['stage_caisr'])
            prob_n3 = f.readSignal(label_to_idx['caisr_prob_n3'])
            prob_n2 = f.readSignal(label_to_idx['caisr_prob_n2'])
            prob_n1 = f.readSignal(label_to_idx['caisr_prob_n1'])
            prob_r  = f.readSignal(label_to_idx['caisr_prob_r'])
            prob_w  = f.readSignal(label_to_idx['caisr_prob_w'])
            arousal = f.readSignal(label_to_idx['arousal_caisr']) if 'arousal_caisr' in label_to_idx else None
            resp    = f.readSignal(label_to_idx['resp_caisr'])    if 'resp_caisr'    in label_to_idx else None

        # 유효 epoch (9=unknown 제외)
        valid = stage != 9
        stage_v = stage[valid]
        total_epochs = len(stage_v)
        tst_epochs = np.sum(stage_v != 5)
        tst_hours = tst_epochs * 30 / 3600

        if total_epochs == 0 or tst_hours == 0:
            return {}

        feat = {}

        # 1. 기본 수면 구조
        feat['sleep_eff']  = float(tst_epochs / total_epochs)
        feat['n2_ratio']   = float(np.mean(stage_v == 2))
        feat['wake_ratio'] = float(np.mean(stage_v == 5))
        feat['waso_min']   = float(np.sum(stage_v == 5) * 0.5)

        # 2. 최장 각성 구간
        wake_bouts = get_bout_lengths(stage_v, 5)
        feat['wake_bout_max'] = float(np.max(wake_bouts)) if len(wake_bouts) > 0 else 0.0

        # 3. 수면 단계 전환 엔트로피
        transitions = list(zip(stage_v[:-1], stage_v[1:]))
        trans_counts = Counter(transitions)
        total_trans = sum(trans_counts.values())
        trans_probs = np.array([v / total_trans for v in trans_counts.values()])
        trans_probs = np.clip(trans_probs, 1e-9, 1)
        feat['transition_entropy'] = float(-np.sum(trans_probs * np.log(trans_probs)))

        # 4. 전반/후반 수면 구조
        half = total_epochs // 2
        feat['n3_ratio_2nd']  = float(np.mean(stage_v[half:] == 1))
        feat['rem_ratio_1st'] = float(np.mean(stage_v[:half] == 4))

        # 5. CAISR confidence
        probs = np.stack([prob_n3, prob_n2, prob_n1, prob_r, prob_w], axis=1)
        prob_max = np.max(probs, axis=1)
        feat['stage_confidence_min'] = float(np.min(prob_max))

        probs_clip = np.clip(probs, 1e-9, 1)
        entropy = -np.sum(probs_clip * np.log(probs_clip), axis=1)
        feat['stage_entropy_std'] = float(np.std(entropy))

        feat['prob_r_mean'] = float(np.mean(prob_r))
        feat['prob_w_mean'] = float(np.mean(prob_w))

        # 6. arousal index (onset 기준)
        # ※ arousal_interval_std는 site 간 분포 차이가 너무 커서 제외
        if arousal is not None:
            n_ar = count_events(arousal, 1)
            feat['arousal_index'] = float(n_ar / tst_hours)
        else:
            feat['arousal_index'] = np.nan

        # 7. AHI (onset 기준)
        if resp is not None:
            n_oa = count_events(resp, 1)
            n_ca = count_events(resp, 2)
            n_hy = count_events(resp, 4)
            feat['ahi'] = float((n_oa + n_ca + n_hy) / tst_hours)
        else:
            feat['ahi'] = np.nan

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

    # CAISR feature 추출
    records = []
    for _, row in df.iterrows():
        pid = row.get('BDSPPatientID', '')
        feat = {'BDSPPatientID': pid}
        edf_path = find_annot_file(pid, data_folder)
        if edf_path:
            feat.update(extract_caisr_features(edf_path))
        records.append(feat)

    df_feat = pd.DataFrame(records)
    df_all = df.merge(df_feat, on='BDSPPatientID', how='left')

    # 라벨
    if 'Cognitive_Impairment' in df_all.columns:
        y = df_all['Cognitive_Impairment'].astype(int)
    else:
        raise ValueError('라벨 컬럼(Cognitive_Impairment)을 찾을 수 없습니다.')

    # X 준비 (feature 순서 고정)
    X = df_all[FEATURE_COLS].copy()

    if verbose:
        print(f'🤖 모델 학습 중... (feature: {len(FEATURE_COLS)}개, 샘플: {len(X)}명)')

    # Imputer 학습 및 변환
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)

    # RandomForest 학습
    model = RandomForestClassifier(
        n_estimators=500,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_imp, y)

    # 저장
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

        # CAISR feature 추출
        edf_path = find_annot_file(patient_id, data_folder)
        feat = {}
        if edf_path:
            feat = extract_caisr_features(edf_path)

        # feature 순서 맞춰 DataFrame 생성
        row = {col: feat.get(col, np.nan) for col in features}
        X = pd.DataFrame([row])[features]

        # Impute → 예측
        X_imp = imputer.transform(X)
        prob   = float(clf.predict_proba(X_imp)[0][1])
        binary = int(prob >= 0.5)

        return binary, prob

    except Exception as e:
        if verbose:
            print(f'⚠️ 예측 실패: {e}')
        return 0, 0.5
