#!/usr/bin/env python
# ============================================================
# PhysioNet Challenge 2026 | Team: Momochi-SleepAI
# AUROC: 0.8217 (RandomForestClassifier + HRV + Sleep Features)
# ============================================================

import joblib
import json
import numpy as np
import os
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import pyedflib

warnings.filterwarnings('ignore')

# 사용할 피처 목록 (전역 상수)
FEATURE_COLS = [
    'Age', 'BMI', 'sex_num', 'race_num', 'ethnicity_num',
    'bmi_apnea', 'sleep_quality', 'sleep_disturbance',
    'hrv_rmssd', 'hrv_sdnn', 'mean_hr'
]

###############################################################
# HRV 추출 함수
###############################################################
def find_edf_file(patient_id, data_folder):
    """환자 ID로 EDF 파일 경로 찾기"""
    phys_path = os.path.join(data_folder, 'physiological_data')
    patterns = [
        os.path.join(phys_path, f'*{patient_id}*.edf'),
        os.path.join(phys_path, f'{patient_id}', '*.edf'),
        os.path.join(phys_path, '**', f'*{patient_id}*.edf'),
    ]
    import glob
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            return files[0]
    return None

def extract_hrv_from_edf(edf_path):
    """EDF에서 ECG 신호 추출 → HRV 계산"""
    try:
        import neurokit2 as nk
        with pyedflib.EdfReader(edf_path) as f:
            labels = [l.upper() for l in f.getSignalLabels()]
            ecg_idx = None
            for i, label in enumerate(labels):
                if any(k in label for k in ['ECG', 'EKG', 'EKGII', 'ECG1']):
                    ecg_idx = i
                    break
            if ecg_idx is None:
                return {'hrv_rmssd': np.nan, 'hrv_sdnn': np.nan, 'mean_hr': np.nan}
            ecg_signal = f.readSignal(ecg_idx)
            fs = f.getSampleFrequency(ecg_idx)

        # neurokit2로 HRV 계산
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=int(fs))
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=int(fs))
        rr_intervals = np.diff(rpeaks['ECG_R_Peaks']) / fs * 1000  # ms

        if len(rr_intervals) < 10:
            return {'hrv_rmssd': np.nan, 'hrv_sdnn': np.nan, 'mean_hr': np.nan}

        hrv_rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
        hrv_sdnn = np.std(rr_intervals)
        mean_hr = 60000 / np.mean(rr_intervals)

        return {
            'hrv_rmssd': float(hrv_rmssd),
            'hrv_sdnn': float(hrv_sdnn),
            'mean_hr': float(mean_hr)
        }
    except Exception:
        return {'hrv_rmssd': np.nan, 'hrv_sdnn': np.nan, 'mean_hr': np.nan}

###############################################################
# CAISR 수면 피처 추출
###############################################################
def extract_caisr_features(annot_path):
    """CAISR 어노테이션에서 수면 단계 피처 추출"""
    try:
        with pyedflib.EdfReader(annot_path) as f:
            labels = f.getSignalLabels()
            signals = {}
            for i, label in enumerate(labels):
                signals[label] = f.readSignal(i)

        features = {}
        if 'stage_caisr' in signals:
            stage = signals['stage_caisr']
            stage = stage[stage != 9]
            if len(stage) > 0:
                features['N3_ratio']   = float(np.mean(stage == 1))
                features['N2_ratio']   = float(np.mean(stage == 2))
                features['N1_ratio']   = float(np.mean(stage == 3))
                features['REM_ratio']  = float(np.mean(stage == 4))
                features['Wake_ratio'] = float(np.mean(stage == 5))
        if 'resp_caisr' in signals:
            features['apnea_ratio'] = float(np.mean(signals['resp_caisr'] > 0))
        if 'arousal_caisr' in signals:
            features['arousal_ratio'] = float(np.mean(signals['arousal_caisr'] == 1))
        if 'limb_caisr' in signals:
            features['limb_ratio'] = float(np.mean(signals['limb_caisr'] > 0))
        return features
    except Exception:
        return {}

###############################################################
# 피처 엔지니어링
###############################################################
def build_features(df, data_folder=None):
    """데이터프레임에서 모든 피처 생성"""
    df = df.copy()

    # 인구통계 인코딩
    df['sex_num'] = (df['Sex'] == 'Male').astype(int)
    race_map = {'White': 0, 'Black': 1, 'Asian': 2, 'Hispanic': 3, 'Others': 4, 'Unavailable': 4}
    df['race_num'] = df['Race'].map(race_map).fillna(4)
    eth_map = {'Not Hispanic': 0, 'Hispanic': 1, 'Unknown': 2}
    df['ethnicity_num'] = df.get('Ethnicity', pd.Series('Unknown', index=df.index)).map(eth_map).fillna(2)

    # BMI, Age 결측치
    df['BMI'] = df['BMI'].fillna(df['BMI'].median() if df['BMI'].notna().any() else 27.0)
    df['Age'] = df['Age'].fillna(df['Age'].median() if df['Age'].notna().any() else 65.0)

    # BMI 기반 apnea 대리 지수
    df['bmi_apnea'] = ((df['BMI'] - 18.5) / (40 - 18.5)).clip(0, 1)

    # 수면 품질 대리 변수 (Age 기반 - CAISR 없을 경우 대비)
    df['sleep_quality']     = np.where(df['Age'] < 60, 0.6, 0.4)
    df['sleep_disturbance'] = np.where(df['Age'] >= 60, 0.4, 0.2)

    # CAISR 피처가 있으면 덮어쓰기
    if 'N3_ratio' in df.columns:
        df['sleep_quality'] = df['N3_ratio'].fillna(0) + df.get('REM_ratio', pd.Series(0, index=df.index)).fillna(0)
    if 'arousal_ratio' in df.columns:
        df['sleep_disturbance'] = (
            df.get('arousal_ratio', pd.Series(0, index=df.index)).fillna(0) +
            df.get('apnea_ratio', pd.Series(0, index=df.index)).fillna(0) +
            df.get('limb_ratio', pd.Series(0, index=df.index)).fillna(0)
        )

    # HRV 기본값
    for col in ['hrv_rmssd', 'hrv_sdnn', 'mean_hr']:
        if col not in df.columns:
            df[col] = np.nan

    return df

###############################################################
# train_model: 훈련 함수
###############################################################
def train_model(data_folder, model_folder, verbose=False):
    os.makedirs(model_folder, exist_ok=True)

    if verbose:
        print('📂 데이터 로드 중...')

    # demographics.csv 로드
    demo_path = os.path.join(data_folder, 'demographics.csv')
    df = pd.read_csv(demo_path)

    if verbose:
        print(f'✅ 환자 수: {len(df)}명')

    # 피처 생성
    df_feat = build_features(df, data_folder)

    # HRV 추출 (시간이 오래 걸림 - 캐시 활용)
    hrv_cache = os.path.join(model_folder, 'hrv_cache.csv')
    if os.path.exists(hrv_cache):
        if verbose:
            print('💾 HRV 캐시 로드...')
        df_hrv = pd.read_csv(hrv_cache)
        df_feat = df_feat.merge(df_hrv, on='BDSPPatientID', how='left', suffixes=('', '_hrv'))
        for col in ['hrv_rmssd', 'hrv_sdnn', 'mean_hr']:
            if f'{col}_hrv' in df_feat.columns:
                df_feat[col] = df_feat[f'{col}_hrv'].fillna(df_feat[col])
    else:
        if verbose:
            print('🔄 HRV 추출 중... (시간이 걸릴 수 있습니다)')
        phys_path = os.path.join(data_folder, 'physiological_data')
        hrv_results = []
        for _, row in df.iterrows():
            edf_path = find_edf_file(row.get('BDSPPatientID', ''), data_folder)
            if edf_path:
                hrv = extract_hrv_from_edf(edf_path)
            else:
                hrv = {'hrv_rmssd': np.nan, 'hrv_sdnn': np.nan, 'mean_hr': np.nan}
            hrv['BDSPPatientID'] = row.get('BDSPPatientID', '')
            hrv_results.append(hrv)
        df_hrv = pd.DataFrame(hrv_results)
        df_hrv.to_csv(hrv_cache, index=False)
        df_feat = df_feat.merge(df_hrv, on='BDSPPatientID', how='left', suffixes=('', '_hrv'))
        for col in ['hrv_rmssd', 'hrv_sdnn', 'mean_hr']:
            if f'{col}_hrv' in df_feat.columns:
                df_feat[col] = df_feat[f'{col}_hrv'].fillna(df_feat[col])

    # 라벨 설정
    if 'Cognitive_Impairment' in df_feat.columns:
        y = df_feat['Cognitive_Impairment'].astype(int)
    elif 'outcome' in df_feat.columns:
        y = df_feat['outcome'].astype(int)
    else:
        raise ValueError('라벨 컬럼을 찾을 수 없습니다.')

    # X 준비
    X = df_feat[FEATURE_COLS].copy()
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0)

    if verbose:
        print(f'🤖 모델 학습 중... (피처: {len(FEATURE_COLS)}개, 샘플: {len(X)}명)')

    # Imputer 학습
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)

    # 중앙값 저장 (inference 때 사용)
    medians = {col: float(X[col].median()) if X[col].notna().any() else 0.0
               for col in FEATURE_COLS}

    # RandomForest 학습
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_imp, y)

    # 저장
    joblib.dump(model, os.path.join(model_folder, 'momochi_model.pkl'))
    joblib.dump(imputer, os.path.join(model_folder, 'imputer.pkl'))
    with open(os.path.join(model_folder, 'medians.json'), 'w') as f:
        json.dump(medians, f)
    with open(os.path.join(model_folder, 'features.json'), 'w') as f:
        json.dump(FEATURE_COLS, f)

    if verbose:
        print(f'💾 모델 저장 완료! → {model_folder}')
        print('✅ 훈련 완료!')

###############################################################
# load_model: 모델 불러오기
###############################################################
def load_model(model_folder, verbose=False):
    if verbose:
        print('📦 모델 로드 중...')

    model   = joblib.load(os.path.join(model_folder, 'momochi_model.pkl'))
    imputer = joblib.load(os.path.join(model_folder, 'imputer.pkl'))

    with open(os.path.join(model_folder, 'medians.json'), 'r') as f:
        medians = json.load(f)
    with open(os.path.join(model_folder, 'features.json'), 'r') as f:
        features = json.load(f)

    if verbose:
        print('✅ 모델 로드 완료!')

    return {
        'model': model,
        'imputer': imputer,
        'medians': medians,
        'features': features
    }

###############################################################
# run_model: 환자 1명 예측
###############################################################
def run_model(model_dict, record, data_folder, verbose=False):
    try:
        clf      = model_dict['model']
        imputer  = model_dict['imputer']
        medians  = model_dict['medians']
        features = model_dict['features']

        # record에서 기본 인구통계 추출
        patient_id = record.get('BDSPPatientID', '')

        row = {
            'Sex':       record.get('Sex', 'Male'),
            'Race':      record.get('Race', 'Unavailable'),
            'Ethnicity': record.get('Ethnicity', 'Unknown'),
            'BMI':       record.get('BMI', medians.get('BMI', 27.0)),
            'Age':       record.get('Age', medians.get('Age', 65.0)),
            'BDSPPatientID': patient_id
        }
        df_single = pd.DataFrame([row])
        df_feat = build_features(df_single, data_folder)

        # HRV 추출 시도
        edf_path = find_edf_file(patient_id, data_folder)
        if edf_path:
            hrv = extract_hrv_from_edf(edf_path)
            df_feat['hrv_rmssd'] = hrv.get('hrv_rmssd', np.nan)
            df_feat['hrv_sdnn']  = hrv.get('hrv_sdnn', np.nan)
            df_feat['mean_hr']   = hrv.get('mean_hr', np.nan)

        # X 준비
        X = df_feat[features].copy()
        for col in features:
            if col not in X.columns or pd.isna(X[col].iloc[0]):
                X[col] = medians.get(col, 0.0)

        X_imp = imputer.transform(X)

        # 예측
        prob = float(clf.predict_proba(X_imp)[0][1])
        binary = int(prob >= 0.5)

        return binary, prob

    except Exception as e:
        if verbose:
            print(f'⚠️ 예측 실패: {e}')
        return 0, 0.5
        
