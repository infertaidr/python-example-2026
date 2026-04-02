#!/usr/bin/env python

import joblib
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import pyedflib

###############################################################
# 수면 피처 추출 함수
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
                features['N3_ratio']   = np.mean(stage == 1)
                features['N2_ratio']   = np.mean(stage == 2)
                features['N1_ratio']   = np.mean(stage == 3)
                features['REM_ratio']  = np.mean(stage == 4)
                features['Wake_ratio'] = np.mean(stage == 5)

        if 'resp_caisr' in signals:
            features['apnea_ratio'] = np.mean(signals['resp_caisr'] > 0)

        if 'arousal_caisr' in signals:
            features['arousal_ratio'] = np.mean(signals['arousal_caisr'] == 1)

        if 'limb_caisr' in signals:
            features['limb_ratio'] = np.mean(signals['limb_caisr'] > 0)

        return features
    except:
        return {}


def make_smart_features(df):
    """임상 지식 기반 스마트 피처 생성"""
    df = df.copy()

    # 성별, 인종 숫자 변환
    df['Sex_num'] = (df['Sex'] == 'Male').astype(int)
    race_map = {'White':0,'Black':1,'Asian':2,'Others':3,'Unavailable':4}
    df['Race_num'] = df['Race'].map(race_map).fillna(3)

    # 수면 품질 (깊은잠 + REM)
    df['sleep_quality'] = (
        df.get('N3_ratio', pd.Series(0, index=df.index)).fillna(0) +
        df.get('REM_ratio', pd.Series(0, index=df.index)).fillna(0)
    )

    # 수면 방해 지수
    df['sleep_disturbance'] = (
        df.get('arousal_ratio', pd.Series(0, index=df.index)).fillna(0) +
        df.get('apnea_ratio', pd.Series(0, index=df.index)).fillna(0) +
        df.get('limb_ratio', pd.Series(0, index=df.index)).fillna(0)
    )

    # BMI × 무호흡 복합 위험도
    df['bmi_apnea'] = (
        df['BMI'].fillna(df['BMI'].median()) *
        df.get('apnea_ratio', pd.Series(0, index=df.index)).fillna(0)
    )

    return df


###############################################################
# train_model: 훈련 함수 (필수!)
###############################################################

def train_model(data_folder, model_folder, verbose=False):
    """모델 훈련 및 저장"""

    print("훈련 시작...")

    # 1. 데이터 로드
    demo_path = os.path.join(data_folder, 'demographics.csv')
    df = pd.read_csv(demo_path)

    annot_base = os.path.join(data_folder, 'algorithmic_annotations')

    # 2. 수면 피처 추출
    sleep_features = []
    for _, row in df.iterrows():
        bids = row['BidsFolder']
        site = row['SiteID']
        ses  = row['SessionID']
        annot_file = os.path.join(
            annot_base, site,
            f'{bids}_ses-{ses}_caisr_annotations.edf'
        )
        if os.path.exists(annot_file):
            feat = extract_caisr_features(annot_file)
        else:
            feat = {}
        feat['BDSPPatientID'] = row['BDSPPatientID']
        sleep_features.append(feat)

    df_sleep = pd.DataFrame(sleep_features)
    df = df.merge(df_sleep, on='BDSPPatientID', how='left')

    # 3. 스마트 피처 생성
    df = make_smart_features(df)

    # 4. 학습 데이터 준비
    FEATURES = [
        'Age', 'BMI', 'Sex_num', 'Race_num',
        'sleep_quality', 'sleep_disturbance', 'bmi_apnea'
    ]

    X = df[FEATURES].values
    y = df['Cognitive_Impairment'].astype(int).values

    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    # 5. 모델 훈련
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    # 6. 저장
    os.makedirs(model_folder, exist_ok=True)
    joblib.dump(model,   os.path.join(model_folder, 'model.pkl'))
    joblib.dump(imputer, os.path.join(model_folder, 'imputer.pkl'))
    joblib.dump(FEATURES, os.path.join(model_folder, 'features.pkl'))

    print(f"✅ 모델 저장 완료! → {model_folder}")


###############################################################
# run_model: 예측 함수 (필수!)
###############################################################

def run_model(data_folder, model_folder, output_folder):
    """새 환자 예측"""

    print("예측 시작...")

    # 1. 모델 로드
    model   = joblib.load(os.path.join(model_folder, 'model.pkl'))
    imputer = joblib.load(os.path.join(model_folder, 'imputer.pkl'))
    FEATURES = joblib.load(os.path.join(model_folder, 'features.pkl'))

    # 2. 데이터 로드
    demo_path = os.path.join(data_folder, 'demographics.csv')
    df = pd.read_csv(demo_path)

    annot_base = os.path.join(data_folder, 'algorithmic_annotations')

    # 3. 수면 피처 추출
    sleep_features = []
    for _, row in df.iterrows():
        bids = row['BidsFolder']
        site = row['SiteID']
        ses  = row['SessionID']
        annot_file = os.path.join(
            annot_base, site,
            f'{bids}_ses-{ses}_caisr_annotations.edf'
        )
        if os.path.exists(annot_file):
            feat = extract_caisr_features(annot_file)
        else:
            feat = {}
        feat['BDSPPatientID'] = row['BDSPPatientID']
        sleep_features.append(feat)

    df_sleep = pd.DataFrame(sleep_features)
    df = df.merge(df_sleep, on='BDSPPatientID', how='left')

    # 4. 스마트 피처 생성
    df = make_smart_features(df)

    # 5. 예측
    X = df[FEATURES].values
    X = imputer.transform(X)
    probs = model.predict_proba(X)[:, 1]

    # 6. 결과 저장
    os.makedirs(output_folder, exist_ok=True)
    output = pd.DataFrame({
        'BDSPPatientID': df['BDSPPatientID'],
        'Cognitive_Impairment_Probability': probs
    })
    output.to_csv(
        os.path.join(output_folder, 'predictions.csv'),
        index=False
    )
    print(f"✅ 예측 완료! → {output_folder}")
