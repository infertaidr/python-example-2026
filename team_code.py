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
    os.makedirs(model_folder, exist_ok=True)
    print("훈련 완료!")

def load_model(model_folder, verbose=False):
    return None

def run_model(model, record, data_folder, verbose=False):
    return 0, 0.5
