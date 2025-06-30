#!/usr/bin/env python3
"""
Step 2: GRPO 기본 예제 구현 및 검증

작성일: 2025.01.30
목적: TRL GRPOTrainer를 사용한 기본 GRPO 동작 검증 및 펀드평가 적용 가능성 확인

참고: step1_grpo_research_analysis.md의 연구 결과 기반 구현
"""

import os
import json
import torch
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# TRL 및 관련 라이브러리
from trl import GRPOConfig, GRPOTrainer
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed
)
from datasets import Dataset, load_dataset
import numpy as np
import matplotlib.pyplot as plt

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('step2_grpo_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass 
class ExperimentConfig:
    """실험 설정 클래스"""
    # 모델 설정 (step1 연구 결과 기반)
    model_name: str = "Qwen/Qwen2-0.5B-Instruct"  # TRL 공식 예제 모델
    output_dir: str = "step2_grpo_experiment"
    
    # GRPO 핵심 설정 (step1 분석 기반)
    num_generations: int = 8  # 그룹 크기
    max_prompt_length: int = 512
    max_completion_length: int = 256
    learning_rate: float = 1e-6
    kl_penalty: float = 0.04
    
    # 훈련 설정
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    save_steps: int = 100
    logging_steps: int = 10
    
    # 실험 설정
    max_samples: int = 100  # 빠른 검증을 위한 샘플 수
    seed: int = 42

class FundEvaluationRewardFunctions:
    """펀드평가 특화 보상 함수 (step1 연구 기반)"""
    
    @staticmethod
    def json_format_reward(completions: List[str], **kwargs) -> List[float]:
        """JSON 형식 준수 보상 (0-10점)"""
        rewards = []
        for completion in completions:
            try:
                # JSON 파싱 시도
                json.loads(completion.strip())
                rewards.append(10.0)  # 완벽한 JSON
            except json.JSONDecodeError:
                # JSON 키워드 존재 여부로 부분 점수
                json_keywords = ['{', '}', '"', ':']
                score = sum(2.5 for keyword in json_keywords if keyword in completion)
                rewards.append(max(0.0, min(score, 10.0)))
        return rewards
    
    @staticmethod
    def length_quality_reward(completions: List[str], target_length: int = 150, **kwargs) -> List[float]:
        """적절한 길이 보상 (0-30점) - 펀드평가 결과의 적정 길이"""
        rewards = []
        for completion in completions:
            length = len(completion.split())
            # 목표 길이 주변에서 최고 점수
            length_diff = abs(length - target_length)
            if length_diff <= 20:
                reward = 30.0 - (length_diff * 0.5)
            elif length_diff <= 50:
                reward = 20.0 - ((length_diff - 20) * 0.3)
            else:
                reward = max(0.0, 10.0 - ((length_diff - 50) * 0.1))
            rewards.append(reward)
        return rewards
    
    @staticmethod
    def evaluation_keywords_reward(completions: List[str], **kwargs) -> List[float]:
        """펀드평가 관련 키워드 보상 (0-40점)"""
        # 펀드평가 핵심 키워드 (step1 연구에서 도출)
        fund_keywords = [
            '수익률', '위험', '변동성', '샤프지수', '정보비율', 
            '추적오차', '벤치마크', '운용', '펀드', '투자',
            'return', 'risk', 'volatility', 'sharpe', 'fund'
        ]
        
        rewards = []
        for completion in completions:
            completion_lower = completion.lower()
            keyword_count = sum(1 for keyword in fund_keywords if keyword in completion_lower)
            # 키워드 개수에 따른 점수 (최대 40점)
            reward = min(40.0, keyword_count * 4.0)
            rewards.append(reward)
        return rewards
    
    @staticmethod
    def reasoning_structure_reward(completions: List[str], **kwargs) -> List[float]:
        """추론 구조 품질 보상 (0-20점) - DeepSeek R1 스타일"""
        rewards = []
        reasoning_indicators = ['분석', '결론', '평가', '판단', '근거', '이유', 'analysis', 'conclusion']
        
        for completion in completions:
            completion_lower = completion.lower()
            structure_score = sum(2.5 for indicator in reasoning_indicators if indicator in completion_lower)
            reward = min(20.0, structure_score)
            rewards.append(reward)
        return rewards

def combined_reward_function(completions: List[str], **kwargs) -> List[float]:
    """통합 보상 함수 (총 100점 만점)"""
    # 각 보상 함수 실행
    json_rewards = FundEvaluationRewardFunctions.json_format_reward(completions, **kwargs)
    length_rewards = FundEvaluationRewardFunctions.length_quality_reward(completions, **kwargs)
    keyword_rewards = FundEvaluationRewardFunctions.evaluation_keywords_reward(completions, **kwargs)
    reasoning_rewards = FundEvaluationRewardFunctions.reasoning_structure_reward(completions, **kwargs)
    
    # 통합 점수 계산
    total_rewards = []
    for i in range(len(completions)):
        total_score = (json_rewards[i] + length_rewards[i] + 
                      keyword_rewards[i] + reasoning_rewards[i])
        total_rewards.append(total_score)
    
    logger.info(f"보상 점수 분포: 평균={np.mean(total_rewards):.2f}, "
                f"최고={max(total_rewards):.2f}, 최저={min(total_rewards):.2f}")
    
    return total_rewards

def create_fund_evaluation_dataset(num_samples: int = 100) -> Dataset:
    """펀드평가 실험용 데이터셋 생성"""
    
    # 실제 펀드평가 시나리오 프롬프트
    prompts = [
        "다음 펀드의 투자 매력도를 JSON 형식으로 평가해주세요: 삼성 글로벌 성장주 펀드 (3년 수익률: 12.5%, 변동성: 15.2%)",
        "아래 펀드의 위험도를 분석하고 JSON으로 결과를 제시하세요: 미래에셋 신흥국 채권 펀드 (샤프지수: 0.85, 최대낙폭: -8.3%)",
        "다음 정보를 바탕으로 펀드 추천 여부를 JSON 형식으로 답변하세요: KB 국내 가치주 펀드 (정보비율: 0.45, 추적오차: 3.2%)",
        "이 펀드의 운용 성과를 평가하고 JSON으로 요약하세요: 한국투자 글로벌 배당주 펀드 (연평균 수익률: 8.7%, 벤치마크 대비 초과수익: +2.1%)",
        "다음 펀드의 투자 리스크를 분석하여 JSON 형식으로 보고하세요: 신한 테크놀로지 펀드 (베타: 1.25, 상관계수: 0.89)"
    ]
    
    # 샘플 수만큼 프롬프트 반복 및 확장
    extended_prompts = []
    for i in range(num_samples):
        base_prompt = prompts[i % len(prompts)]
        # 다양성을 위한 변형
        if i >= len(prompts):
            variation = f" (케이스 {i//len(prompts)+1})"
            base_prompt = base_prompt.replace("펀드", f"펀드{variation}")
        extended_prompts.append(base_prompt)
    
    dataset_dict = {"prompt": extended_prompts}
    dataset = Dataset.from_dict(dataset_dict)
    
    logger.info(f"펀드평가 데이터셋 생성 완료: {len(dataset)}개 샘플")
    return dataset

def setup_model_and_tokenizer(config: ExperimentConfig):
    """모델과 토크나이저 설정"""
    logger.info(f"모델 로딩 시작: {config.model_name}")
    
    # 토크나이저 로딩
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 모델 로딩 (메모리 효율성을 위한 설정)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    logger.info(f"모델 로딩 완료. 파라미터 수: {model.num_parameters():,}")
    return model, tokenizer

def run_grpo_experiment(config: ExperimentConfig):
    """GRPO 실험 실행"""
    logger.info("=== Step 2: GRPO 기본 예제 실험 시작 ===")
    
    # 시드 설정
    set_seed(config.seed)
    
    # 모델 및 토크나이저 설정
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # 데이터셋 생성
    train_dataset = create_fund_evaluation_dataset(config.max_samples)
    
    # GRPO 설정 (step1 연구 결과 기반)
    grpo_config = GRPOConfig(
        output_dir=config.output_dir,
        num_generations=config.num_generations,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        learning_rate=config.learning_rate,
        kl_penalty=config.kl_penalty,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        report_to=None,  # 로깅 비활성화 (실험용)
        remove_unused_columns=False,
    )
    
    logger.info("GRPO 설정 완료:")
    logger.info(f"  - 그룹 크기: {config.num_generations}")
    logger.info(f"  - 학습률: {config.learning_rate}")
    logger.info(f"  - KL 페널티: {config.kl_penalty}")
    
    # GRPOTrainer 초기화
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=combined_reward_function,
    )
    
    logger.info("GRPOTrainer 초기화 완료")
    
    # 훈련 전 샘플 생성 테스트
    logger.info("=== 훈련 전 모델 출력 샘플 ===")
    test_prompt = "삼성 글로벌 성장주 펀드 (3년 수익률: 12.5%)를 JSON 형식으로 평가하세요:"
    
    inputs = tokenizer(test_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # 샘플 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            num_return_sequences=3,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # 결과 출력
    for i, output in enumerate(outputs):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        response = generated_text[len(test_prompt):].strip()
        logger.info(f"훈련 전 샘플 {i+1}: {response[:100]}...")
    
    # GRPO 훈련 실행
    logger.info("=== GRPO 훈련 시작 ===")
    try:
        trainer.train()
        logger.info("GRPO 훈련 완료")
    except Exception as e:
        logger.error(f"훈련 중 오류 발생: {e}")
        raise
    
    # 모델 저장
    trainer.save_model()
    logger.info(f"모델 저장 완료: {config.output_dir}")
    
    # 훈련 후 성능 테스트
    logger.info("=== 훈련 후 모델 출력 샘플 ===")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            num_return_sequences=3,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # 결과 비교
    for i, output in enumerate(outputs):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        response = generated_text[len(test_prompt):].strip()
        logger.info(f"훈련 후 샘플 {i+1}: {response[:100]}...")
    
    return trainer

def analyze_grpo_performance(trainer):
    """GRPO 성능 분석"""
    logger.info("=== GRPO 성능 분석 ===")
    
    # 훈련 로그 분석
    if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
        logs = trainer.state.log_history
        
        # 손실 변화 분석
        train_losses = [log.get('train_loss') for log in logs if 'train_loss' in log]
        if train_losses:
            logger.info(f"훈련 손실 변화: 시작={train_losses[0]:.4f}, 끝={train_losses[-1]:.4f}")
            logger.info(f"손실 감소: {train_losses[0] - train_losses[-1]:.4f}")
        
        # 보상 분석
        rewards = [log.get('train_reward') for log in logs if 'train_reward' in log]
        if rewards:
            logger.info(f"보상 변화: 평균={np.mean(rewards):.2f}, 최대={max(rewards):.2f}")
    
    logger.info("성능 분석 완료")

def main():
    """메인 실행 함수"""
    # 실험 설정
    config = ExperimentConfig()
    
    logger.info("Step 2: GRPO 기본 예제 구현 및 검증")
    logger.info("="*50)
    logger.info(f"실험 시작 시간: {datetime.now()}")
    logger.info(f"사용 모델: {config.model_name}")
    logger.info(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # GRPO 실험 실행
        trainer = run_grpo_experiment(config)
        
        # 성능 분석
        analyze_grpo_performance(trainer)
        
        logger.info("실험 성공적으로 완료!")
        logger.info(f"결과 저장 위치: {config.output_dir}")
        
    except Exception as e:
        logger.error(f"실험 실행 중 오류: {e}")
        raise
    
    finally:
        logger.info(f"실험 종료 시간: {datetime.now()}")

if __name__ == "__main__":
    main()
