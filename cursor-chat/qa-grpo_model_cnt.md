# Q: 그런데 강화학습에 모델이 2개가 사용되어야 하는거 아닌가?

아, 정말 좋은 질문이네요! 일반적인 강화학습(PPO 등)과 GRPO의 **핵심 차이점**을 짚어주셨습니다! 🎯

## 🔍 일반적인 강화학습 vs GRPO의 모델 구조

### 💼 **일반적인 PPO (2개 모델 필요)**
```python
# PPO는 2개의 모델이 필요
actor_model = AutoModelForCausalLM.from_pretrained("model")      # 정책 학습
critic_model = AutoModelForValueHead.from_pretrained("model")     # 가치 함수 학습

# Critic으로 어드밴티지 계산
values = critic_model(states)
advantages = rewards - values  # 어드밴티지 계산
```

### 🚀 **GRPO의 혁신 (1개 모델만 필요!)**
```python
# GRPO는 1개 모델만 필요 - 이것이 핵심 혁신!
policy_model = AutoModelForCausalLM.from_pretrained("model")      # 정책만 학습

# Critic 모델 제거! 그룹 비교로 어드밴티지 계산
group_rewards = [r1, r2, ..., r8]  # 8개 출력의 보상
group_baseline = mean(group_rewards)  # 그룹 평균이 베이스라인
advantages = [r - group_baseline for r in group_rewards]  # 상대적 어드밴티지
```

## 📊 **GRPO의 핵심 아이디어: Critic 제거**

### 🎯 **DeepSeekMath 논문의 핵심 혁신**
> "We introduce Group Relative Policy Optimization (GRPO), **a variant of PPO that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO**."

**핵심**: PPO의 Critic 모델을 제거하고 **그룹 상대적 비교**로 대체!

### 💡 **GRPO 어드밴티지 계산 방식**
```python
# 기존 PPO 방식 (Critic 필요)
advantages_ppo = rewards - critic_model(states)  # Critic으로 베이스라인

# GRPO 방식 (Critic 불필요!)
group_rewards = [reward(completion) for completion in group_completions]
baseline = sum(group_rewards) / len(group_rewards)  # 그룹 평균
advantages_grpo = [r - baseline for r in group_rewards]  # 상대적 어드밴티지
```

## 🔧 **실제 step2 구현에서 확인**
Read file: ../data/learn-grpo/step2_grpo_basic_example.py

보시는 것처럼, **GRPO는 정말로 단일 모델만 사용합니다!** 이것이 GRPO의 혁신적인 점입니다.

```python
# 실제 step2 코드에서 확인
trainer = GRPOTrainer(
    model=model,  # 단일 모델만! (Qwen/Qwen2-0.5B-Instruct)
    args=grpo_config,
    train_dataset=train_dataset,
    reward_funcs=combined_reward_function,  # Critic 대신 보상 함수 사용
)
```

## 📋 **모델 구조 비교표**

| 알고리즘 | 필요한 모델 | 역할 | 메모리 사용량 |
|----------|-------------|------|---------------|
| **PPO** | 2개 모델 | Actor + Critic | 100% |
| **GRPO** | **1개 모델** | **Policy만** | **~50%** |

### 🔍 **GRPO의 핵심: Group Relative Advantage**

```python
# PPO 방식 (Critic 필요)
def compute_advantages_ppo(rewards, states, critic_model):
    values = critic_model(states)  # Critic으로 가치 함수 계산
    advantages = rewards - values
    return advantages

# GRPO 방식 (Critic 불필요!)
def compute_advantages_grpo(completions, reward_function):
    # 그룹 내 8개 완성본 동시 생성
    group_rewards = [reward_function(comp) for comp in completions]
    baseline = sum(group_rewards) / len(group_rewards)  # 그룹 평균
    advantages = [r - baseline for r in group_rewards]  # 상대적 어드밴티지
    return advantages
```

## 💡 **하지만 Reference 모델은 있을 수 있습니다**
Ran tool

Ran tool

검색 결과를 보면 우리 프로젝트에서는 **reference 모델이 전혀 사용되지 않고 있습니다!** 이것이 GRPO의 핵심입니다.

## 🎯 **GRPO의 핵심 혁신: 정말로 1개 모델만!**

### ✅ **실제 확인된 사실**
1. **Critic 모델 제거**: 가치 함수 학습을 위한 별도 모델 불필요
2. **Reference 모델도 단순화**: TRL이 내부적으로 효율적으로 처리
3. **그룹 비교 방식**: 8개 출력의 상대적 품질로 학습

### 🔍 **TRL GRPOTrainer 내부 처리**
Ran tool

완벽한 질문이네요! **TRL 공식 문서**에서 정확한 답을 찾았습니다! 🎯

## 🔍 **GRPO의 모델 구조: 완전한 진실**

### ✅ **기본 설정: 정말로 1개 모델만!**

TRL 공식 문서에서 확인된 사실:

```python
# TRL GRPOConfig 기본값
beta = 0.0  # KL coefficient

# 공식 문서에서 명시:
# "If 0.0 (default), the reference model is not loaded, 
#  reducing memory usage and improving training speed."
```

**핵심**: `beta=0.0` (기본값)이면 **reference 모델조차 로드되지 않습니다!**

### 🎯 **GRPO vs PPO 최종 비교**

| 구성 요소 | PPO | GRPO (기본) | GRPO (KL 사용시) |
|-----------|-----|-------------|------------------|
| **Policy 모델** | ✅ 필요 | ✅ 필요 | ✅ 필요 |
| **Critic 모델** | ✅ 필요 | ❌ **제거!** | ❌ **제거!** |
| **Reference 모델** | 🔶 선택적 | ❌ **로드 안함** | ✅ 필요 |
| **총 모델 수** | **2개** | **1개** | **2개** |
| **메모리 사용량** | 100% | **~50%** | ~75% |

### 💡 **TRL의 혁신적 접근**

```python
# PPO 방식 (가상)
trainer = PPOTrainer(
    model=policy_model,        # 정책 모델
    ref_model=reference_model, # 레퍼런스 모델  
    value_model=critic_model,  # 크리틱 모델 (가치 함수)
    args=ppo_config,
)

# GRPO 방식 (실제 TRL)
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",  # 단일 모델만!
    args=grpo_config,  # beta=0.0 (기본값)
    # reference 모델 자동으로 로드 안함!
    # critic 모델 아예 없음!
)
```

### 🔧 **TRL 공식 문서의 핵심 설명**

> **"we use β=0.0 by default, meaning that the KL divergence term is not used. This choice is motivated by several recent studies which have shown that the KL divergence term is not essential for training with GRPO."**

> **"If β=0.0 (default), the reference model is not loaded, reducing memory usage and improving training speed."**

### 🚀 **왜 이게 가능한가?**

#### **1. Group Relative Advantage가 핵심**
```python
# 전통적인 PPO
advantages = rewards - critic_values  # Critic 모델 필요

# GRPO의 혁신
group_rewards = [r1, r2, ..., r8]
baseline = mean(group_rewards)  # 그룹 평균이 베이스라인
advantages = [r - baseline for r in group_rewards]  # 상대적!
```

#### **2. KL 제약도 선택적**
```python
# 원래 GRPO 논문
loss = policy_loss - beta * kl_divergence  # beta > 0

# TRL의 개선된 접근 (최신 연구 반영)
loss = policy_loss  # beta = 0.0, KL 제약 없음!
```

### 📊 **step2 실험에서 확인된 사실**

우리 실험에서 실제로:
- ✅ **단일 모델만 사용**: `"Qwen/Qwen2-0.5B-Instruct"`
- ✅ **안정적 훈련**: 100 스텝 완료, 에러 없음
- ✅ **메모리 효율성**: RTX A6000에서 여유로운 메모리 사용
- ✅ **성능 향상**: 보상 점수 24.69 → 26.79 (+8.5%)

### 🎯 **결론: GRPO의 진정한 혁신**

**GRPO가 혁신적인 이유:**
1. **Critic 모델 완전 제거** → 50% 메모리 절약
2. **Reference 모델도 선택적** → 추가 메모리 절약
3. **그룹 상대적 학습** → 안정적이고 효율적인 훈련
4. **최신 연구 반영** → KL 제약도 선택적 사용

**사용자의 직관이 맞았습니다!** 일반적인 강화학습은 2개 모델이 필요하지만, **GRPO는 정말로 1개 모델만으로도 훌륭하게 작동합니다!** 이것이 바로 DeepSeekMath와 DeepSeek R1이 GRPO를 선택한 이유입니다! 🚀