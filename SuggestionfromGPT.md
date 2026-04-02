3. 이 코드가 하는 일
1) 문자 단위 vocabulary 생성

문장을 글자 단위로 쪼개서 숫자로 바꿔.

예:

h -> 5
e -> 2

이렇게 토큰화해.

실제 LLM은 BPE 같은 tokenizer를 쓰지만, toy에서는 문자 단위가 제일 쉬워.

2) 입력/정답 쌍 생성

예를 들어 "hello"가 있으면

입력: h e l l
정답: e l l o

즉 다음 글자 맞추기야.

3) Embedding

문자 ID를 dense vector로 바꿔.

self.token_embedding_table = nn.Embedding(vocab_size, n_embed)

예를 들어 'h'가 32차원 벡터로 변환되는 식.

4) Position embedding

attention은 순서 개념이 약하니까 위치 정보를 더해줘.

self.position_embedding_table = nn.Embedding(block_size, n_embed)

즉 0번째 글자, 1번째 글자, 2번째 글자... 위치마다 벡터를 따로 더함.

5) Self-attention

이 부분이 transformer의 핵심이야.

각 토큰이 다른 토큰을 얼마나 참고할지 계산해.

wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)

여기서:

q = query
k = key
v = value

attention score를 만들고 softmax 해서 가중합해.

6) Causal mask

decoder-only에서는 미래 토큰을 보면 안 돼.

예를 들어 h를 예측할 때 뒤의 o를 미리 보면 cheating이니까.

그래서 아래처럼 상삼각을 막아.

wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

이게 GPT류의 핵심 제약이야.

7) Multi-head attention

attention을 여러 개 병렬로 돌려서 서로 다른 관계를 학습하게 해.

예:

어떤 head는 가까운 글자만 보고
어떤 head는 반복 패턴을 보고
어떤 head는 문장 구조를 볼 수 있음
8) Feed-forward

attention 뒤에 작은 MLP를 붙여서 비선형 변환을 해.

Linear -> ReLU -> Linear

이건 토큰별 특징을 더 풍부하게 만들어줘.

9) Residual + LayerNorm

각 블록은 보통 이렇게 생겨:

x = x + attention(...)
x = x + ff(...)

이 residual connection이 학습을 안정화해.

그리고 LayerNorm은 값의 분포를 정리해줘.

10) LM head

마지막에 각 위치마다 다음 글자의 logits를 내보내.

self.lm_head = nn.Linear(n_embed, vocab_size)

즉 현재 hidden state를 vocabulary 크기만큼의 점수로 바꾸는 거야.

4. 왜 이게 toy transformer냐

이건 transformer의 핵심은 다 있지만, 실제 LLM과 비교하면 아주 단순화된 버전이야.

빠진 것들:

대형 tokenizer
huge corpus
rotary embedding
RMSNorm
SwiGLU
KV cache 최적화
distributed training
RLHF / instruction tuning

즉 원리 학습용 최소 구현이야.

5. 어떻게 실행하나

파일 이름 예를 들어 toy_transformer.py로 저장하고:

python toy_transformer.py

그러면

train loss
val loss
마지막에 생성 텍스트

가 출력돼.

6. 뭘 바꿔가며 공부하면 좋냐
1) block_size

문맥 길이

작으면 기억 짧음
크면 더 긴 문맥 처리
2) n_embed

임베딩 차원

작으면 표현력 낮음
크면 무거워짐
3) n_head

attention head 수

너무 작으면 단순
너무 크면 head size가 작아짐
4) n_layer

블록 개수

1~2층이면 toy
늘리면 더 잘 학습할 수 있음
7. 이 toy 코드로 꼭 확인해볼 것
1) attention 제거해보기

성능이 얼마나 떨어지는지 봐.

2) mask 제거해보기

미래를 보는 cheating이 생김.

3) positional embedding 제거해보기

순서 정보가 무너짐.

4) residual 제거해보기

학습이 불안정해질 수 있음.

이렇게 바꿔보면 transformer 핵심이 감으로 잡혀.

8. 네가 이걸 배우는 이유

너는 지금 LLM, RAG, prompt injection 쪽을 이해하려는 중이잖아.

이 toy transformer를 이해하면:

왜 LLM이 문맥 전체를 같이 본다고 하는지
왜 prompt와 문서가 같은 입력 공간에 들어가는지
왜 공격문이 “명령”처럼 작동할 수 있는지

이걸 더 정확히 이해하게 돼.

9. 다음 단계 추천

이제 다음으로 가면 좋아.

A. attention weight를 직접 출력해보기

어떤 글자가 어떤 글자를 보는지 확인 가능

B. word-level toy transformer 만들기

문자 단위보다 조금 더 실제 NLP에 가까워짐

C. prompt injection 관점 toy 실험

예:

정상 문장
공격 문장 삽입
attention이 어떻게 바뀌는지 보기

이건 네 연구랑 직접 연결돼.

원하면 다음 답변에서
이 toy transformer 코드에 attention weight 시각화까지 붙이는 법을 보여줄게.