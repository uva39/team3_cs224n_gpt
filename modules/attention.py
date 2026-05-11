import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    # query와 key를 곱해서 각 토큰이 다른 토큰을 얼마나 참고할지에 대한 점수를 계산한다.
    attention_scores = torch.matmul(query, key.transpose(-1, -2))
    # 점수 값이 너무 커지지 않도록 head 차원 크기의 제곱근으로 나눠 스케일링한다.
    attention_scores = attention_scores / (self.attention_head_size ** 0.5)

    # 현재 입력의 시퀀스 길이를 가져와 causal mask 크기를 맞춘다.
    seq_len = query.size(-2)
    # 미래 토큰을 보지 못하게 하기 위해 아래 삼각 행렬 형태의 mask를 만든다.
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool))
    # 배치 차원과 head 차원에 자동으로 broadcast 되도록 차원을 두 개 추가한다.
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    # 미래 위치에 해당하는 점수는 softmax 이후 0이 되도록 매우 작은 값으로 바꾼다.
    attention_scores = attention_scores.masked_fill(~causal_mask, torch.finfo(attention_scores.dtype).min)

    # padding 토큰은 상위에서 전달된 attention_mask를 더해 추가로 가린다.
    attention_scores = attention_scores + attention_mask
    # 마지막 차원에 softmax를 적용해 attention 확률로 변환한다.
    attention_probs = torch.softmax(attention_scores, dim=-1)
    # attention 확률에 dropout을 적용해 과적합을 줄인다.
    attention_probs = self.dropout(attention_probs)

    # attention 확률로 value를 가중합해서 각 head의 문맥 벡터를 만든다.
    context = torch.matmul(attention_probs, value)
    # head 차원을 뒤로 보내서 나중에 다시 합치기 쉬운 형태로 바꾼다.
    context = rearrange(context, 'b h t d -> b t h d')
    # 여러 attention head를 이어 붙여 원래 hidden_size 차원으로 복원한다.
    context = rearrange(context, 'b t h d -> b t (h d)')
    # 최종 self-attention 결과를 반환한다.
    return context


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
