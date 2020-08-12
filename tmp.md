#### Intrinsic Motivation

针对于稀疏奖励问题，往往需要动作序列满足特定的要求才可以获得非零奖励，这使得传统强化学习算法难以求解。现有主流的一种方法是自己生成内在奖励，这类算法可以建模为

$$\pi_t^* = \max_{\pi} \mathbb{E}_{s_i,a_i\sim \pi} \sum_{i=0}^{\infty} \gamma^i (r(s_i,a_i) + \eta r_{M_t} (s_i,a_i))$$

其中$M_t$代表环境模型，$r_{M_t}:S\times A \to \mathbb{R}$ 为一族衡量状态动作对新颖程度的函数。

理论分析的点：

$$\lim_{t\to\infty} M_t = M$$

$$\lim_{t\to\infty} r_{M_t}(s,a) = 0 $$ for all $s,a$

$$\lim_{t\to\infty}\pi_{t}^* = \pi^*$$

**待调研：时变MDP的分析**

#### Novelty Loss

想法一：由于novelty 无法即时更新，所以对经验回放池中的状态动作对加入虚拟损失。

$$\max_{\pi} \mathbb{E}_{s_i,a_i\sim \pi} \sum_{i=0}^{\infty} \gamma^i (r(s_i,a_i) + \eta r_{M_t} (s_i,a_i))-\beta f(s_i,a_i)$$

其中$f(s,a)$为根据经验回放池取得的(s,a) 处的novelty loss。我们希望函数族$f:S\times A\to \mathbb{R}$具备的性质：
- 函数f是计算高效的
- 函数f与



