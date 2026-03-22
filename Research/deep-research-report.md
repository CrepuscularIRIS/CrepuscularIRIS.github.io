# 深度本体论：把 depth 从“层号”变成“可寻址对象”的文献谱系、空白地形与可发题库（2015–2026）

## 研究基准：用《三维质疑法》判卷，用《科研操作系统》约束产出

你给的两份文档可以直接当作本次 DeepResearch 的“评分标准 + 交付规范”。

《三维质疑法》把架构质疑拆成三条主轴：**关系阶数**（pairwise/chain vs 高阶/集合）、**均匀性假设**（哪些维度被强制等权/等算力）、**可分离性假设**（哪些维度被硬拆开导致表达受限）。这三条刚好对应你要做的“深度本体论”：把 depth 从离散编号的链式结构，升级为可寻址、可度量、可约束的几何对象。fileciteturn0file1

《科研操作系统》提供了研究产出应当呈现的形态：**性价比模式 / 标准模式 / 极限模式**分流；强制 **MVE（最小可行验证）** 先行；持续做“魔鬼代言人式”反证，避免“大叙事、小结果”的伪创新。fileciteturn0file0

下面的分析将严格用这两份文档“判卷”。

## 问题本体判断：你是在修补 residual，还是在重写 depth 的建模对象

**关键结论**：从 2015–2026 的主线看，社区确实已经在持续“修补 residual”（门控、零初始化、跨层加权、跨层注意力、多流残差等），但真正把 **depth 当作坐标/记忆/守恒流形** 来建模的工作仍稀缺；更稀缺的是把这件事做到“可插拔、可诊断、可复现实验结论”的范式化框架。

为了让“本体判断”可操作，我把当前主流范式抽象成四条默认公理（对应三维质疑法的三轴）：

**默认公理一：depth 是离散层号（有序、可数、链式）**。ResNet 将深度组织为逐层堆叠，残差块形如学习残差函数、并通过 identity shortcut 让更深网络更易优化。citeturn0search0 Transformer 也以堆叠 block 形成深度，并依赖残差连接维持训练稳定。citeturn8search0  
*三维质疑法映射*：这是一种典型 **二阶/成对 + 链式** 的关系结构（每层主要与近邻发生结构性耦合）。fileciteturn0file1

**默认公理二：跨深度的信息聚合是“均匀”的（最典型：单位权重累加）**。现代 LLM 中常见的 PreNorm 残差形式，本质上沿 depth 方向累积各层输出；AttnRes 直接把这一点点名为“固定单位权重累积”，并指出会导致隐藏态随深度增长、稀释单层贡献。citeturn12search6turn1search3  
*三维质疑法映射*：这是典型的 **深度维度均匀性暴政**（默认每层“平等一票”）。fileciteturn0file1

**默认公理三：层间更新是局部一步（Markov 式），而不是全深度可检索/可积分的场**。ResNet/Transformer 的标准形态都把深度当作“逐层推进”的离散动力系统：信息要么被逐层变换，要么通过 shortcut 近似“跳过变换”。citeturn0search0turn8search0  
*三维质疑法映射*：关系阶数仍然偏低（局部链式），缺少“集合式、全局式”的深度交互。

**默认公理四：算力/步数在 depth 与 token 上近似均匀分配**。标准 Transformer 对所有 token 运行同样的层数；Mixture-of-Depths 明确把这一点作为问题陈述：Transformer 将 FLOPs 均匀铺在序列上，而模型可以学习对不同位置/不同深度动态分配计算。citeturn4search2turn4search18  
*三维质疑法映射*：这是**均匀性假设**在“计算预算维度”的直接体现。fileciteturn0file1

接下来回答你 A 部分要求的三个问题（以“判卷标准”语言表述）：

**这个方向是在修补 residual，还是在重写 depth 的建模对象？**  
从历史谱系看，大多数工作仍属于“修补 residual”：把 `x + f(x)` 的加法路径改得更可训练、更稳定、更高效（例如 Highway 的门控、ReZero 的零初始化标量门、DeepNorm 的残差缩放/初始化）。citeturn1search0turn0search1turn12search1turn8search3  
而“重写 depth 的建模对象”的代表主要集中在两类：把深度连续化的 **Neural ODE / CDE**（把层堆叠改为连续动力学求解），以及把深度隐式化为平衡点的 **DEQ**（把“层序列”改为“求 fixed point”）。citeturn2search2turn2search3turn3search0turn6search3

**当前主流论文中，哪些仍默认离散层、按层编号、局部更新、固定层数、均匀计算？**  
ResNet 与 Transformer 原典都以“固定层数的离散堆叠 + 残差稳定化”作为基本形态。citeturn0search0turn8search0 深层训练稳定化的经典路线（如 Pre-LN/LayerNorm 放置理论、DeepNorm/DeepNet）多数仍保留“离散层 + 链式推进”，只是改变了归一化与残差缩放来稳定训练。citeturn8search2turn12search1

**哪些论文已经开始松动这些假设？**  
松动点可以按“松动哪条默认公理”来分：

- 松动“均匀残差累加”：DenseFormer 用深度加权平均（DWA）显式对历史层做加权聚合；DeepCrossAttention 用输入相关权重组合历史层并引入深度向 cross-attention；Attention Residuals（AttnRes）则把“深度方向的聚合”直接注意力化（softmax over preceding layers），并给出 Block 版本降低代价。citeturn1search1turn9search4turn11view0turn12search6turn1search11
- 松动“单流链式拓扑”：Hyper-Connections 以多残差流 + 可学习混合替代单一残差流，并在 ICLR 2025 发表。citeturn10search9turn10search0
- 松动“固定计算步数/均匀算力”：ACT 让模型学习需要多少步；Universal Transformer 通过权重共享的循环式 refinement（并可配合动态停机）弱化“层=参数不同的一次性堆叠”；Mixture-of-Depths 对 token/层动态分配计算预算。citeturn4search0turn4search1turn13search10turn4search2
- 松动“深度必须离散”：Neural ODE / CDE 连续化深度；DEQ 把深度变为“隐式无限深”的平衡点。citeturn2search2turn2search3turn3search0

这就形成一个非常清晰的“空白靶心”：**大部分工作仍把 depth 当作“层号”或“步数”，而不是一个可寻址的几何对象（坐标/相位/流形/记忆场）**。你的优势在于：用“判卷标准”把它系统化，并用 MVE 先证伪伪创新。fileciteturn0file0turn0file1

## 研究谱系图：2015–2026 关键论文、它质疑了什么、又保留了什么

下面按时间线给出“最小但覆盖你清单”的谱系图。每篇都用同一模板：**一句话核心思想｜它质疑的默认公理｜它保留的旧假设**。

| 年份 | 论文（英文） | 一句话核心思想 | 质疑了哪个默认公理 | 仍保留哪些旧假设 | 官方来源 |
|---|---|---|---|---|---|
| 2015 | *Highway Networks* | 用门控单元调节信息在多层间的通行，缓解深网训练难题 | 残差/跳连不必固定相加；引入可学习“门”控制跨层流量 | depth 仍是离散有序层；更新仍是链式局部 | arXiv citeturn1search0 |
| 2015 | *Deep Residual Learning for Image Recognition* | 以残差学习 + identity shortcut 让极深网络更易优化 | “深度不可训练”这一工程瓶颈（用梯度高速公路解决） | depth=离散层号；残差聚合固定为加法；关系阶数主要仍是链式 | arXiv citeturn0search0 |
| 2015 | *Unitary Evolution Recurrent Neural Networks* | 用酉矩阵参数化使特征传播的谱半径为 1，缓解长程梯度爆炸/消失 | “传播可随意放大/缩小”可被约束为近守恒（至少在 RNN 递推中） | 主要是时间维/递归维工作；不直接给出 depth 记忆机制 | arXiv citeturn5search1 |
| 2016 | *Layer Normalization* | 用样本内统计做归一化，稳定深网（尤其 RNN）隐藏态动力学 | 训练稳定性不必依赖 batch 统计（BN） | 仍在现有离散层/时间步框架内稳定优化 | arXiv citeturn8search1 |
| 2016 | *Adaptive Computation Time (ACT)* | 让网络学习每个输入需要多少计算步（可微停机） | “固定步数/固定深度”不是必须 | 仍是离散步；更像“步数调度”而非 depth 最终本体重写 | arXiv citeturn4search0 |
| 2016 | *FractalNet: Ultra-Deep Neural Networks without Residuals* | 用分形式多路径结构实现超深网且不依赖 residual | “残差是超深训练的唯一道路”被证伪；强调训练可从浅到深过渡 | depth 仍然离散；结构虽多路径但多数仍是预定义拓扑 | arXiv citeturn0search2 |
| 2016 | *Deep Predictive Coding Networks (PredNet)* | 各层做局部预测，只向上传递预测误差（residual error） | 层间传递不必是 feature 本体，可以是 error | 主要在时序/视频；并未把 depth 变为可寻址记忆空间 | arXiv citeturn7search0 |
| 2017 | *Attention Is All You Need* | 用注意力替代递归/卷积建模序列，全局依赖并行化 | 主要质疑“序列必须递归处理” | depth 仍是离散堆叠；残差仍是固定加法稳定化的一部分 | arXiv citeturn8search0 |
| 2018 | *Neural Ordinary Differential Equations* | 把残差堆叠视为 ODE 的离散化，用求解器得到连续深度模型 | depth 不必是离散层数；可变为连续时间/深度 | 多数落点仍在“求解/内存/精度-速度权衡”，不自动得到可寻址深度记忆机制 | arXiv / NeurIPS citeturn2search2turn2search14 |
| 2018 | *Universal Transformers* | 通过权重共享的循环式 refinement 把“层”变成“迭代步骤”，可配合动态停机 | “每层参数不同 + 固定层数”不是必须；引入循环归纳偏置 | 仍是离散步；depth 仍更像“迭代次数”而非连续坐标或可检索记忆场 | arXiv / ICLR citeturn4search1turn13search10 |
| 2018–2019 | *Hypergraph Neural Networks (HGNN)* | 用超图编码高阶关系（超边连接多个节点） | 关系阶数不必停留在 pairwise 图/链式结构 | 主要在图结构数据；“超图思想”尚未系统迁移到 layer orchestration | arXiv / AAAI 提示citeturn5search2turn5search6 |
| 2019 | *Hamiltonian Neural Networks* | 用 Hamiltonian 结构把守恒律嵌入动力学学习，得到可逆/能量守恒性质 | “表示传播可任意耗散/注入能量”可被结构约束 | 主要在物理动力学；给出守恒/可逆的可微模板但未直接解决 Transformer 深度记忆 | arXiv citeturn5search0 |
| 2019 | *Deep Equilibrium Models (DEQ)* | 把“无限深网络”的极限表示为 fixed point，用隐式微分训练 | depth 的本体可变为“平衡点”而非层堆叠；隐式无限深 | 仍需选择求根器/收敛条件；不等价于“深度可寻址记忆”，更像“深度被消解” | arXiv / NeurIPS citeturn3search0turn3search12 |
| 2019 | *Recurrent Independent Mechanisms (RIMs)* | 多机制模块稀疏通信、仅在相关时更新，促进模块专门化 | 均匀更新假设被挑战：不是所有模块每步都更新 | 主要在时间递归与模块化；可作为“slot 化深度工作台”的跨域起点 | arXiv / ICLR citeturn3search3turn3search11 |
| 2020 | *ReZero is All You Need* | 用零初始化标量门让层初始近似 identity，极深网络更易收敛 | 残差不必固定强度；可学习并从 0 开始逐步“打开” | depth 仍是离散链；更偏优化稳定化而非新 depth 几何 | arXiv / PMLR(UAI) citeturn12search8turn12search0 |
| 2020 | *Neural Controlled Differential Equations* | 用受控微分方程扩展 Neural ODE，适配不规则采样并引入“控制项” | 连续动力学可被外部输入控制（比纯 ODE 更灵活） | 仍主要服务时间序列建模；“连续 depth 的读写与守恒”仍需另起框架 | arXiv citeturn2search3 |
| 2020 | *Object-Centric Learning with Slot Attention* | 通过竞争式注意力把表示压缩为少量可交换 slots（对象化表征） | “必须保留全部分布式特征”被挑战：可用少量对象槽承载信息 | 原任务是对象发现/集合预测；迁移到 depth-memory 仍是空白 | NeurIPS / arXiv citeturn13search2turn13search14 |
| 2021 | *RoFormer: Rotary Position Embedding (RoPE)* | 用旋转矩阵把位置信息编码进注意力的相对结构 | 位置编码可相位化/旋转化（提供“相位寻址”类比） | 讨论对象是序列位置，不是 network depth；仍需把“相位”迁移到 depth 坐标 | arXiv citeturn2search0 |
| 2021–2023 | *Neural Operator: Learning Maps Between Function Spaces* | 用积分算子+非线性构成“算子网络”，强调离散化不变性 | 映射对象可从有限维向量升级为函数空间上的算子；天然支持核/Green’s function 视角 | 主要面向 PDE 解算子；把 depth 当作连续域的“可积记忆场”仍未在 Transformer 主流中系统化 | arXiv / JMLR citeturn5search3turn5search11 |
| 2022 | *Perceiver IO* | 用固定大小 latent 数组承载信息，靠 query 读出任意结构输出 | “必须对所有 token 做全量交互”被挑战：先压缩到 latent workspace | depth 仍是离散堆叠；但提供“slot/latent 工作台”范式，可迁移到 depth memory | arXiv / ICLR citeturn13search1turn13search21 |
| 2022 | *DeepNet: Scaling Transformers to 1,000 Layers (DeepNorm)* | 用 DeepNorm 改造残差缩放与初始化稳定极深 Transformer | 极深训练可通过残差/归一化重参数化实现 | 仍在“离散层 + 链式堆叠”内；主要是稳定化路线 | arXiv citeturn12search1 |
| 2023 | *Mamba* | 通过输入依赖的选择性 SSM 增强长序列建模，同时保持线性时间 | “非注意力结构无法内容选择”被挑战：SSM 可做内容依赖选择/遗忘 | 关注的是序列长度维（time/position），不是 depth；但“选择性传播/守恒约束”可迁移到 depth 传播 | arXiv / GitHub citeturn4search3turn4search23 |
| 2024 | *Mixture-of-Depths* | 在每层对 token 做 top‑k 路由，实现“token 级 compute 预算” | 计算不必均匀分配；可在深度×序列维进行动态算力分配 | depth 仍离散；更偏算力-效果 tradeoff，而非把 depth 重写成可寻址几何对象 | arXiv citeturn4search2 |
| 2024 | *DenseFormer* | 每层后做 Depth‑Weighted Average（DWA）显式加权聚合历史表示 | 深度聚合不必等权；可学习“远层复用模式” | 仍以离散层为基本单位；DWA 多为参数化的加权平均而非可查询记忆 | arXiv / NeurIPS / GitHub citeturn9search1turn9search4turn9search2 |
| 2024 | *Transformers are SSMs* | 给出 Transformer 与 SSM 的结构对偶与算法视角 | 对“注意力 vs 状态空间”的范式边界提出统一观点 | 不是专门讨论 depth 本体，但为“守恒/对偶/算子视角”提供理论工具箱 | arXiv / ICML(PMLR) citeturn6search2turn6search6 |
| 2025 | *DeepCrossAttention (DCA)* | 用输入相关权重组合任意前层输出，并做 depth-wise cross-attention | 直接把“层输出聚合”从固定加法升级为可学习选择 | depth 仍离散；更多是一种强 cross-layer aggregation 模块 | arXiv / ICML(PMLR) citeturn0search7turn11view0 |
| 2025 | *Hyper-Connections* | 用多残差流 + 动态混合替代残差单流，缓解梯度消失与表示崩溃的权衡 | 残差拓扑不必是单一流；深度连接强度可学习并可“重排” | depth 仍离散；更像宏观拓扑重写，但未必把 depth 变为可寻址坐标 | arXiv / ICLR citeturn10search0turn10search9 |
| 2025 | *PoPE: Polar Coordinate Positional Embeddings* | 指出 RoPE 的 what/where 纠缠，提出极坐标解耦位置编码 | “相位编码=位置”可以更严格解耦（给 depth phase 的直接类比） | 目标仍是序列位置；尚未迁移到 depth coordinate | arXiv / OpenReview citeturn2search1turn2search5 |
| 2026 | *Attention Residuals (AttnRes)* | 用 softmax 注意力在深度方向聚合前序层输出，并提出 block 化降开销 | 直接质疑“残差等权累加”作为深度聚合公理；把 depth 维度注意力化 | depth 仍离散，但更接近“深度可检索记忆库”的范式；下一步是把 depth 从“索引集合”升级为“坐标/流形” | arXiv / GitHub citeturn12search6turn1search11 |

**时间线的结构性结论（从 2015 到 2026）**  
残差连接从“梯度高速公路”（ResNet/Highway）逐步演化为“深度方向的可学习聚合算子”（DenseFormer/DCA/AttnRes），并开始触碰《三维质疑法》里最关键的两刀：**打破深度均匀性**与**提升深度交互阶数**。citeturn1search1turn11view0turn12search6turn0file1  
但与此同时，绝大多数方法仍把 depth 当作“离散层集合”，而不是你真正想要的“可寻址几何对象（坐标/相位/流形/守恒场）”。正因为 AttnRes 等工作已经把“深度注意力化”做到工程可用，你下一步最有价值的空间反而更清晰：**把 depth 从集合索引升级为连续/相位/对象化的坐标系，并证明这不是 reparameterization trick。**citeturn12search6turn0file0

## 选题空白分析：真正还没被充分做过的“深度本体论”空白

下面给出若干“空白方向”，每个都按你要求标注：更适合哪种模式（性价比/标准/极限）、最接近工作、关键差异、潜在致命漏洞、MVE。

### 深度相位寻址从“类比”变成“可验证机制”

**空白是什么**：RoPE/PoPE 证明“相位/极坐标”能成为位置寻址机制，并强调 what/where 解耦的重要性；但目前几乎没有把这种“相位寻址”系统迁移到 **network depth** 上，让中间态成为“可相位对齐、可检索”的深度记忆库。citeturn2search0turn2search1

**最接近的已有工作**：AttnRes/DCA 已经在离散层集合上做“深度注意力聚合”。citeturn12search6turn11view0  
**关键差异**：它们的 depth key 仍隐式等同于“层号”，没有把 depth 变成连续坐标/相位坐标，也没有解决“内容—深度纠缠”这一在 PoPE 里被点名的结构性问题（只是对象从 position 变成 layer）。citeturn2search1turn12search6

**潜在致命漏洞**：很容易退化为“layer embedding 的花哨重参数化”，或者退化为“偏爱近层的门控”，而不是出现真正的远层共振/相位对齐。citeturn0file0turn0file1

**模式建议**：性价比模式（1–2 个月），因为可以做成可插拔模块并在合成任务快速证伪。fileciteturn0file0

**MVE（最低成本实验）**：  
固定一个小 Transformer（如 12–24 层 decoder-only），把每层输出绑定连续深度相位 θ(t)，读历史时使用“相位对齐检索”（例如 query 与 phase-aligned memory 的注意力），并记录 **near_vs_far_read_mass** 与 **phase_alignment_score** 是否在需要长程检索的任务上显著偏向远层，而不是坍缩到近邻层。你在文档里列的诊断字段（hidden/grad RMS、读熵等）非常契合这条线。fileciteturn0file0turn0file1

### 深度积分核与 Green’s function：从“跨层注意力”到“算子学习”

**空白是什么**：DenseFormer/DCA/AttnRes 都是在“离散层集合”上做加权；而 Neural Operator/FNO 提供了把映射表述为 **积分算子核** 的范式，并强调离散化不变性。真正的空白是：把 depth 视作连续域 t，把隐藏态视作 h(t)，直接学习核 K(t,s) 执行 **全深度积分式读写**，并要求跨不同层数离散仍稳定。citeturn5search3turn6search0turn9search4

**最接近的已有工作**：Neural Operator / FNO 的核积分与离散化不变性；以及 AttnRes 的深度注意力聚合。citeturn5search3turn6search0turn12search6  
**关键差异**：AttnRes 是 *attention over discrete layers*；Neural Operator 是 *integral operator over function space*。你要的“深度场”是把两者合并：**深度方向的算子学习**，并把“不随层数离散变化而失效”作为核心公理。citeturn5search3turn6search0

**潜在致命漏洞**：容易被 rebuttal 说成“又一个更贵的跨层 attention/MLP”，或者只是 smoothing，没产生真实检索能力；同时核学习若不加结构约束，可能数值不稳。citeturn0file0turn12search6

**模式建议**：极限模式（6–12+ 月）主线更合适；但必须先做小规模 MVE，否则容易“大叙事、小结果”。fileciteturn0file0

**MVE**：  
用少量 depth knots（如 8 个）做连续插值，学习一个低秩/傅里叶参数化的 K(t,s)，比较三者：标准 residual、AttnRes、积分核。重点测 **跨层数泛化**：同一模型在 12 层训练、测试时改为 24/48 层插值（或更细离散）是否保持行为一致（离散化不变性）。citeturn12search6turn5search3

### Slot 化深度工作台：把“全部历史层”压成“少量可查询对象”

**空白是什么**：Slot Attention/Perceiver IO 证明“先压缩到少量 slots/latents 再查询”能得到对象化工作台；但在 language/Transformer 深度方向，主流跨层方法仍然保留全部层输出（AttnRes 只是 block 化减少开销）。把 depth memory 直接对象化成 K 个 slots，并让每层只读写 slots，是一个仍未充分系统化的方向。citeturn13search2turn13search21turn12search6

**最接近的已有工作**：Perceiver IO 的 latent array + query readout；Slot Attention 的竞争式 slots；AttnRes 的 block 表示作为压缩。citeturn13search1turn13search14turn12search6  
**关键差异**：AttnRes 的 block 仍是“按层分块”的结构压缩，而不是“语义/任务自组织的对象槽”；Perceiver/Slot 是对象化，但主要面向输入/视觉对象，而非深度中间态的记忆管理。citeturn12search6turn13search14

**潜在致命漏洞**：slots 可能塌成“平均池化”，或被模型学成“近层缓存”；也可能因为 slots 太少导致不可逆信息丢失。fileciteturn0file0turn0file1

**模式建议**：性价比模式或标准模式（取决于你是否要扩到语言模型基准）。fileciteturn0file0

**MVE**：  
16 层模型只维护 4–8 个深度 slots；每层不再 attend 全历史，只对 slots 做读写；记录 slot_usage_entropy、slot 专门化程度，以及在 recall/associative retrieval 任务中远层信息是否能通过 slots 恢复。fileciteturn0file0

### 守恒/酉残差输运：把“深度传播”先约束成可控介质

**空白是什么**：unitary RNN 与 HNN 展示了“守恒/谱半径=1/可逆”带来的长程稳定性；但 Transformer/LLM 的 depth propagation 仍常被视为只要能训就行。若把 depth 看成传播介质，那传播算子是否应满足近守恒（norm/phase）是非常强的本体问题。citeturn5search1turn5search0turn0search0

**最接近的已有工作**：Unitary Evolution RNN（谱半径=1 的酉参数化）；Hamiltonian Neural Networks（守恒律嵌入）；以及 DeepNorm/DeepNet（用缩放稳定极深）。citeturn5search1turn5search0turn12search1  
**关键差异**：DeepNorm 是“稳定训练”；守恒/酉输运更像“先定义 **深度传播的物理边界**”，再在边界内做可读写机制（例如与 depth memory 结合）。citeturn5search0turn12search1

**潜在致命漏洞**：约束过强会锁死表达能力；或只提升稳定性但不提升信息读写能力——被 rebuttal 归为“优化/正则”。fileciteturn0file0

**模式建议**：性价比模式可先做“正交/酉混合 + 小规模任务诊断”；若要形成范式论文，通常需要标准模式验证。fileciteturn0file0

**MVE**：  
把 `h_{l+1}=h_l+f(h_l)` 改为“受约束的传播 + 注入”：`h_{l+1}=U_l h_l + g(h_l)`，U_l 受正交/酉约束；记录 hidden_rms_by_depth、grad_rms_by_depth、jacobian_spectral_proxy，并看在长依赖合成任务上是否优于仅做 DeepNorm/Pre-LN 稳定化。citeturn5search1turn12search1turn0file0

### 预测编码式残差：传递 error routing 而不是 feature routing

**空白是什么**：PredNet 明确采用“只上传预测误差”的层间通信机制；而 Transformer 的残差/跨层聚合几乎都是 feature routing。把 depth 通信改为 error routing，并把“误差能量稀疏化/可解释”作为设计目标，是跨域迁移的空白。citeturn7search0

**最接近的已有工作**：PredNet（误差传播）；AttnRes（深度选择性聚合）；ReZero/DeepNorm（稳定化）的组合可以作为对照组。citeturn7search0turn12search6turn12search0turn12search1  
**关键差异**：AttnRes 仍聚合 feature；预测编码把“可解释的误差信号”设为唯一上行通道，从而在本体层面回答“残差像不像脑/像不像误差修正”。citeturn7search0turn0file1

**潜在致命漏洞**：误差信号不一定对下游任务友好；也可能退化成“又一种归一化/门控”。fileciteturn0file0

**模式建议**：性价比模式（先合成任务+小 LM），因为结构改动可控。fileciteturn0file0

**MVE**：  
每层先预测下一层隐藏态，向后传递 residual error；记录 error_energy_by_depth、误差稀疏度，及其与远层检索性能的相关性。citeturn7search0turn0file0

### 超图式 Layer Orchestration：把“深度交互”从 pairwise 升级为 group-wise

**空白是什么**：HGNN 在数据结构上实现高阶关系；但在网络结构的 depth 维度，多数方法仍做“对所有前层的 pairwise 注意力权重”。真正的超图式 orchestration 是让某个 token 在某一刻由一组层集合共同解释，并让这个集合本身可学习/可诊断。citeturn5search2turn12search6

**最接近的已有工作**：HGNN 的高阶超边；Hyper-Connections 的多流拓扑；AttnRes/DCA 的跨层聚合。citeturn5search2turn10search0turn12search6turn11view0  
**关键差异**：AttnRes 仍是“对每个前层的权重”，更像高阶但仍成对；超图是“集合关系”先验（超边），并可让“层组合模板”成为可视化与可解释对象。citeturn5search2turn0file1

**潜在致命漏洞**：实现复杂度与工程成本过高，容易在 A 会被质疑“复杂换一点点”；也可能只是把 attention 结构换个写法。fileciteturn0file0

**模式建议**：标准模式更稳；但可以先性价比 MVE 做“block 级 hyperedge”来压成本。fileciteturn0file0

**MVE**：  
先在 block 级构建超边（每个 token 选择一个层集合），观察是否出现稳定的“语义→层集合模板”对应；用 depth_read_entropy 与模板熵衡量是否学到了结构而不是噪声。fileciteturn0file0turn0file1

## 候选题库：12 个可发的“深度本体论”研究题

以下每个 idea 都按你规定字段给出。为避免“看起来宏大但不可落地”，每个都强调 **可微近似结构 + MVE**，并标注是否容易滑向“只是优化 trick”。fileciteturn0file0

**Idea 1：Beyond Layer Indices: Rotary Depth Coordinates for Transformer Memory**  
题目草案：*Beyond Layer Indices: Rotary Depth Coordinates for Transformer Memory*  
核心公理质疑：depth 必须是整数层号吗？能否成为可对齐的连续相位坐标？  
借鉴学科：信号处理（相位编码）、几何表示  
可微近似结构：为每层输出附加 θ(t) 的旋转/相位编码；跨层检索时做 phase-aligned attention（不同于直接 layer embedding）  
最低成本实验设计：copy / associative recall / long-range retrieval；看 phase_alignment_score 与 near_vs_far_read_mass 是否在困难样本上显著偏向远层  
投稿 venue：ICLR / NeurIPS（偏机制创新与诊断）  
风险等级：中  
是否容易只是优化 trick：**是**（最大风险：退化为 layer embedding 重参数化）

**Idea 2：PoDE: Polar Depth Embeddings for Disentangled Content–Depth Addressing**  
核心公理质疑：跨层检索里“what（内容）/where（深度）”是否纠缠？  
借鉴学科：极坐标表示（PoPE 类比）、表示解耦  
结构：把深度相位与内容幅度解耦（PoPE 对 position 的做法迁移到 depth）citeturn2search1  
MVE：构造“只按深度索引/只按内容索引”的诊断任务（仿 PoPE 的诊断思想），验证解耦必要性  
venue：ICLR  
风险：中-高  
只是 trick：中（若诊断任务不成立会被认为是包装）

**Idea 3：GreenDepth: Learning Integral Operators over Continuous Transformer Depth**  
核心公理质疑：深度更新必须局部一步步走吗？能否是全深度的积分算子读写？  
借鉴：PDE/Green’s function、Neural Operator/FNOciteturn5search3turn6search0  
结构：学习 K(t,s)（低秩/傅里叶参数化）对 h(s) 做积分聚合得到 h(t) 的读写项  
MVE：12 层训练、在 24/48 层插值测试“离散化不变性”；对比 AttnRes（离散注意力）citeturn12search6  
venue：NeurIPS / ICML（算子学习+LLM 交叉）  
风险：高  
只是 trick：中（若证明不了离散化不变性，会被归为贵 attention）

**Idea 4：SlotDepth: Object-Centric Memory Slots Across Network Depth**  
核心公理质疑：历史层状态必须全量保留吗？能否压成少量“深度对象”供查询？  
借鉴：Slot Attention、Perceiver IOciteturn13search14turn13search21  
结构：维护 K 个 depth slots；每层只读写 slots，不直接 attend 全历史  
MVE：16 层→4–8 slots；测 slot_usage_entropy 与可恢复性  
venue：NeurIPS（结构化记忆/压缩）  
风险：中  
只是 trick：中（若 slots 实际只是池化/缓存）

**Idea 5：Phase Residual Transport: Unitary Depth Updates for Stable Sequence Models**  
核心公理质疑：深度传播介质是否应满足近守恒（norm/phase）？  
借鉴：Unitary RNN、Hamiltonian dynamicsciteturn5search1turn5search0  
结构：正交/酉混合 U_l + 可学习注入 g(h_l)（可与 AttnRes/slots 组合）  
MVE：监控 hidden/grad RMS、Jacobian 谱代理；对比 DeepNorm/Pre-LN 稳定化citeturn12search1turn8search2  
venue：ICLR  
风险：中  
只是 trick：**高风险**（容易被说“只是更稳”）

**Idea 6：Predictive Residual Coding: Error-Driven Communication Across Depth**  
核心公理质疑：层间传递的应是 feature 还是 prediction error？  
借鉴：PredNet / predictive codingciteturn7search0turn7search23  
结构：每层预测下一层隐藏态，仅传递误差；深度聚合在 error 空间进行  
MVE：误差能量稀疏度、远层依赖稳定性；对比 feature routing  
venue：NeurIPS（跨域生物启发+机制）  
风险：中-高  
只是 trick：中（若误差机制只是换了归一化）

**Idea 7：HyperDepth Orchestration: Dynamic Hypergraph Routing over Layers**  
核心公理质疑：层关系为什么默认 pairwise/chain，而不是组合作用（超边）？  
借鉴：HGNN（高阶关系）citeturn5search2turn5search6  
结构：token 动态选择层集合（超边）；集合聚合后再注入当前层  
MVE：层集合模板是否稳定、是否与语义/任务阶段相关  
venue：ICLR / NeurIPS  
风险：高  
只是 trick：中-高（若只是换一种 attention 写法会被打回）

**Idea 8：Budgeted Depth Memory: Token-wise Halting Meets Depth Retrieval**  
核心公理质疑：为什么每个 token 都要走同样深度？  
借鉴：ACT、Mixture-of-Depthsciteturn4search0turn4search2  
结构：token 学 halting/budget；只有“值得思考”的 token 才激活深度检索（phase/slots 任意一种做 memory）  
MVE：matched FLOPs 下的质量-速度曲线；看 halting_mass 与检索强度相关性  
venue：ICML（效率/条件计算）  
风险：中  
只是 trick：低-中（若严格 matched FLOPs/params，较难被归为炼丹）

**Idea 9：Discretization-Invariant Depth: Training at 12 Layers, Testing at 48**  
核心公理质疑：深度表征能否像算子一样跨离散化泛化？  
借鉴：Neural Operator 的离散化不变性思想citeturn5search3  
结构：把跨层聚合写成可插值的连续核/基函数展开  
MVE：跨层数泛化是主指标；对比 AttnRes/DCA（通常对层数敏感）citeturn12search6turn11view0  
venue：NeurIPS  
风险：高  
只是 trick：中（若跨层数泛化失败，就失去核心卖点）

**Idea 10：Equilibrium Depth Fields with Phase Constraints**  
核心公理质疑：深度是否应被消解为平衡点？平衡点是否可带相位/流形结构？  
借鉴：DEQciteturn3search0  
结构：在 DEQ fixed-point 中加入相位约束或守恒约束，形成“平衡流形”  
MVE：小模型先证明收敛与可控性，再谈性能  
venue：NeurIPS（隐式层/理论）  
风险：很高  
只是 trick：中（更容易被质疑“求解器花活”）

**Idea 11：Depthwise LSTM as Depth Memory Controller**  
核心公理质疑：跨层信息融合为何只能靠残差/attention，能否用门控控制器跨深度读写？  
借鉴：Highway（门控）、depth-wise LSTM（已有工作把 LSTM 用于跨深度连接 Transformer 层）citeturn1search0turn14search2  
结构：用轻量 depth-controller（LSTM/GRU）控制跨层读写（并可加入 phase/slots）  
MVE：看 controller 是否只偏爱近层；加入远层依赖任务  
venue：AAAI / EMNLP（若偏 NLP 应用）  
风险：中  
只是 trick：中（需要证明不是“更复杂的门控”）

**Idea 12：Depth Geometry Diagnostics Suite**  
核心公理质疑：社区缺少判定“新深度几何” vs “优化 trick”的统一诊断协议  
借鉴：你的科研操作系统强调强制 MVE 与诊断字段记录fileciteturn0file0  
结构：提出一套开源诊断：hidden/grad RMS by depth、read entropy、near/far mass、Jacobian proxy、matched compute 规范，并给出对 DenseFormer/DCA/AttnRes/HC 等的可复现对照（不宣称新 SOTA）citeturn9search4turn11view0turn10search9turn12search6  
MVE：用小模型+合成任务即可出结果  
venue：NeurIPS（Datasets & Benchmarks/工具化）或 TMLR  
风险：低-中  
只是 trick：低（工具论文更看重规范与复现）

## 严格文献表：官方/一手链接优先（arXiv / OpenReview / PMLR / NeurIPS / ICLR / 官方 GitHub）

表中“官方链接”不直接贴营销博客；优先 arXiv、OpenReview、PMLR、NeurIPS/ICLR 官方页面与官方 GitHub。

| 标题（英文） | 年份 | Venue | 作者 | 关键词 | 官方链接 |
|---|---:|---|---|---|---|
| Highway Networks | 2015 | arXiv | Srivastava, Greff, Schmidhuber | gated skip, very deep nets | arXiv citeturn1search0 |
| Deep Residual Learning for Image Recognition | 2015/2016 | arXiv / CVPR | He et al. | ResNet, residual, identity shortcut | arXiv citeturn0search0 |
| Unitary Evolution Recurrent Neural Networks | 2015 | arXiv | Arjovsky, Shah, Bengio | unitary/orthogonal, long-term gradients | arXiv citeturn5search1 |
| Layer Normalization | 2016 | arXiv | Ba, Kiros, Hinton | LayerNorm, training stability | arXiv citeturn8search1 |
| Adaptive Computation Time for Recurrent Neural Networks | 2016 | arXiv | Graves | adaptive compute, halting | arXiv citeturn4search0 |
| FractalNet: Ultra-Deep Neural Networks without Residuals | 2016 | arXiv | Larsson, Maire, Shakhnarovich | fractal, drop-path, no residual | arXiv citeturn0search2 / GitHub citeturn0search18 |
| Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning | 2016 | arXiv / ICLR(后续) | Lotter, Kreiman, Cox | predictive coding, error propagation | arXiv citeturn7search0 / GitHub citeturn7search1 |
| Attention Is All You Need | 2017 | arXiv (NIPS 2017) | Vaswani et al. | Transformer, attention, residual | arXiv citeturn8search0 |
| Universal Transformers | 2018 | arXiv / ICLR 2019 | Dehghani et al. | weight tying, recurrence over depth | arXiv citeturn4search1 / ICLR citeturn13search10 |
| Neural Ordinary Differential Equations | 2018 | arXiv / NeurIPS 2018 | Chen et al. | continuous depth, ODE solver | arXiv citeturn2search2 / NeurIPS citeturn2search14 |
| Hypergraph Neural Networks | 2018/2019 | arXiv / AAAI 2019 | Feng et al. | hypergraph, high-order relations | arXiv citeturn5search2 / GitHub citeturn5search6 |
| Hamiltonian Neural Networks | 2019 | arXiv / NeurIPS 2019 | Greydanus et al. | Hamiltonian, conservation, reversibility | arXiv citeturn5search0 / GitHub citeturn5search12 |
| Deep Equilibrium Models | 2019 | arXiv / NeurIPS 2019 | Bai, Kolter, Koltun | implicit depth, fixed point | arXiv citeturn3search0 / GitHub citeturn6search3 |
| Recurrent Independent Mechanisms | 2019 | arXiv / ICLR 2020 | Goyal et al. | modularity, sparse updates | arXiv citeturn3search3 / OpenReview citeturn3search11 |
| ReZero is all you need: fast convergence at large depth | 2020/2021 | arXiv / UAI 2021(PMLR) | Bachlechner et al. | zero-init residual gate | arXiv citeturn12search8 / PMLR citeturn12search0 |
| Neural Controlled Differential Equations for Irregular Time Series | 2020 | arXiv / NeurIPS 2020(Spotlight) | Kidger et al. | CDE, controlled dynamics | arXiv citeturn2search3 / GitHub citeturn2search26 |
| Object-Centric Learning with Slot Attention | 2020 | arXiv / NeurIPS 2020 | Locatello et al. | slots, object-centric | NeurIPS citeturn13search2 / arXiv citeturn13search14 |
| RoFormer: Enhanced Transformer with Rotary Position Embedding | 2021 | arXiv | Su et al. | RoPE, rotary, phase | arXiv citeturn2search0 / GitHub citeturn2search20 |
| Neural Operator: Learning Maps Between Function Spaces | 2021/2023 | arXiv / JMLR | Kovachki et al. | operator learning, integral kernel | arXiv citeturn5search3 / JMLR citeturn5search11 |
| Fourier Neural Operator for Parametric PDEs | 2020 | arXiv / OpenReview | Li et al. | FNO, Fourier kernel | arXiv citeturn6search0 / OpenReview citeturn6search8 |
| Perceiver IO: A General Architecture for Structured Inputs & Outputs | 2021/2022 | arXiv / ICLR 2022 | Jaegle et al. | latent array, query readout | arXiv citeturn13search1 / dblp(I CLR) citeturn13search21 |
| DeepNet: Scaling Transformers to 1,000 Layers | 2022 | arXiv | Wang et al. | DeepNorm, deep stability | arXiv citeturn12search1 |
| Mamba: Linear-Time Sequence Modeling with Selective State Spaces | 2023 | arXiv | Gu, Dao | selective SSM, long sequence | arXiv citeturn4search3 / GitHub citeturn4search23 |
| Mixture-of-Depths: Dynamically allocating compute in transformer LMs | 2024 | arXiv | Raposo et al. | token-wise routing, compute budget | arXiv citeturn4search2 |
| DenseFormer: Depth Weighted Averaging | 2024 | arXiv / NeurIPS 2024 | Pagliardini et al. | DWA, depth aggregation | NeurIPS citeturn9search4 / GitHub citeturn9search2 |
| Transformers are SSMs (Structured State Space Duality) | 2024 | arXiv / ICML 2024(PMLR) | Dao, Gu | duality, SSM vs attention | arXiv citeturn6search2 / PMLR citeturn6search6 |
| DeepCrossAttention: Supercharging Transformer Residual Connections | 2025 | arXiv / ICML 2025(PMLR) | Heddes et al. | cross-layer aggregation, depth-wise cross-attn | PMLR(ICML) citeturn11view0 / arXiv citeturn0search7 |
| Hyper-Connections | 2024/2025 | arXiv / ICLR 2025 | Zhu et al. | multi-stream residual, rearrange layers | arXiv citeturn10search0 / ICLR citeturn10search9 |
| PoPE: Polar Coordinate Positional Embeddings | 2025 | arXiv / OpenReview(未确认最终 venue) | Gopalakrishnan et al. | disentangle what/where, RoPE fix | arXiv citeturn2search1 / OpenReview citeturn2search5 |
| Attention Residuals | 2026 | arXiv | Kimi Team et al. | depth-wise attention over layers | arXiv citeturn12search6 / GitHub citeturn1search11 |

## 红队审计：最容易伪创新的方向、最易被 rebuttal 的攻击面、最值得先跑的 MVE

这部分严格对齐你文档里的“负科学”与“强制 MVE”精神：优先找**最可能自我欺骗**的地方，并给出可证伪的最小实验。fileciteturn0file0turn0file1

**最容易伪创新的三类方向（与你预警高度一致）**

第一类：**相位/坐标化最后只是 layer embedding 或重参数化**。如果你的 depth-phase 最终等价于“给每层加一个可学习向量再做注意力”，那很可能只是把 AttnRes 的 depth keys 换皮，而不是重写 depth 本体。AttnRes 已经把“深度注意力化”做成 drop-in baseline，因此你必须证明“相位坐标”带来的是 **可对齐/可插值/可泛化** 的新性质，而不是更强的自由度。citeturn12search6turn1search11

第二类：**连续深度最后只是 solver/隐式层技巧**。Neural ODE/CDE/DEQ 很强，但社区经常把贡献落在“常数内存、优雅求解、可变精度”等工程/数值层面。要避免伪创新，你需要把 continuous depth 的核心目标从“求解器”转成“depth 作为可寻址记忆场的读写与守恒”。否则很容易被评审一句话打回“这是 ODE/DEQ 的变体”。citeturn2search2turn3search0

第三类：**守恒/酉约束只带来更稳，不带来更强的读写能力**。Unitary/Hamiltonian 思想很美，但若你的收益仅是梯度更稳、loss 更平滑，而没有证据表明模型实现了新的信息组织方式（比如远层检索、slot 专门化、跨层离散化泛化），就会被归为“优化/正则化 trick”。citeturn5search1turn5search0

**最容易被 rebuttal 指出“只是增加参数/增加 FLOPs/改了优化”的攻击面**

- **参数/FLOPs 不匹配**：跨层注意力（AttnRes/DCA）类方法常被质疑“你只是给了模型更多路径与带宽”。你必须做 matched params、matched FLOPs、matched wallclock 的对照，且记录你文档里的 matched_flops / matched_params / matched_wallclock。citeturn12search6turn11view0turn0file0  
- **近层偏好伪装成远层检索**：很多跨层加权最终会塌成“只看最近 1–2 层”。DenseFormer/AttnRes 都强调远层复用或跨层选择，你的工作必须用 near_vs_far_read_mass、depth_read_entropy 等诊断证明“远层确实被用过”。citeturn9search4turn12search6turn0file0  
- **稳定化因素混淆**：DeepNorm/Pre-LN/LayerNorm 放置会显著影响训练稳定性。若你不控制这些变量，提升可能来自归一化/初始化而不是“深度几何”。citeturn8search2turn12search1turn8search1  

**最值得优先做 MVE 的三条（与你建议一致，但加上“判卷指标”）**

1) **深度相位寻址**：因为它是最有机会把 depth 从“层号”变成“坐标”的方向，同时成本低；MVE 关键是“出现远层相位对齐信号”。fileciteturn0file0  
2) **Slot 化深度记忆**：因为它最直接把 depth 变成“对象化工作台”，并且能用 slot 专门化/熵做可解释诊断。citeturn13search14turn0file0  
3) **守恒/酉残差传输**：因为它是“深度传播介质”的第一性原理注入，但必须强制用“信息读写能力”指标证明不是只更稳。citeturn5search1turn0file0  

如果这三条里任意一条通过 MVE，你后续就能像《科研操作系统》所说，从“找题”切换到“围绕新公理分叉量产题”。fileciteturn0file0