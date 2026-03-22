# **深度与残差机制的范式重构：第一性原理驱动的顶会选题侦察报告**

## **A. 问题本体判断：修补残差还是重写深度？**

在深度学习架构的演进历史中，“深度（Depth）”与“残差连接（Residual Connection）”一直被视为不可撼动的公理。自从ResNet提出以来，基于离散层堆叠和恒等映射的加法模型主导了从卷积神经网络到现代大语言模型（如Transformer）的基础拓扑结构 1。然而，随着模型参数量向千亿乃至万亿规模扩展，传统残差机制的几何与表征瓶颈日益凸显。当前前沿研究正处于一个关键的分水岭：一部分研究致力于在现有框架内“修补残差（Residual Tuning）”，而另一部分先锋工作则试图从第一性原理出发，彻底“重写深度的建模对象（Rewriting Depth Ontology）” 3。

通过对最新文献的深入剖析，可以明确界定这两种范式的本质区别。修补残差的工作，其核心逻辑依然承认网络是由一系列离散的操作块（Blocks）组成，优化的目标是改善这些操作块之间的信息流（Information Flow）。例如，Hyper-Connections（HC）拓宽了残差流的宽度，允许跨层的信息进行动态组合 4。然而，未受约束的宽残差流在极深网络中会导致严重的数值不稳定性，使得信号在传播过程中被放大数千倍，引发梯度爆炸 2。为此，DeepSeek团队提出的 mHC（Manifold-Constrained Hyper-Connections）通过Sinkhorn-Knopp算法将残差混合矩阵投影至Birkhoff多胞形（双随机矩阵流形）上，强制维持了恒等映射的守恒特性 2。类似地，DeepCrossAttention 6 与 Attention Residuals（Kimi, 2026）7 将固定的加法操作替换为基于注意力机制的历史层加权求和。这些工作虽然有效解决了深层网络中的“隐藏状态幅度无控增长”和“早期信息稀释”问题 7，但它们在拓扑学意义上并未跳出“离散层”的传统概念。

与此相对，重写深度的工作试图推翻“层”作为离散处理单元的必要性。例如，Neural Differential Manifold (NDM, 2025\) 提出将整个神经网络重新概念化为一个可微的黎曼流形（Riemannian Manifold），网络中的层被定义为该流形上的局部坐标图（Coordinate Chart），而网络参数则直接参数化了每一点的黎曼度量张量 8。这种视角的转换将深度的离散递进变为了流形上的连续几何演化。在图神经网络领域，Implicit Hypergraph Neural Networks (IHGNN, 2025\) 以及 Deep Equilibrium Models (DEQ) 完全抛弃了层间的前向传播，将深度的尽头定义为非线性算子的“隐式不动点（Implicit Equilibrium）” 10。而在偏微分方程（PDE）的神经求解器中，DyMixOp 框架将深度演化视为复杂动力系统中的积分算子，利用惯性流形理论，用全局与局部的积分核彻底替代了传统的残差块 12。

当前主流论文（约占顶会发表量的90%）仍然被锁死在“均匀性假设（Uniformity Assumption）”与“可分离性假设（Separability Assumption）”的暴政中 3。这些模型默认了五大隐形公理：第一，深度必须是离散且良序的整数编号（Integer Layer Indices）；第二，状态更新必须是局部马尔可夫的（Local Markovian Updates），即下一层仅依赖于上一层或简单的历史拼接；第三，网络的深度视界是固定且前置绑定的（Fixed Horizon），缺乏对不同难度样本的自适应性；第四，计算力在深度方向上是均匀分配的（Uniform Compute）；第五，序列维度与深度维度的信息交互是完全分离的。

然而，一系列具有跨域启发性质的论文已经开始剧烈松动这些旧有假设。在松动局部马尔可夫性与维度分离性方面，MoDA (Mixture-of-Depths Attention, 2026\) 提出序列维度与深度维度具有内在的一致性，通过统一的Softmax操作，使Query能够同时对序列的键值对（Sequence KV）与历史深度的键值对（Depth KV）进行动态混合检索 14。在松动固定深度与均匀计算方面，Mixture-of-Depths (MoD, 2024\) 引入了Top-k动态路由机制，允许Token在特定深度直接跳过计算，实现了预算的动态分配 16。在松动离散相加与离散坐标方面，除了前述的mHC与NDM，Hamiltonian Neural Networks (HNN) 将层演化重写为相空间（Phase Space）内的保体积辛积分 18，而Slot Attention 与 Perceiver IO 则通过引入信息瓶颈（Information Bottleneck），将原本的链式层间关系重构为基于对象（Object-centric）的潜在空间读写操作 20。这种从离散走向连续、从局部走向全局、从加法走向算子的趋势，构成了未来架构演进的核心主线。

## **B. 研究谱系图：2015–2026 深度与残差的演化史**

为了系统性地捕捉深度学习架构的底层演化逻辑，本分析以时间线梳理了2015年至2026年间的关键节点模型。这一演化史不仅是性能提升的记录，更是对神经网络第一性原理（从信息论、拓扑学到物理动力学）认知不断深化的过程。下表精准提取了每一项标志性工作的核心思想，并批判性地分析了其所质疑的旧有公理以及仍然受制于的传统假设。

| 标题/模型名称 | 年份 | 一句话核心思想 | 质疑的默认公理 (守恒破缺) | 仍然保留的旧假设 (演化局限) |
| :---- | :---- | :---- | :---- | :---- |
| **Highway Networks** | 2015 | 引入门控机制（Gating Mechanism）动态调节输入信息与非线性变换特征的融合比例。 | 质疑了网络层必须对输入进行无条件、强制性非线性变换的传统假设。 | 仍保留了离散的层级结构与局部的成对信息交互。 |
| **ResNet** | 2015 | 通过构建 ![][image1] 的恒等映射建立梯度高速公路，解决极深网络的优化退化问题 1。 | 质疑了“每一层都必须学习完整的端到端映射函数”的假设。 | 残差权重固定为1，受制于深度维度的“均匀暴政” 3。 |
| **FractalNet** | 2016 | 抛弃显式的跨层残差连接，利用不同深度的分形路径交互来实现极深网络的有效训练。 | 质疑了梯度流动必须依赖于单一、显式的直通路径（Shortcut）。 | 网络的拓扑结构在初始化时依然是静态且前置预设的。 |
| **Universal Transformer** | 2018 | 在序列处理引入自适应计算时间（ACT），对同一个Transformer块进行动态的循环迭代。 | 质疑了网络深度必须是固定常数，提出了依赖于输入难度的动态深度。 | 每一层的参数被强制完全共享（权重绑定），限制了深度特征的异质性。 |
| **Neural ODE** | 2018 | 将残差网络的层叠视为常微分方程的欧拉离散化，利用ODE求解器实现连续深度建模。 | 彻底质疑了“深度必须是离散的整数坐标”这一绝对公理。 | 将表示演化限制在同维度的连续流中，忽略了突变和多尺度拓扑交互。 |
| **DEQ (Deep Equilibrium)** | 2019 | 无论堆叠多少层，直接通过隐式求根算法寻找无穷深层网络的稳态隐式不动点。 | 质疑了特征表示必须随着层的叠加而线性展开的计算范式。 | 稳态（Equilibrium）假设过于严苛，抹杀了瞬态动力学的表达力。 |
| **Hamiltonian NN** | 2019 | 用神经网络参数化哈密顿量，确保系统在相空间的演化严格遵循能量守恒的辛几何结构 21。 | 质疑了特征演化是无约束的非线性耗散过程。 | 主要应用于已知的物理动力学系统，未推广至通用的自然语言或视觉表征。 |
| **Neural CDE** | 2020 | 将Neural ODE扩展为受控微分方程，允许连续时间模型响应不规则采样的外部输入序列 22。 | 质疑了连续深度模型只能处理平滑初值问题的局限。 | 依赖于高维插值路径，对离散的高频噪声或突变信号敏感。 |
| **ReZero / SkipInit** | 2020 | 使用初始化为零的可学习标量控制残差分支，以此消解对LayerNorm等归一化层的依赖。 | 质疑了深层残差网络必须依赖复杂的张量归一化技术才能稳定收敛。 | 仅仅是引入了全局的标量缩放，无法实现深层特征结构的细粒度重组。 |
| **Perceiver IO** | 2021 | 维护一个固定维度的潜在数组（Latent Array），通过交叉注意力迭代读取输入并输出结果 20。 | 质疑了网络内部计算的维度必须与输入数据的物理维度（如像素数）强绑定。 | 迭代过程依然是离散的步长更新，缺乏连续动力学的微分几何约束。 |
| **Slot Attention** | 2021 | 引入竞争性的Slot机制，强制网络学习解耦的、以对象为中心的（Object-centric）表征 20。 | 质疑了层与层之间传递的信息是密集的、未分化的纠缠特征张量。 | 迭代次数固定，且在处理超大规模序列时存在计算复杂度的瓶颈。 |
| **unitary RNN** | 2021 | 在复数域内通过参数化酉矩阵（Unitary Matrices）进行隐状态更新，保证特征范数在时序上严格不变 23。 | 质疑了实数域网络中非正交变换不可避免带来的梯度消失与爆炸。 | 复数张量优化的计算开销极大，且严格范数守恒限制了信息的遗忘机制。 |
| **RIMs** | 2021 | 构建多个独立的循环机制（Independent Mechanisms），通过注意力进行稀疏的模块间通信 24。 | 质疑了网络层是一个单一的、单体式的（Monolithic）密集计算单元。 | 模块的划分是人工预设的，缺乏数据驱动的内生动力学聚类。 |
| **PoPE / RoPE** | 2021 | 将绝对位置编码转化为复数域上的旋转相位（Rotational Phase），将位置信息融入特征的相对夹角中。 | 质疑了位置与深度信息必须通过显式向量相加或拼接的欧几里得几何方式。 | 仅解决了序列维度的平移不变性，未将相位流形思想推广至层深（Depth）维度。 |
| **Mamba** | 2023 | 将连续状态空间模型（SSM）与依赖于输入的动态硬件感知算法结合，实现线性复杂度的序列建模。 | 质疑了长序列全局依赖必须付出注意力机制的二次时间与空间复杂度。 | 其核心仍建立在系统控制论的离散化上，层级间的残差堆叠并未改变。 |
| **DenseFormer** | 2024 | 在每个标准Transformer块之后插入一个额外的步骤，对所有历史深度层的表示进行加权平均 25。 | 质疑了当前层只能与紧邻的上一层（二阶马尔可夫关系）进行直接信息交互 3。 | 深度加权平均（DWA）的权重是静态学习的，不具备对输入内容的动态感知。 |
| **Mixture-of-Depths** | 2024 | 利用Top-k路由机制动态决定特定Token在当前层是参与注意力/MLP计算还是直接无操作跳过 16。 | 质疑了每一层的庞大计算力必须在序列的所有Token上均匀分配的假设 17。 | 即使跳过计算，其残差流仍然是基于简单的离散加法，未触及深度的几何本质。 |
| **DeepCrossAttention** | 2025 | 引入基于输入内容的动态层间权重与深度方向的交叉注意力，实现对历史层的选择性聚焦 6。 | 质疑了残差相加的恒等系数应该独立于输入内容并保持不变的假设。 | 仍然依赖显式的层堆叠缓存，O(L)的时间复杂度约束并未被彻底消除。 |
| **mHC (DeepSeek)** | 2025 | 将多路宽残差流的混合矩阵通过Sinkhorn算法投影至双随机矩阵流形，防止深层信号爆炸 2。 | 质疑了宽泛的残差拓扑可以无约束地进行线性组合，强调了守恒的重要性。 | 尽管引入了流形约束，但其本质仍在修补离散层的交互，并非连续深度建模。 |
| **NDM** | 2025 | 将神经网络定义为可微的黎曼流形，各层为局部坐标图，参数直接用于生成度量张量 9。 | 彻底质疑了网络参数空间必须是平直的欧几里得空间，引入了曲率和测地线。 | 度量张量的动态生成计算复杂度极高，目前难以扩展至千亿参数的基座模型。 |
| **IHGNN** | 2025 | 将超图的信息传播表述为非线性单调算子的隐式不动点方程，无需堆叠固定数量的显式层 11。 | 质疑了捕捉高阶长程依赖必须依赖随着深度加深而可能导致过度平滑的深层架构。 | 隐式求根算法（如Anderson加速）在面对病态曲率的图数据时存在不收敛风险。 |
| **DyMixOp** | 2025 | 基于惯性流形理论，用局部-全局混合（LGM）的积分核彻底取代传统网络算子求解复杂PDE 13。 | 质疑了PDE的非线性动力学必须在无界高维空间进行线性化处理的传统假设。 | 架构设计主要针对流体等物理方程，其在NLP等离散语义空间的通用性待验证。 |
| **Attention Residuals** | 2026 | 用针对前序历史层的Softmax注意力操作彻底取代标准残差的加法累积，引入块级（Block）机制降本 7。 | 质疑了深度维度上的隐状态累积必须是无差别的，直击了PreNorm导致的信息稀释。 | 虽然使用了注意力，但在宏观拓扑上仍视深度为一个有向无环的因果链。 |
| **MoDA (ByteDance)** | 2026 | 统一序列与深度维度，使注意力机制的Query同时对序列的KV与历史深度的KV进行混合Softmax检索 15。 | 质疑了序列维度的全局注意力与深度维度的残差直连必须是完全分离设计的系统。 | 将深度KV纳入检索上下文会引发不可忽略的KV Cache显存随深度的二次增长。 |

## ---

**C. 选题空白分析：第一性原理下的破局路径**

通过将上述谱系图映射至“第一性原理”框架，我们可以清晰地发现，尽管诸如 DeepCrossAttention 6、Attention Residuals 26 和 MoDA 14 等最新工作取得了显著的经验性成功，但它们的本质仅仅是将“序列/空间”维度上已经高度成熟的注意力机制直接“平移”并借用到“深度”维度上。**真正的研究空白并未被充分填补：即打破“可分离性假设”，将物理学中严格的守恒定律、微分几何的连续流形以及控制论中的不动点反馈真正内化为深度的构建逻辑。**

以下针对三个极具潜力的真实空白方向进行深度分析，并根据资源风险矩阵匹配最合适的执行模式。

### **空白方向一：基于隐式高阶图的算子平衡（Operator Equilibrium）**

* **匹配模式：性价比模式（1–2 个月，模块型突破）**  
* **创新点：** 当前的跨层注意力模型（如 Attention Residuals）为了获取深层关系，必须在每个离散步中显式地计算与所有历史层的密集点积，这在极深网络中不仅计算冗余，而且割裂了层与层的全局拓扑。此方向的创新在于打破这种因果链式依赖，借鉴隐式超图神经网络（IHGNN）11 的思想，将“跨深度的相互参照”抽象为一个整体的超图拉普拉斯算子求根问题。网络的深层输出不再是一层层算出来的，而是将多个中间表示视为超图的节点，直接求解一个满足单调算子理论的非线性不动点方程。  
* **最接近的已有工作：** IHGNN 11 与 DeepCrossAttention 6。  
* **关键差异：** 现有的 IHGNN 主要用于处理外部图结构数据（如社交网络、引文网络）的节点分类 27，而此方向是将其“内化”，用于神经网络**内部层与层之间特征图**的融合。它将深度的“层间关系”从 Pairwise（成对的交叉注意力）降维为 Operator Equilibrium（算子平衡），大幅降低了计算图的内存占用。  
* **潜在致命漏洞：** 如果内部表示流形的局部曲率过高或李普希茨常数（Lipschitz constant）不受控，隐式求根求解器（如 Anderson Acceleration 或 Broyden 方法）极大概率会面临不收敛或陷入死循环的窘境。  
* **最小可行验证 (MVE)：** 抽取一个已预训练好的 12 层 BERT 模型，完全冻结其前 8 层主干。将其最后的 4 个 Transformer 块的残差相加融合逻辑，替换为一个轻量级的隐式高阶超图求解层。测试该融合模块在 GLUE 基准上的微调性能，若能在参数量更少、显存占用更低的情况下达到持平的精度，则 MVE 成功。

### **空白方向二：基于预测误差传播的微分反馈深度**

* **匹配模式：标准模式（3–6 个月，系统级增强）**  
* **创新点：** 当前深度学习的基石是全局的反向传播算法（Backpropagation），残差连接 ![][image2] 的核心物理意义仅仅是为梯度的回传提供一条“高速公路”以避免梯度消失 3。此方向从计算神经科学的预测编码（Predictive Coding）理论出发 28，彻底重写残差的物理意义：将其定义为“预测误差（Prediction Error）” ![][image3]。网络的训练不再依赖于需要缓存所有中间激活的全局链式法则，而是通过层与层之间局部误差节点的能量最小化（Local Energy Minimization）来隐式传递学习信号 30。  
* **最接近的已有工作：** Zero-divergence Inference Learning (Z-IL) 31 与 Meta-PCN 32。  
* **关键差异：** 现有的预测编码网络（PCN）大多局限于极浅层的 MLP 或 CNN 架构验证，一旦加深便会遭遇严重的误差累积与稳定性崩溃 32。此方向结合了现代 Transformer 的归一化优势，提出使用辅助阻尼神经元（Auxiliary neurons）和精度加权（Precision-weighted）来减缓残差连接中的能量爆炸 33，使其能够真正扩展到包含数百层的生成式预训练架构中。  
* **潜在致命漏洞：** 预测编码在推断阶段（Relaxation phase）需要多次前向-后向的微小迭代才能达到能量的局部极小值，这可能导致模型的前向推理延迟（Latency）成倍增加。在当下追求极致吞吐量的大模型工程界，这可能被视为致命的实用性缺陷。  
* **最小可行验证 (MVE)：** 在一个标准 ResNet-50 的瓶颈层残差连接处引入预测误差节点。使用局部的 Hebbian 学习律和能量最小化损失替代端到端的全局交叉熵反向传播。测量在 CIFAR-100 上的收敛轨迹以及显存消耗峰值（理论上应大幅降低），并验证最终精度是否能逼近 BP 算法。

### **空白方向三：相空间坐标与可微连续流形积分**

* **匹配模式：极限模式（6–12+ 个月，底层范式重建）**  
* **创新点：** 彻底废除“层”的离散编号概念。将网络的深层表示嵌入到一个严密的微分流形（Differentiable Manifold）中，网络的正向传播不再是函数的离散复合，而是在相空间（Phase Space）中进行的保体积辛积分（Symplectic Integration） 19 或者是黎曼流形上的测地线演化。通过引入辛结构（Symplectic Structure），保证了深度方向的信息（能量）零散度守恒 34，从而从根本的物理几何层面上消除了随深度加深而带来的信号衰减与表示崩塌。  
* **最接近的已有工作：** Hamiltonian Neural Networks (HNN) 18 与 Neural Differential Manifold (NDM) 8。  
* **关键差异：** 现有的 HNN 绝大多数仅用于拟合和预测已知的物理动力学系统（如双摆、天体力学等）的时间轨迹 21。而此方向的创举在于，将哈密顿方程或黎曼流形演化作为现代基座模型（如大语言模型或视觉模型）的**深层残差连接替代品**。即不再用它去预测外部物理世界，而是用物理规则去约束网络内部隐藏状态在“深度”维度的演化。  
* **潜在致命漏洞：** 如果采用辛积分器（如 Leapfrog），反向传播时展开的计算图极其深邃，可能会导致雅可比矩阵的条件数变得极为病态。此外，严格的能量守恒系统是一个无耗散的系统，这可能会限制网络通过过滤噪声来吸收新特征的表达力。  
* **最小可行验证 (MVE)：** 构建一个微型的序列生成模型（如百兆参数级），剥离所有传统的 Transformer 残差块。将其隐藏状态拆分为“广义动量”与“广义位置”，在深度演化上仅运行单步的辛积分。在 WikiText-103 上进行训练，重点观测前向传递过程中，随着虚拟层数增加到 1000 层，隐藏状态的 L2 范数是否能够保持严格的有界守恒，且不发生梯度消失。

## ---

**D. 候选题库：12 个范式重构 Idea**

基于第一性原理的推演，以下提供 12 个旨在“重写 Depth / Residual 本体”的深入课题。每个方案都附带了明确的学科借鉴、可微近似结构以及最小可行实验（MVE）设计。

### **1\. 题目草案：Symplectic Residuals: Hamiltonian Phase Space as a Differentiable Depth Operator**

* **核心公理质疑：** 质疑深度的单向、非守恒加法累积（这是导致 PreNorm 稀释的元凶），主张深度的前向传播应当是相空间中的保体积积分。  
* **借鉴学科：** 经典力学（辛几何与哈密顿动力学）21。  
* **可微近似结构：** 取消 ![][image2]。将隐藏状态张量均匀拆分为动量 ![][image4] 与位置 ![][image5]。利用交替更新的哈密顿算子替代传统残差：![][image6]；![][image7]，其中 ![][image8] 为参数化的神经网络。  
* **最低成本实验 (MVE)：** 在 20 层的字符级语言模型中替换残差块，不加任何归一化层，直接观测深层梯度流的方差是否优于同等规模的标准 ResNet。  
* **适合投稿：** ICLR / NeurIPS。  
* **风险等级：** 高（要求数学推导极度严密）。  
* **伪创新审计：** 必须证明严格辛结构带来了经验上的“超深度可训练性（如无需 LayerNorm 直接支持 1000 层）”，否则会被 reviewer 视为仅仅是“将通道拆分为两半交替更新的 trick”。

### **2\. 题目草案：Predictive Error Routing: Local Energy Minimization as a Substitute for Backprop-Residuals**

* **核心公理质疑：** 质疑残差仅仅是为了给梯度回传提供“高速公路”，主张残差的物理意义是局部的“预测误差（Prediction Error）”。  
* **借鉴学科：** 计算神经科学（Predictive Coding）28。  
* **可微近似结构：** 移除残差的全局反向传播，重写拓扑：每层维护状态节点 ![][image9] 和误差节点 ![][image10]。训练过程不再是从最后一层回传梯度，而是利用局部 Hebbian 学习律在各层内部独立进行能量函数 ![][image11] 的最小化。  
* **最低成本实验 (MVE)：** 在 CIFAR-10 上使用局部预测误差更新代替 BP 训练 VGG 模型，证明其不仅能收敛，且显存峰值（因无需保存全局激活）显著低于标准框架。  
* **适合投稿：** NeurIPS。  
* **风险等级：** 极高（突破传统 BP 的算力底座挑战极大）。  
* **伪创新审计：** 如果推理时依然需要多次迭代收敛，则易被批评为“用高昂的推理时间换取理论的优美”。

### **3\. 题目草案：Integral Kernel Depth: Continuous Feature Aggregation via Global Neural Operators**

* **核心公理质疑：** 质疑深度维度必须依赖局部层的一阶逐步递归，主张直接用连续的全局积分核一次性跨越任意深度的演化 12。  
* **借鉴学科：** 偏微分方程分析与算子理论（Neural Operator）13。  
* **可微近似结构：** 将离散的深度序号 ![][image12] 视作连续时间 ![][image13]。在频域内构建全局傅里叶积分核（Fourier Integral Kernel）来代替 Block AttnRes 中的分组加权操作，将 ![][image14] 的全层相互注意力计算转化为 ![][image15] 的频域算子相乘。  
* **最低成本实验 (MVE)：** 将预训练的 Vision Transformer (ViT) 的最后 6 层切断，替换为单层的傅里叶神经算子积分层，测试 ImageNet 的线性探测精度。  
* **适合投稿：** ICML / ICLR。  
* **风险等级：** 低（算子网络有成熟的 FNO/DyMixOp 代码库可直接迁移）。  
* **伪创新审计：** 需要证明算子不仅拟合了残差，还在分布外（OOD）深度插值上表现出连续性。

### **4\. 题目草案：Implicit Hypergraph Equilibrium: Operator Roots over Chain Residuals**

* **核心公理质疑：** 质疑 Attention Residuals 中使用 Pairwise（两两）点积检索深度信息的效率，主张深度的特征融合是高阶多体（多对多）的平衡态 11。  
* **借鉴学科：** 拓扑图论（Hypergraph Theory）与单调算子论。  
* **可微近似结构：** 不同深度的浅层输出作为超图的节点，动态生成超边（Hyperedges）聚集那些“语义互补”的层。利用 Anderson 加速器求取这一高阶层间图的不动点，直接替代多层的显式融合。  
* **最低成本实验 (MVE)：** 在节点分类任务中，用超边单步连接多层 GCN 的初始表示，对比标准 DenseNet 或 Skip-Connection 的性能与耗时。  
* **适合投稿：** KDD / AAAI。  
* **风险等级：** 高（隐式求根算法在非凸流形上极易不收敛）。  
* **伪创新审计：** 必须使用严格的 Iso-FLOPs 对比，证明超图求根的算力收益。

### **5\. 题目草案：Thermodynamic Mixture-of-Depths: Entropy-Guided Depth Condensation**

* **核心公理质疑：** 质疑 MoD 中依赖固定阈值进行 Top-k 层跳过的纯工程启发式设计 16，主张深度的计算预算分配应基于严密的热力学熵增（Entropy）原理。  
* **借鉴学科：** 信息论与非平衡态热力学。  
* **可微近似结构：** 实时监控每个 Token 隐藏状态的香农熵。将深度演化等效为降噪过程，当熵值梯度低于阈值（信息增益衰减极限）时，触发类似于 ACT（Adaptive Computation Time）的“早退（Early Exit）”机制，并引入最小描述长度（MDL）作为损失正则项。  
* **最低成本实验 (MVE)：** 冻结 Llama-3-8B 的主干，插入熵评估轻量探针，仅调整推断跳层熵阈值，测试保证 PPL 不降的前提下的最大推断加速比。  
* **适合投稿：** ACL / EMNLP。  
* **风险等级：** 低（工程实现极其明确，业务落地价值大）。  
* **伪创新审计：** 很容易被指责为“只是基于置信度的早退策略换了个热力学的名词”，必须推导信息瓶颈与熵阈值的严密数学联系。

### **6\. 题目草案：Holographic Phase Coordinates: Wave-Interference Layer Routing**

* **核心公理质疑：** 质疑层与层之间的特征传递是“粒子式”的点对点欧氏加法或标量混合，主张其本质可以拓展为波的干涉。  
* **借鉴学科：** 波动光学与数字信号处理。  
* **可微近似结构：** 引入复数域的相位坐标（Phase coordinates）。层间残差计算替换为全息干涉模型：![][image16]。利用相位的对齐与相消干涉（而非简单的门控标量）来决定深度特征的正交保留与擦除。  
* **最低成本实验 (MVE)：** 在极长时序预测任务中，使用复数张量重写 Transformer 的残差连接，验证相位干涉是否自然过滤了高频噪声。  
* **适合投稿：** ICLR。  
* **风险等级：** 中（复数网络优化较为困难）。  
* **伪创新审计：** 需证明相位坐标赋予了实数域加法无法表达的干涉消隐机制。

### **7\. 题目草案：Geodesic Depth: Charting the Neural Differential Manifold via Curvature Regularization**

* **核心公理质疑：** 质疑特征在深度的演化是在平直欧式空间中的线性叠加，主张其应沿着具有曲率约束的非欧流形测地线（Geodesic）移动 9。  
* **借鉴学科：** 微分几何与黎曼几何 8。  
* **可微近似结构：** 简化原始 NDM 的度量张量生成。通过归一化流（Normalizing Flows）对层间过渡进行可逆保测度变换，并显式添加标量曲率正则化（Scalar Curvature Penalty），惩罚内部流形的极端扭曲 36。  
* **最低成本实验 (MVE)：** 在自监督对比学习（如 SimCLR）的主干网络中注入黎曼层级变换，测试能否以更少的物理层数撑开特征表示空间，实现更好的线性可分性。  
* **适合投稿：** ICML。  
* **风险等级：** 中。  
* **伪创新审计：** 如果曲率惩罚退化为普通的 L2 正则化，则属于伪几何。

### **8\. 题目草案：Episodic Slot Depth: Routing Layers through Object-Centric Latents**

* **核心公理质疑：** 质疑每一层的全宽张量包含大量未解耦的混杂信息，主张深度的传递通道应当是严格解耦的离散对象（Slots） 20。  
* **借鉴学科：** 认知科学（全局工作区理论 / GWT）与对象神经表征。  
* **可微近似结构：** 彻底移除残差的直接通道。层与层之间设立固定维度的全局 Slots。第 ![][image12] 层的输出只能通过交叉注意力更新这些 Slots 的状态，而第 ![][image17] 层仅能从 Slots 中读取信息。使深度之间存在极致的信息瓶颈过滤。  
* **最低成本实验 (MVE)：** 在多对象推理数据集（如 CLEVR）上测试 Slot Depth 架构，提取层级间的 Slot 注意力图，检查其是否在极深层依然保持对独立物体的稳定跟踪。  
* **适合投稿：** ICLR。  
* **风险等级：** 中。  
* **伪创新审计：** 需要证明它不仅是 Perceiver IO 的简单堆叠，而是构建了随深度演化的对象记忆流。

### **9\. 题目草案：Orthostochastic Residual Streams via Newton-Schulz Iterations**

* **核心公理质疑：** 质疑 Hyper-Connections 中矩阵自由混合导致的范数爆炸 2，同时质疑 mHC 中复杂且昂贵的 Sinkhorn 投影（双随机约束不够严密），主张深度混合必须是纯正交投影。  
* **借鉴学科：** 矩阵分析论与李群李代数。  
* **可微近似结构：** 将宽残差流的混合矩阵更新策略，替换为仅需 3-5 步矩阵乘法的 Newton-Schulz 近似迭代，直接将深度的混合操作投影向正交化流形（Orthostochastic Manifold）。正交矩阵不仅保证了列与行的总和一致，更严密地保持了特征演化的能量等距同构（Isometry）。  
* **最低成本实验 (MVE)：** 在微型 GPT 架构上（如 nanoGPT），验证采用 Newton-Schulz 投影的前向计算在深度达到 64 层时，隐藏状态方差是否比原始 HC 甚至 mHC 更稳定。  
* **适合投稿：** ICLR / NeurIPS。  
* **风险等级：** 低（基于已有 mHC 代码极易实现改进）。  
* **伪创新审计：** 如果没有深入推导正交投影在保持残差特征各向同性上的优势，极易被认为是“换了个快速计算公式的微调”。

### **10\. 题目草案：Diffusion Denoising as Layer Evolution: Stochastic Differential Depth**

* **核心公理质疑：** 质疑深度的决定性确切映射（Deterministic Mapping），提出网络深度的推进本质上应当是对输入特征的一阶随机去噪（Langevin Dynamics）。  
* **借鉴学科：** 随机偏微分方程（SDE）与控制理论。  
* **可微近似结构：** 将残差的递推重构为 SDE 的欧拉-丸山离散化：![][image18]，其中 ![][image19] 为布朗运动噪声。训练时不仅优化分类/生成损失，还要使 ![][image20] 最大化局部互信息并压缩冗余。  
* **最低成本实验 (MVE)：** 在深度 ResNet 中引入这种布朗层间演化，评估其对对抗样本攻击（Adversarial Attacks）的防御能力（随机深度破坏了攻击者的梯度估算）。  
* **适合投稿：** CVPR / ICML。  
* **风险等级：** 中。  
* **伪创新审计：** 与 Dropout 存在相似性，必须从 SDE 的概率分布演化角度证明这是深度的连续化建模。

### **11\. 题目草案：Positional Phase Manifolds: Baking Depth into Rotational Kinematics (RoPE Depth)**

* **核心公理质疑：** 质疑每一层都需要独立的权重矩阵和层编号索引，主张深度的演化如同序列位置一样，可以编码为复数空间上的绝对相位旋转（Rotational Position Encoding in Depth）。  
* **借鉴学科：** 量子力学与信号调变（Unitary Evolution）37。  
* **可微近似结构：** 受 RoPE 启发，将层编号 ![][image12] 直接编码为特征向量复数表示的旋转角度 ![][image21]。使用参数完全共享的单一算子（Universal Transformer 风格），仅通过输入特征在复平面的旋转偏移 ![][image21] 来指示其目前处于深度的哪一阶段，从而驱动层级间的逻辑递进。  
* **最低成本实验 (MVE)：** 训练一个 12 层权重完全共享的 Transformer，分别使用标准残差和 RoPE Depth 编码层信息，比较其在逻辑推理数据集（如 GSM8K）上的逐层泛化能力。  
* **适合投稿：** ICLR / ACL。  
* **风险等级：** 中。  
* **伪创新审计：** 如果只是把 RoPE 的公式照搬给层索引，很容易被拒。需深入探讨相位演化与特征语义提取深度的数学同构性。

### **12\. 题目草案：Fractional-Order Residuals: Long-Memory Deterministic Depth Evolution**

* **核心公理质疑：** 质疑当前连续深度模型（Neural ODE）仅使用一阶导数离散化，主张大语言模型的深层逻辑具有“极长程记忆”特性，其演化应当遵循分数阶动力学。  
* **借鉴学科：** 分数阶微积分（Fractional Calculus）与反常扩散理论。  
* **可微近似结构：** 引入 Caputo 分数阶微积分的离散化形式：当前层的状态不只由前一层累加，而是与所有历史深度发生确定性的幂律衰减（Power-law decay）交互：![][image22]。这是一个完全无需额外参数学习就能实现类“Attention Residuals”功能的强大基线。  
* **最低成本实验 (MVE)：** 零新增加参数。在 Long Range Arena (LRA) 任务的 Transformer 模型中修改加法规则为分数阶求和，验证其梯度流穿透极深网络的效果。  
* **适合投稿：** AAAI / NeurIPS。  
* **风险等级：** 低（实现极简，理论优美且不可反驳）。  
* **伪创新审计：** 此方向因为不增加参数极难被归类为“堆算力 trick”，只要数学证明扎实，是性价比极高的破局点。

## ---

**E. 严格文献表**

本节对报告中引用的核心概念与支撑事实的原始文献进行了严格梳理。下述列表排除了非正式博客，仅收录具官方来源、arXiv 预印本或顶级学术会议发表的一手资料。

| 标题 | 年份 | Venue | 作者 | 核心关键词 | 官方链接 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Attention Residuals** | 2026 | arXiv | Guangyu Chen (Kimi) | Depth-wise Attention, Block AttnRes, PreNorm Dilution | [arXiv:2603.15031](https://arxiv.org/abs/2603.15031) |
| **Mixture-of-Depths Attention** (MoDA) | 2026 | arXiv | Lianghui Zhu (ByteDance) | Dynamic mixture, Depth KV pairs, Signal degradation | [arXiv:2603.15619](https://arxiv.org/abs/2603.15619) |
| **mHC: Manifold-Constrained Hyper-Connections** | 2025 | arXiv | Zhenda Xie (DeepSeek) | Birkhoff polytope, Doubly stochastic, Identity mapping | [arXiv:2512.24880](https://arxiv.org/abs/2512.24880) |
| **DeepCrossAttention: Supercharging Transformer Residual Connections** | 2025 | ICML | Mike Heddes (Google) | Learnable residual, Depth-wise cross-attention | [arXiv:2502.06785](https://arxiv.org/abs/2502.06785) |
| **Deep Manifold Part 2: Neural Network Mathematics** | 2025 | arXiv | Max Y. Ma, et al. | Stacked piecewise manifolds, Coordinate system evolution | [arXiv:2512.06563](https://arxiv.org/abs/2512.06563) |
| **The Neural Differential Manifold: An Architecture with Explicit Geometric Structure** | 2025 | arXiv | Di Zhang | Riemannian metric, Differentiable manifold, Coordinate charts | [arXiv:2510.25113](https://arxiv.org/abs/2510.25113) |
| **Implicit Hypergraph Neural Networks: A Stable Framework...** | 2025 | IEEE BigData | Xiaoyu Li, et al. | Higher-Order Relational Learning, Nonlinear fixed-point | [arXiv:2508.09427](https://arxiv.org/abs/2508.09427) |
| **DyMixOp: Guiding Neural Operator Design for PDEs with Local-Global-Mixing** | 2025 | arXiv | Pengyu Lai, et al. | Complex Dynamics, Inertial manifold theory, Integral kernels | [arXiv:2508.13490](https://arxiv.org/abs/2508.13490) |
| **Hyper-Connections** | 2024 | arXiv | Defa Zhu, et al. | Residual stream width, Seesaw effect, Representation collapse | [arXiv:2409.19606](https://arxiv.org/abs/2409.19606) |
| **Mixture-of-Depths: Dynamically allocating compute in transformer-based language models** | 2024 | arXiv | David Raposo (DeepMind) | Conditional computation, Top-k routing, Compute allocation | [arXiv:2404.02258](https://arxiv.org/abs/2404.02258) |
| **DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging** | 2024 | NeurIPS | Matteo Pagliardini, et al. | Depth-Weighted-Average, Information Flow | [arXiv:2402.02622](https://arxiv.org/abs/2402.02622) |
| **Predictive Coding Networks (Meta-PCN)** | 2025 | arXiv | Chang Qi, et al. | Prediction error propagation, precision-weighted optimization | [arXiv:2506.23800](https://arxiv.org/abs/2506.23800) |
| **Hamiltonian Neural Networks** | 2019 | NeurIPS | Sam Greydanus, et al. | Symplectic gradient, Energy-like quantity conservation | [http://papers.neurips.cc/paper/9672-hamiltonian-neural-networks.pdf](http://papers.neurips.cc/paper/9672-hamiltonian-neural-networks.pdf) |

*(注：部分年份标注基于学术预印本的v1提交时间或接收定稿时间，体现了从底层构想到学术会议发表的真实演变脉络。)*

## ---

**F. 红队审计：对抗范式摩擦力的护航指南**

在将第一性原理推演转化为可发表的顶会论文的过程中，最大的阻力并非代码的实现，而是来自学术同行的“认知孤岛”与审稿人（Reviewer）的惯性摩擦。为了确保创新的硬核程度，我们必须在此执行残酷的“魔鬼代言人（Devil's Advocate）”红队审计 3。

### **1\. 哪些方向最容易伪创新（Pseudo-Innovation）？**

最典型的伪创新陷阱是\*\*“深度的盲目注意力化（Attention over Everything）”\*\*。例如，如果你的设计仅仅是简单地组合了序列 Attention 和深度的交叉 Attention，而不去解释底层的物理连续性或信息的守恒性质。

* **审计指控：** 审稿人会敏锐地指出，你的设计在数学上可能等价于传统的 DenseNet 中使用简单的 1x1 卷积来融合历史特征。如果你使用了复杂的自注意力公式仅仅是为了找回之前层的特征，那么这只是操作算子的重用（Operator Reuse），而非架构哲学的重塑。  
* **规避策略：** 必须深入论证“为何固定的加法累积会导致隐状态幅度的非理性增长”，并通过严密的梯度分布方差图表，证明新机制如何本质地压制了这种增长 7。

另外一个高危陷阱是\*\*“堆叠各类归一化操作并伪装为几何守恒（Faux-Conservation）”\*\*。

* **审计指控：** 宣称模型做到了某种几何投影（例如正交流形），但代码实际上只是在特征输出端强行加了一个 LayerNorm 和缩放因子。这不仅掩盖了高维向量空间中的“角度灾难”，还会被直接揭穿缺乏真实的拓扑约束不变性。  
* **规避策略：** 必须使用真正的流形代数优化器（如 mHC 采用的 Sinkhorn-Knopp 迭代 2，或 Newton-Schulz 迭代），在数学上严格保证变换矩阵的特定范数特征。

### **2\. 哪些方向最容易被 Rebuttal 击穿（例如“只是增加了计算量”或“变相改进了优化器”）？**

在隐式图、超图架构（如 Implicit Hypergraph 层间融合）以及引入复杂神经 ODE 求解器的方向上，最容易遭遇性能归因的反击。

* **审计指控：** 审稿人极大概率会提出质疑——“如果你使用等价的参数量去增加标准网络的宽度，或者延长标准网络的训练时间，利用更大学习率和权重衰减带来的隐式正则化，是否能达到同样的性能？”  
* **规避策略：** 为了抵御此类攻击，所有对比实验必须严格建立在 **Iso-FLOPs（同等浮点运算量预算）** 和 **Iso-Parameter（同等参数量）** 的协议之上 16。  
* 同样，对于基于算子平衡或连续积分的方向（如 DEQ 或 DyMixOp），\*\*Wall-clock time（挂钟时间）**和**推理延迟（Latency）\*\*是阿喀琉斯之踵。即使算法理论上实现了 ![][image23] 的内存消耗，但如果前向过程的求解或求根运算时间使推理吞吐量断崖式下降，在工业界便毫无价值。必须在论文最显著的位置（通常是摘要或图1），如 Attention Residuals 一样，清晰地标明其推断延迟开销的绝对上限（如 ![][image24]）26。

### **3\. 哪些方向最值得优先做 MVE（强制最小可行验证）？**

根据科研战术中的风险原则，在申请数十万卡时去训练巨型范式架构前，必须利用数十行代码在单一 GPU 上阻断理论错觉。

* **优先级 TOP 1：分数阶残差衰减 (候选 Idea 12\)**  
  * **理由：** 这是一个真正的**零新参数、零架构级 FLOPs 增加**的极限测试。只需在现有的基线代码中修改两行加法逻辑，引入带有超参数 ![][image25] 的历史深度多项式衰减系数。如果这个微小改动能在 Long Range Arena (LRA) 或 WikiText 上显著压低验证集损失，这将直接通过纯数学的路径证明：问题的核心在于“长程层级信息的稀释方式”，从而对后续复杂的“跨层注意力”构型提出本质的挑战。  
* **优先级 TOP 2：正交流形投影 (候选 Idea 9\)**  
  * **理由：** 复制开源的 mHC （DeepSeek）架构 38，果断移除其复杂的 Sinkhorn-Knopp 模块，替换为仅由基础矩阵乘法构成的 3 步 Newton-Schulz 近似迭代。核心验证目标是：在超宽残差流下，观测前向网络层的 Gain Magnitude（放大增益幅度）是否能同样被死死压制在 1.0 附近（消除 3000 倍的信号爆炸 2）。如果 MVE 成功，此方向将凭借极低的工程门槛和坚实的矩阵分析解释，成为一篇完美的底层基础设施级别（Infrastructure-level）的顶会爆款。

**最终行动纪律：** 无论选取哪一条路线探索“深度与残差”的重写，架构设计者都必须能够用一句通俗且基于第一性原理的语言陈述其方案。例如：“我的架构不再堆叠层，而是将深度转化为相空间中保体积的辛积分”，或者“我的模型不再使用反向传播链，而是依靠局部能量下降吸收预测误差”。倘若无法提炼出如此锐利的结构可解释性陈述，创新往往只是堆砌了巧合的工程trick。

#### **引用的著作**

1. Residual neural network \- Wikipedia, 访问时间为 三月 19, 2026， [https://en.wikipedia.org/wiki/Residual\_neural\_network](https://en.wikipedia.org/wiki/Residual_neural_network)  
2. Manifold-Constrained Hyper-Connections (mHC): A Comprehensive Summary \- Dr. Robert Li, 访问时间为 三月 19, 2026， [https://drli.blog/posts/analysis-mhc-deepseekai/](https://drli.blog/posts/analysis-mhc-deepseekai/)  
3. 三维质疑法.docx  
4. Hyper-Connections \- arXiv, 访问时间为 三月 19, 2026， [https://arxiv.org/html/2409.19606v1](https://arxiv.org/html/2409.19606v1)  
5. mHC: Manifold-Constrained Hyper-Connections \- arXiv, 访问时间为 三月 19, 2026， [https://arxiv.org/html/2512.24880](https://arxiv.org/html/2512.24880)  
6. DeepCrossAttention: Supercharging Transformer Residual Connections \- arXiv, 访问时间为 三月 19, 2026， [https://arxiv.org/pdf/2502.06785](https://arxiv.org/pdf/2502.06785)  
7. Paper page \- Attention Residuals \- Hugging Face, 访问时间为 三月 19, 2026， [https://huggingface.co/papers/2603.15031](https://huggingface.co/papers/2603.15031)  
8. \[2510.25113\] The Neural Differential Manifold: An Architecture with Explicit Geometric Structure \- arXiv, 访问时间为 三月 19, 2026， [https://arxiv.org/abs/2510.25113](https://arxiv.org/abs/2510.25113)  
9. The Neural Differential Manifold: An Architecture with Explicit Geometric Structure \- arXiv, 访问时间为 三月 19, 2026， [https://arxiv.org/html/2510.25113v1](https://arxiv.org/html/2510.25113v1)  
10. Implicit Hypergraph Neural Networks: A Stable Framework for Higher-Order Relational Learning with Provable Guarantees \- arXiv, 访问时间为 三月 19, 2026， [https://arxiv.org/html/2508.09427v1](https://arxiv.org/html/2508.09427v1)  
11. \[2508.09427\] Implicit Hypergraph Neural Networks: A Stable Framework for Higher-Order Relational Learning with Provable Guarantees \- arXiv, 访问时间为 三月 19, 2026， [https://arxiv.org/abs/2508.09427](https://arxiv.org/abs/2508.09427)  
12. \[2508.13490\] DyMixOp: A Neural Operator Designed from a Complex Dynamics Perspective with Local-Global Mixing for Solving PDEs \- arXiv.org, 访问时间为 三月 19, 2026， [https://arxiv.org/abs/2508.13490](https://arxiv.org/abs/2508.13490)  
13. DyMixOp: Guiding Neural Operator Design for PDEs from a Complex Dynamics Perspective with Local-Global-Mixing \- arXiv, 访问时间为 三月 19, 2026， [https://arxiv.org/html/2508.13490v1](https://arxiv.org/html/2508.13490v1)  
14. \[2603.15619\] Mixture-of-Depths Attention \- arXiv, 访问时间为 三月 19, 2026， [https://arxiv.org/abs/2603.15619](https://arxiv.org/abs/2603.15619)  
15. Mixture-of-Depths Attention \- arXiv, 访问时间为 三月 19, 2026， [https://arxiv.org/html/2603.15619v1](https://arxiv.org/html/2603.15619v1)  
16. Mixture-of-Depths: Dynamically allocating compute in transformer-based language models \- arXiv, 访问时间为 三月 19, 2026， [https://arxiv.org/pdf/2404.02258](https://arxiv.org/pdf/2404.02258)  
17. Mixture-of-Depths: Dynamically allocating compute in transformer-based language models, 访问时间为 三月 19, 2026， [https://arxiv.org/html/2404.02258v1](https://arxiv.org/html/2404.02258v1)  
18. Frequency-Separable Hamiltonian Neural Network for Multi-Timescale Dynamics \- arXiv, 访问时间为 三月 19, 2026， [https://arxiv.org/html/2603.06354v1](https://arxiv.org/html/2603.06354v1)  
19. GeoHNNs: Geometric Hamiltonian Neural Networks \- arXiv, 访问时间为 三月 19, 2026， [https://arxiv.org/html/2507.15678v1](https://arxiv.org/html/2507.15678v1)  
20. INTERACTION ASYMMETRY: AGENERAL PRINCIPLE FOR LEARNING COMPOSABLE ABSTRACTIONS \- ICLR Proceedings, 访问时间为 三月 19, 2026， [https://proceedings.iclr.cc/paper\_files/paper/2025/file/735c847a07bf6dd4486ca1ace242a88c-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2025/file/735c847a07bf6dd4486ca1ace242a88c-Paper-Conference.pdf)  
21. Hamiltonian Neural Networks \- NIPS, 访问时间为 三月 19, 2026， [http://papers.neurips.cc/paper/9672-hamiltonian-neural-networks.pdf](http://papers.neurips.cc/paper/9672-hamiltonian-neural-networks.pdf)  
22. On Neural Differential Equations \- SciSpace, 访问时间为 三月 19, 2026， [https://scispace.com/pdf/on-neural-differential-equations-1kltl88t.pdf](https://scispace.com/pdf/on-neural-differential-equations-1kltl88t.pdf)  
23. PHOTONIC COMPUTING ARCHITECTURES FOR CLASSICAL AND QUANTUM INFORMATION PROCESSING \- Ben Bartlett, 访问时间为 三月 19, 2026， [https://bencbartlett.com/assets/pdf/Ben\_Bartlett\_PhD\_Dissertation.pdf](https://bencbartlett.com/assets/pdf/Ben_Bartlett_PhD_Dissertation.pdf)  
24. Compositional Visual Reasoning and Generalization with Neural Networks \- Aleksandar Stanić, 访问时间为 三月 19, 2026， [https://astanic.github.io/data/thesis.pdf](https://astanic.github.io/data/thesis.pdf)  
25. DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging, 访问时间为 三月 19, 2026， [https://proceedings.neurips.cc/paper\_files/paper/2024/hash/f67449c7ab72f441d3a713b046c6818c-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f67449c7ab72f441d3a713b046c6818c-Abstract-Conference.html)  
26. Attention Residuals \- arXiv, 访问时间为 三月 19, 2026， [https://arxiv.org/pdf/2603.15031](https://arxiv.org/pdf/2603.15031)  
27. Implicit Hypergraph Neural Networks: A Stable Framework for Higher-Order Relational Learning with Provable Guarantees \- OpenReview, 访问时间为 三月 19, 2026， [https://openreview.net/pdf?id=xpw8CTDHQ1](https://openreview.net/pdf?id=xpw8CTDHQ1)  
28. Brain-inspired Predictive Coding Improves the Performance of Machine Challenging Tasks, 访问时间为 三月 19, 2026， [https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2022.1062678/full](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2022.1062678/full)  
29. Reverse Differentiation via Predictive Coding \- MRC Brain Network Dynamics Unit, 访问时间为 三月 19, 2026， [https://www.mrcbndu.ox.ac.uk/sites/default/files/20788-Article%20Text-24801-1-2-20220628.pdf](https://www.mrcbndu.ox.ac.uk/sites/default/files/20788-Article%20Text-24801-1-2-20220628.pdf)  
30. A Predictive Coding Model of the N400 \- PMC, 访问时间为 三月 19, 2026， [https://pmc.ncbi.nlm.nih.gov/articles/PMC10984641/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10984641/)  
31. Reverse Differentiation via Predictive Coding \- PMC, 访问时间为 三月 19, 2026， [https://pmc.ncbi.nlm.nih.gov/articles/PMC7614546/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7614546/)  
32. Stable and Scalable Deep Predictive Coding Networks with Meta-Prediction Errors, 访问时间为 三月 19, 2026， [https://openreview.net/forum?id=kE5jJUHl9i](https://openreview.net/forum?id=kE5jJUHl9i)  
33. \[2506.23800\] Towards the Training of Deeper Predictive Coding Neural Networks \- arXiv, 访问时间为 三月 19, 2026， [https://arxiv.org/abs/2506.23800](https://arxiv.org/abs/2506.23800)  
34. Hamiltonian and Lagrange Neural Networks | AI Weekly Report, 访问时间为 三月 19, 2026， [https://weeklyreport.ai/briefings/hamiltonian-neural-networks.pdf](https://weeklyreport.ai/briefings/hamiltonian-neural-networks.pdf)  
35. Neural Symplectic Form: Learning Hamiltonian Equations on General Coordinate Systems \- NeurIPS, 访问时间为 三月 19, 2026， [https://proceedings.neurips.cc/paper\_files/paper/2021/file/8b519f198dd26772e3e82874826b04aa-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2021/file/8b519f198dd26772e3e82874826b04aa-Paper.pdf)  
36. \[Literature Review\] The Neural Differential Manifold: An Architecture with Explicit Geometric Structure \- Moonlight, 访问时间为 三月 19, 2026， [https://www.themoonlight.io/en/review/the-neural-differential-manifold-an-architecture-with-explicit-geometric-structure](https://www.themoonlight.io/en/review/the-neural-differential-manifold-an-architecture-with-explicit-geometric-structure)  
37. Mapping EEG Metrics to Human Affective and Cognitive Models: An Interdisciplinary Scoping Review from a Cognitive Neuroscience Perspective \- PMC, 访问时间为 三月 19, 2026， [https://pmc.ncbi.nlm.nih.gov/articles/PMC12649996/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12649996/)  
38. tokenbender/mHC-manifold-constrained-hyper-connections: implementations and experimentation on mHC by deepseek \- https://arxiv.org/abs/2512.24880 · GitHub \- GitHub, 访问时间为 三月 19, 2026， [https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections](https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD8AAAAWCAYAAAB3/EQhAAABC0lEQVR4Xu2VsWoCQRCGR1SwsFFsJCGFrYWFNhFMJYFA7PMaYpEmrS9hYy+KVfrUqSwU8gCCvaUg5h925Ybjir3D4gbngw92dre4n9ndIzIMwzAMIwNduIlP3gsqw/fgSNQv8FXUoagKX4DvcOudwDa5AH+wE20NQlX4XzgQ9QX+wDc//hRrIagKv4QlP66QCzwl1/E9fPRroagKLxnCA2zFFxIowzGcxVzDY8I8K9+U3MFH/JvcCciKqs6f4Ipct7nrfNevPMMHUYegJnyD3B3fwQ8/5o9n+FjPyf0N0qAmPAfjzvPD9gWf4NnXC1iPtgajJjxThU1R8+veF3VaVIW/NUVYi08aRj74Bxb8L9mP6Kj0AAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAI4AAAAZCAYAAADnnhbzAAAEkUlEQVR4Xu2aT+hVRRTHj2iQJZQpRWT/XAQh0kILCmuVQYtCCvmFmbgzWokbQwQFCVoJCpJIoO4klWgRRLn4oSJhLQrKFulCUaTATWCgonU+b+78fnPPb2Z+97133/u9J/OBL+/OzH333nPvmXPOnfdECoVCoVAoFAqFQmEMOag6o3rADlScUh2ynX3wiuq67SyMF++p9qo+U+2uD3XAmc6rltiBPpkQ50CFIbLQdvTIu6r/VGtVt1Qb6sPyuOo3ae98lruq7bazMBgWqd6xnT3ypTjHmScuovAZ8oHqnulrk0nVH7azMBjadBzqjDu2M+Cm6oTtbJE14s6/wA54Vkvd2DdUbwXt+wlm7TrVC6b99NQe/dGm4xBNckUq0WiH7Qx4TpxtPlJhM+2mPKG6VH3WeF91odrm4OQ0LnSx6pi4gmwumC8ufz/ZhR7sfDPPr6rN1Tb1A7YeVn2iui3O7n5pw3G8TTjG8Wqb+xHysOpf1cumH54VZ6t3GI6DrQ+Js/UnaW7rUYnYQzX+etDmBJOqt6vtuSqMlql+VF3pQh91vpmGcLtHpm8mYZhQv0rcrMJeHnq/tOE4gGPkIspS1T+ql+yA8oU4Wz2kG2z1EWRSmttK8NhmO8mPPn8xY7lQduRirop7gCGcuMnMHkVeVC0P2ltkeuZ9o/orGAOiXtObG9KW4/iIwySOwfjl6tPyi9Rt5TjYyrVhqz0m0SxlK8EjG0DeFBfOwhNaYs6Ug9dEbuI1cR4/KjBbf1d9aAcCGIulgRAW3axIfd9H+lE3r82s33CNXGuMnOOE8H0cJ2drKuXBrI7D4LeSjij0p8a3ykwvBm4U/btktByH9MT1EIVS7Jf0Q8vRVsQh8ocZwdLUcbD1huRtzTlo1HEoCE+KizJEm/Dhv6p6KmgTkVAMDpy7WYx34ziDKI5XirORG8gDYRZ6HlGdC9rsw83uhTYcx0dEok4KzuNrNMtFcbbidNjKepAHW3kx8GAraTvFjOLYhzAukIUktv1FsIxNyA0XnJgBqTTWtuMMAiIIr7c4Pyk3dJwJ1dmgTVjPrZ/kaMNxqDFThW8INsTORT+2Pi/O1jBiYGv48wS28qIQgwKd38FqzolTEHE48E7VM+Jex2l/pXpsetepiOTBsSiU/Wyngt8UtIkWRA3PKDgOa1V/V8JWbPZtm98pJJmpvdCG4/AmxcNPpSkP9zR2nd6uH8TZihPR/lPqtvJigK2p8/gFQLti3TGSB+2h8H0taHuYpTiZh33Cou9nqReE+8QtPnlGwXEAZw/XQ7AjVuyTpnLhO0cbjmNTaQqWLJjoFj+x/eR9VJyt4WSG2VIyz63JdSShKEafS7yIGodU1Q2+WFxhBxqS+vtDDl4keEikBu7Vd/XhKP3+yEn6xtZPJW4r0eZj29kNLBpR4xyQeFhr4ji88o0D5HVmPJGIQnJYUM/gOF+LKxn4RbwJG6X5vhacFFupZ62tOOPpSH9XkONy//eYzXHGDUI7GjZHxP1hy6aU2cABKBF6IWbn0P7IRf7MrRMUBgtRgReatuCnqPW2s1Domf8BiujvrgmtYpcAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKYAAAAZCAYAAAC7FFXXAAAFHUlEQVR4Xu2aXagVVRSAV6TQLxVaYUWdIgMp6KEyzR56SMFCCyqukm8+GL4VVOKrCD2KFIEI0kMEFkSEBNnDlV6iXhSUoB+6RT8oaBAUmGSt7+5Zd9bsM3Nm5nbPmfG6P1jcM3vvmbNm7bX3WmufK5JIJBKJRCKRSCQSicQlwlUqK1Q2qTwT9cFnKgfixh7wm8pzcWNi8bBf5d9Mvo/6lqp8qbIsau8DUyqn48bE4uJaCY75umu7ReWkytWurW+sl6LOiUXGPSoXVZ50bVuytj5zncrXKrfGHYnFwV6VIxLyTeNPlQ/cdV+5oPJe3BhDTvKsysNxRw8YSNDtiuz6vuy6awiVhCRsBzjHExLCqwd9KVAMbLzBXXt4R/qq+oH7eR7fQ4Gzu9g9G9rjNs9A+mFP8uI4N57jaZWfVNZk1xiZ1bZkbkQzrpRQIbaROu5SOSG5ATE4Fd01KjtVvlK5Kevrgjcl6GZ63ZG1f6jyT/b5fpV7VR6REF7PqNys8okUcyzu5Z6XXRvPJFQb70sYY/Y4L+G77XsBZ/1LwvfF9M2e70jQYQhKdl50ddQeh4YmYBwcvI3U8bbKHnfN1k+YIi9hpU1LyFW6YLnKWgkLGOOycxk4HG308RlH2CjBMalIX8n6d2TjcZhvVF7Lxhq/qqzKPrNhcM++vHt2nmjzc4Vef6g86NqMvtmTNKTUMUk+6fDGwElfcteA4m0ddSE4LsUdA11Z1RjvIwmT7ZmknhQb7C44Anq96PrMMdHn06yN0DotQfd1Kr+r3JD14TA23qDvY8kr620S7nlgbkTumB4i0Y/Z35i29qS6H6ejmp0K2Epnx2T3YsWTt5ixPD9LMVx0QZkDxNTpyQLkfg6dm8ioZxnsep+rXO/amGx0tVzzdpUZKVbOHsYizAPOwzMHrp/72G1thzVo4509oxzT08SeVSnBKB6V/P3rKHVMVgKNddUbO9B8QvtC85DKWclDW0wXeuLo70o46PYQHr3BzbH8buVhLLleFTaBsZPQxjt7mjpmnT3hlAQHbgOO+ZQEG9RR6phlB7PGQPLwjlGrVrpnHMXPdxImjN2dBXTQ9bGzb3bXTfVcSHAUdhU/uYRe7Ep4NtB9aAIc9E3HjRJyT55nE+jDKk5u55fkaggwBqfA8WLa2JN3infopvDdTRyzsvihMV7tJNnHJA/pvHDVSh836Ifx75YQsvwimpLiz21d6MnEoaN3TApKChmcypiR0Qfe5yXsTjE/SDgtwWG8YzJHFDFWtTNfj2d9wFh/PGW0sSch3j+zDU0dk/SxdBznV/yeipLkN7+obHX9vPSoEDNuOFpBjqrcKflxy7dSDGtd6YkzMdn8/HdO5W8ZXujAGI6QqiB6cR/vxjzgqL76hoGEeoC5Yiw5Lc+lIIpPVZjsshStqT0p6sgT2VmNG2U44pnEBVJTx+RU4FDcaBCyWRkr4w4JYQJjdwU7A5UqaQJgnMfctdGVnhiWPM30rMpvOcus6vNQceNk8fsZ7Gg+BTJniflChosiaGpPIgDvZaD7qzJcHJpsz4fO0tQxWVg+fWgMiTXyhrRPgidJF3qym5SlQn3g//4TB+9ENNglYVG1pYljrpfhY8nGcOhK7vaWtP8laJJ0oSdHQDjmqOOWLuHck8mfD+R+pAKE2bLjwzrqHJMFwxntfJ49C2HeJ8R9ZdJ6chD9vJNJ7dJtwcEItfOBMD8uqAcoEhOXKexIh+PGjrlN5QUp/tqYSPSb/wBfGDYd7l6c7wAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAYCAYAAADDLGwtAAAAuklEQVR4XmNgGAWDG+wE4n1AfBGI5YD4ExBvA+JfQByHpI5hChAzAvF/IH4OxDJQ8fVA/BemyAWIBYFYhAGiMBomAQTlUDFeEIcDKmgDxG+BWBPKB4GlDBCFIIPAAGQtSHASTAAKvjJAFMIByNqrQByELMgAUfQTWSAdKohsbTAQ3wJieSQxuFtmATErEFsB8XsgVkFWBAK/GSAeASkSZ0B4EAWwMEBMQ/cIBpBmwAw/rCAECYN8TycAAOQRIu8dLW87AAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAbCAYAAACuj6WAAAAAxUlEQVR4XmNgGAVDADADsQwQuyOJgfhgwAjECUA8B4g5oWLPgPg/ECtB+QyuQPwXiMVgAkAwhQGiiAcmcAUqgAxApsLFOKCcrXBpCHgLxWAgyQBR1AqXhgCQ2GkYxxgqEASXhgCQGMhKMOAH4m9AbAqXZmAIBuKvDBAD4ACkq5IBEk5JQPwLiK8CsQiyIhAAeVUYSu8B4nRUaVQACrznDKjWYwBPBoj1GFaBACgo3jFAFIAwKFr6UFQAgRAQh6BhTRQVFAMA2+IkcvG2C0sAAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,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>

[image7]: <data:image/png;base64,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>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAYCAYAAAD3Va0xAAAA3klEQVR4XmNgGAUDBjiAWABdkBzAA8SSQPwWiP8DsS6UDwMgtjtU7jCUD9KDE/xmgCjGBvwYIHLp6BLogIUBovA5ugQDxPYDQHwViEVQpTABSAHIoK3oEkCgxACxYCkQM6LJYQCQk/8BsQu6BBC0MkAsMUWXQAcgb61hgBi0HohnoWGQa0AGEfSWDQMkoEE2YwMgQ3BFAgoAeQukEJu3QAAk9w1dEBsAxQZIMTe6BAMkoEFyoIAmCAZH+gGlBzMgzmaAGHQDiFUZIJqZgVgMiEuB+CdULowBkidHwZAHAAC4MU5Ds2V3AAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAZCAYAAAA4/K6pAAAA70lEQVR4XmNgGAWjYDACEyD2ReLbAbEbEh8nCAbia1A2IxD/BeLnQCwIxCuAuBUqhxOcAmJbJP5/ID4AxJ5QdjmSHFawBohZoGwOBogmkK36QPwEiGWgcjAgzgBRhxW4MECcr4QugQSwGQoHIOduZcBtA0gcQ/4XEK9lgNgKsh3kdxiwBGJpJD7IhSAMByIMED9fBeIIKNsYKscKxPMZIDEDA6CwQfEeSBLkApC/aoBYjgESjSD+KiAWQiiFuxAD8ACxJBIfFEBWSHwYADkdZBnZABR4INzBAPE6yeAuAyQMpjIg0g1JABRewuiCAwsAcWgmJ89JtZwAAAAASUVORK5CYII=>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIsAAAAZCAYAAAABt923AAAECUlEQVR4Xu2aW6iNQRTHl1CuISK5S8ot5FIkeaB4kVIiKeXBA0+USym3lFciJSVelJQnpXjYIQ+8SKRcCpEQSlHIZf3OfHPO7HX2dznn7LO/T82v/tnfmjnHmm/WrFkz+4hEIpFIJBKJRCKRSAFWq66rFtqGhMWqd9ZYAfaonlhjpPfor/qs2qH6oVpS39zGU9VRa6wA+H5eNcw2RJpPH3Eve5fqkOq1amrYQVml2mlsVYIxfFMtsA2R5jJG9UK1UtwqHVLfLP1Ul1RTjL1qECynrDHSXDar/qpG2IaEveLaq84x+T/87ARpcbJqevIZ8XlC0KcqnBT3kskgjbim+m6NAWQjiuKRwfNy1eD2Hq1hjWQEywxxjRdVYxN1FQbof7aIbIpuxCTVVnEBslacj9QEg1Q/Vffbe5bLcHFjei7Ox7R3+FV12xoT1oubB2CMv1WbxGUpfiervVXg+6vk3zo4xn0R52xPuCquoCuqI+7HUmF1nhEXKLBM9Utc4UVtwAusJW1VgWDIyhx/VDetURmvuhc8+22ABeVXOVtYqyBAWYhzbcMjcc74SbH0VY22xhbAKnsQPG+XjnqAl/he3Iv0lOVnCP5R4KZB+wVrFDe2K8Ez25XfBpiwW+ICysP2xILpLXi/NTEnIlYvTpHywlW/JehD0Za1WlrFY8nYR6WYnyyIs11QOEF58Lvxj7oljbRgsZCBsi7tyDxvrDGHgaoTqreSfyxuGCwYGUAY1RYGz0SVDce5T9YYULafvrYgaNMoGiz0I7ukQVtWeyMIlhWqg9LNYPGZ5VxoNDBBpMk8ml3gzhFXMLIdNfLzrrii11PUz96CuxVWe1Y2IuBr1ijObwp2JnSquLGGW+xD1bjgmb78f92B2icvWFILXByrWWMARSXFZavxx1BeCpdYfA6LvDvSccSEsvz0cGtL8Zp1zKWeYRIsfosdpdoo9Teo1Cd8NRDWlGxRBFV3KBIs1EkU650WNNF8WNz3GdQrDGZD0kZqZYtKuzfoTbhv+JDohuqAuL2c52dBPyjTTxgq7kictQVB2qXcbnHZglMp4yToqCOZsMtBPyBI7DHaZu2sDF4kWHIv5Saqlkp9BLMFlJnaWVWccDjpAPcZpHn/7CnbTyaQ1e7vSdJYIu7LxUZBzaQyPmB8jNM/h5Bpwy1ogHQuzENt6+jaRpFgqaleGlsubAWkxn22oWKU5edH6dgeM1diQE++SCQwKGwJouOmrSh5wUKyYFyzbEMW7L2kQyL8vGmrEmX6SYAg/naF+6oikC3p2507If9F5WzVadNWFIJlkTUGECT7Jf3eLZNGqbCKlOHnOnGBOtM25NCTP35iEm0d0iziHz9VlHmq+dZYMlzxT7PGSKRL/ANacdulcG5q+AAAAABJRU5ErkJggg==>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAAAZCAYAAACB6CjhAAACkklEQVR4Xu2XzUtVURTFd2RgKBQpiSAGlYNQFEkbhEP9C5wENY0GNUpQau4f4AcNIgjn4VBQHEg0c6ADRRCEjD4gqKBhEbUX+97ab3nP/XrvvkDeD9bgnH2fZ9919tn3KNKiRYtTzlnVs0jnKXbqOadaUl1T3Vftqa7UPBFxqPqteqjqzdCAakH1IfoNNCTpzKlW3HhHNeLGZUE+u26MdV7QGPnFvFcdufFfplS/VN9UoxRLY1VsASyUBhuApG+6cVmSDPDrjElt/DhSIhfk346eoVgW31UTPOngxJplgAfHARt8iwOej2IGTHMgA5TdMk86OLH/YQDeaZInGbj0XMyEDYrVAyfWbAPeqK6r2lT3KHaCLil/FEJwYs00AF0/7vzdYhucyYyYAQeqPoqVgRMLGYD1sC5KFS/XXhs+QZYB8VfAa97Fg+AovBT7QS7HMuDEkgxAc8KXqEj/yTKgLlA2sWv1womxAZfFLilopKFjhxtdJ81VagB4ImYALj/1wImxAa/E1sFF5Z2T565qnOYqN+C26i1PloATYwO2xAxAlw6xKNbEPJUagCMwK+GSLAInxgbEFRDihuoLT0qFBuBWuCnWDBsBJ8YG3BEzgM94DMr/J09KRQZgx/E5QgXkBUZxeXo4MTYA4F/VH6qvkY7FGh/YFqsSphID8BlKvS8n8EiSE4zhxJIMADCyX+wy5o8eyv+BG8c03ADsPM59ES6pPqmucsDBiYUMCLEvVmGDNN9QA3Ducekpcu6HxUzD7qd1cE7steS/YXaI/X08jxw9SQY8dePcYNfRhPw3OKTP0bNeabsP2IAtseTzcjESk2QAVBhcQfmliiht9wG6vL9Sr6t63LgsqIg1N34cqUVe/gDNqsHfyrJ7vAAAAABJRU5ErkJggg==>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAYAAAAZCAYAAAASTF8GAAAAgUlEQVR4XmNgGASAGYjFgNgBiEOQJTyB+D8SxgAgwQPoghwMEIlWdAkXIH4OxEroEuUMEGN4kAWlgfgBEPshC4IAyJh/DDiMAVmMYgwIPGDA4f7fDBAXoQAWBoT7QZbvh0mIQyWCgHgKEDfAJEDgKxB/AeJiIGZElgAFhz6ywIACALGlGWS271ZmAAAAAElFTkSuQmCC>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAaCAYAAAB7GkaWAAAAh0lEQVR4XmNgGLaAEYjDgJgVXQIEooF4DxBzo0uAdM0H4lZ0CRAQBOLTQOyHLAgy6j8WzAOS5ABiSQaIkSBBEBuE4UATiN9CMQYA2QPSBbITA4CcD5IE2Y8BngPxVyA2hvJB7oADkK6rQCzCAPFvM7rkUqgEyN5gZMntQPwXiJ8CcQRU0RADADPcG4p8yKkfAAAAAElFTkSuQmCC>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADMAAAAZCAYAAACclhZ6AAACkklEQVR4Xu2Xz4uPURTGjyYyIT9GpJRoNqIsZP4AsZBsxiwUO9Es7NQUJQtZ2AlZiGQrCxtFWUxZjNhYTCmlKKVmslGmKGaex3lvc7/Pe+993+/MdxTmU0/znXPuj3PPe95772u2zDL/JX3QUeg2NCK+v46b0GXoDPQN2tHpXhgroPPQY+g5dKnTnYTt7qixYsA86zErzeeJmYXuVr+3Q++hrfNuG4PeRv83ctC8wzPoNHQNmoIeQpuidjEM7JV50Cl+mQeqigMlb6Cz1e9t0Mfqb4Dz3IfWR7Ykg9AkdMHqWSQMlgFoNreY9+sXu7LRvP9rdSQIQQ+pw3x+luB+dcS8s3SwgXPm/r1iP2Ge+Sb2WWcZlTgOfVZjBBdzS42EwV+BfkKHxRfDTHAQDYa2R2JLcQP6Au1Wh8CnwSohp6ANkS9w1TwxNUKQT61cKqHdA7Fz0ItiU9ZBL8xLjOWWg7tXnBhuKOyrHLHMYj6ZO3apQ2DAbMcMB9ZAM9CByJbipHlf/i0RNoYgPoEUqc3hN6EjAyvxxOoBbYa+mr8PJZgA9m0qsbbw6fIp1+YNi2mCL6QGlM2QwInZt1Ri3bAWGrfEjtZmMezMNt/F3nYxpTl41tSCamBRixk2b3Nd7N0shuWYglt7aeNJkV0MT3tOlhuQOwr9YbuM4aBNB1jpfOH1ZFqNLcgmcdR8stQZwwXSV7oP0X9MjRG5nSyM3eaMUpggPmkmswavDz/MB+eTuGd+qk9Y/kYQSB2aq2y+fEvisbCz6tMN2UMzsBo6ZL4YbqW5S6Py0jyoP8k49EFsPaHtRbNXsFL4nu1RR6/gHSr1zi0FXARv9k3lvyhKH2e9ouuPs4XCDyZ+wC0lvEmkjoh/iznUQpw3RFEqiwAAAABJRU5ErkJggg==>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFgAAAAZCAYAAAC1ken9AAAD70lEQVR4Xu2ZT4hPURTHz4Qi5D8hzSQpIQtJio1YWLAgJRaKBQtlIWRKkSyUhSQikoWULJQ/IYspC2JLpNSQUmSjKOTP+cx9d9w5v3vfe/P7M78p71Pfmnvufe9373n3nnPeG5GKioqKior/kA7VPtVN1UPV4YHdUS6IG2vhXpOtUZmiGmGNATNVi1TLbUeLGa2aaI3KOGtQOlUHxK2xNKtVL1X3VTtVJ1QfVdcl7ijgB96r5tkOZYHqs+qP0RfVkmBcCIvx466YvlZzWmrnii6GgwI+qTZZYwyc81x1SOI766m4H7JPa4zqlmqGsVtuiLt+ru3IAecOtYM9P8XNt4iF4pw80nZYXkvcgZ494vo5tiErVN+NLcYbcdePtR05tNPBfucWgb8uS61f+mHAMdUv1VrTF7JU9VVqj0qv6pmxxWCyv62xgHY5eKq4+d6xHQk4lU9UE2wHeMfdE3fcU/hxdsE47aqxWXiITPiD7Sggz8HbVWdVXZI+dSTS9apZ4ta2VVwILDpFy8TN97jtSMD9kjmF5FQmNnaLG0cSCMG20dgsK2VwE/bEHEyeeBu0iYG0OwMbjuVE+uRDJUKI6hK33qgjMoil5As2Q5FPQljfQWsEH2uKnirHhXHbAhslDbZ1gS3GLnE7fY3tKMA6mF3I75EPQmhT5YwK2o9U47M2a+O6I1k7D8LDC3HrZX1l4f42fPbhHVwET5RxlF0eyinCBuEjRb07AqyDeUCxo8iRJtGScIFremRg7crcY3W6hdNGBTHY08b97Wnro4yDfV1qq4UyDi6a8F3VBmvMsA7mb8IBLyEhtMMFkqx7VbOz9qSsf0vWzqOonHwg8ZeOhhxMjGXMKWMv42DCA9emwgj1daqGrtfBnBra51XTxFVJvGn6EJIH4YFrYyGT+9rw5Ek6mLc2OlMVBBOjP/aWBvSRqVPk1b8koby4bB1MCIjFctphiKBtH0JZUhuOEowQk4rLXBNNcrvFdcZqYJ9UeAgpkjfOSL0R7VD9EHd8U1gHswO515nA1pG1z8m/tykcvFfK7dgQv/Nj5STOja3Dk7vRmAiLZRA79pK4nfJY0jWmp1fiLxp+J+TJxnSPj/mhfNzjoe8XV4ZRcn3L2iG+aohpcTDOM19qx8V0zV9gIF73SDw298PW58njYGpdasky8KMpR7WS6eLCgP1uwiJJRMT8sI/Kg9KL+fqd3ixI0nmnuCHKfuwZKnAsuy0G4Yic0My5+o89RSe9IVaJi+XDAU4elYlNhGyEo+IqimY6g8qKh9ZyUh/c28Uccf80eKd6pdosiY8xDdApdXxwrxcS5UlpfnwbrpBIb0ttDqgYKv4CcxgEfKhQqE8AAAAASUVORK5CYII=>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAWgAAAAZCAYAAAAc7LQFAAAMvklEQVR4Xu2ceahtVR3Hv9E8D0YDGVfjqQ02U2FZCqWWVEYKBklIFA2IQdGgUNwI/2gkmoQIbgYRxYMIS0uidgMWJVJRBGZg0QBFBVHRs3F/WPv37u/8zlp7Ovvec+67+wOLd89aZ++9ht+0fmufJ83MzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzBxq7l6Xh8RKx33C53uFzycCbWNifu7hPt8lfD5MMBdfr8sDY8Mewnw/qy7fqcvFrp6+fFL725e9gPlkHBHG3aaX91X6jsF83NV99mzV5W1a/P5h4plalJ395qF1uVuoYz1+HuqyvKgu/8sUI9Z/Re0G7SDyZi2Pk7pS20eatsMEyo1QHYkNDWfU5U2xcgJ26vKXurymLn/WopHBOGPcME7rAn2I8pErX9NysEO/f1CXk0I9PLwuv9TyfaiH2PanujyuacvxR63XSK2LrbrcprJzekVdXhgrJ+aVWjbQrP2OBgQYZ9XlWF2+r8WLnqukIKUBnkiwWAj750L9ZXV5QagbC8qIR0dw7hfaNpnr1e7xmbf/xMoVQN7eU5e3uzr+Nsdp0K9fhLohnBwrRmKGMgc6dHWoe1hdflqXe4f6CLqITqKb6KiBHGHciaT78AQlIx0NRR8IyF6iJLelKH1TQSaviJUNzB1rdlNsmJAH1+VbsbIBGf97XZ4eG3KYIPxXix6FqIXI6TBwqtKC4ZAMog4mcVUQBiJM7v8HpXnl74t0MIQeA7EdKx1n1+X0WLkC5ykpl0WMgIG+zn2Gl6psGPtQaRpHaVFsDpT0daGOYABd68O20v0/5upIjVD6gjEgYjszNrTANTcqPfvXdflNXe7UYj82GZwRfUavSzxf/Z3cGDC+t8RKB7al93yagWJQRBY/1Hq3j+sAQ8Qc4LCeomlydxj5Vyl/HyKodys9a1N5lJIsYGj2i+goAeNcadGgooRHtRhdDqHS6gb6yUr99amvLaXcOeBkYpSEYtLvPjBGi9DRxw8o3X8oj9HyDrlElwO4XakfmwwOnfTCOsEpxl2f5xoNCDAwICYILM6nFpsPBV9VGv8blLb0q0a3bAt/HysDzPuPlba9fWB7+4m6vFF5o98FW9ZH1uXc5jNK/zKlvuYgvfNZ5Z/Ftc9r/p0KnBZrwFoY91c6KIwRNKAAbUrQRqXVDTRGgP56Y4CxRvngQUrz7eH7Me3RBtE212yrPdXUBpHiX5UcSht9DoNxEEP6gR7hpN6ndO41BtaJwPH85jNyfIlS+iZCG+cDz4gNDY+uy4tj5cSgp6SxSikWsDPA3uBhueCc2LBGMAwI+JAyFnYR7CBIQ5wW2sZwq/oZXiLALoFHcVibq5rP5lDtLQAMwZVKqYHPKAnIc5SiNT4/Wykaoz9Ep1yLwXsAFysZYupiSgtH/fJQB9z/Uu32I27jx8J9MEjbSgpIeVdTl4uIUPxK4wxtpXHXGewq2F2giI9Vkj1yzjhlItYcGMp/qGw8cmwrzfH7Q/1QuIfP60cwbOT1+4AsbcfKDDh+DuqONJ9PUZI/6pFpdhg3K/XtAiVj/unmM7KAM+B77OSQbeqRBxw5snd5U89O1cAJ4Yxw7BHOlHhDBrn5mdJbFlPBmrK7IOChz+ySmCccXu6MBnn5VfNvLz6oNAHkRXIR0zrAgJADG1LGck8lAWUO2jxfX7ZjhZISxMgcYfMRYw4ECyPu87I4k2PuM2uGgUUhEGxyjr9z7UalNMYY+VLnBckikVzU83ql6zFwUxlo+k+0zv1yJaYKAEPIWw1+XvpSaTUDbYYg9rNS+b4YhD6RrIdc6b+1vP5DoW9tO2PeBLklVio549x4MKxtqS+u4zAz6hL98IfxyCoyaylF7E8uxYIx41pvmywFxCGoQbCBQ4/5ZYKT7zV/c34xtYEmR//t5m/6aDs7C/yI3D3m4HvJAjdECBgYN+OmJTAwfSLDg8a1SnPAgrObaAOBjYbWQ3s0bEQNRA9EXAilx7bEObhXpeVUg73m5TGDuaMk5BZxeyotXwesO/UIPdhzo2H0kRbRP2PCGbRxulLqKDoFjwlsfJ2zUupXzhgMjkIclfKGpy+W3mBctntD8dvWckx/2dGZXHLA2AY7qVJwxfXXxUoHb2tEw2g7t2gY4Udqf73PDnHjbgEbg1P18F12e+TuMZynLLQmzEBHCFJ8PePIfQ/DzVsbyDcOoisQZew4F2S3C3QbJ2EpFwz/2c3fPAPdjTvAkn4tQcctV4rCMbi2iI4HsU0bCh4SBWzbZq0DJvBy7eZhSwJpmCEptQMKGCf+qHbvHfOmbREo90F4iW5IacQSYbtHJEzklaNSXoBRXurNo3cJENEcUV2X0SCfaemz6LQ8Nk4UzMN18dDQ6GPwtrU8ZxQUaidTj5HoAzJA37zjIKIvpTegT3+NRyhFZBgK5rBtHoDntp15cH2bgUYvYztGxmQ2Ru/fVVk2wOQJ4xTnmN16BOfA99nJ5igZ6EqL/SsZaAOjS3tbnh3s1Vv6FaNxDzYTJ+Hnwr//bGco0e516ddxWFSUGkwQeGAJDkHwcjnawnUm7kItd3TdMHYv2BZJnurqPEQNpdeqjKER9NXhs8cMV99588/J7XQq5QV4qIHmlUyiIfKDXRCFvFbtEbSN0z/PtrB+S+wZYvAilVaLoOlrnEfym21jHNJfDIN3wGYoSxBo3RkrHVwbDbBnaARNwNAWQZs89RkrMFa+T/osx1QG2l4G6IKxI7NdETTBFfczR8087ew2H7cXrI+nS7+On8ZGgeoyUDysFPEVH9ZAe19DAwyWBR5ShoAxw6N64TtL+R/uGG0OysPbGTHqyMGc/CRWOugb27Fcf9gaeljLm5QWHyOde9+2Ul5AcVJREVGyGNEafN/uTz9WMXbA2nnjRT8+rJSmicbBYO4qjXt2pXHXAf1hrrocdYTnRScUYQ0xVieHepwUzyztWEgNUUpwbZvu9TkLMXCc27EycKaSDMb+nlSXd4Y69BAZModguVxPyUD/S4v15uhLa8t3mSfavxnaxoDhRQ/seTwf52WQzsntfKK8L8EWzVt6A2PAIDBcOZgQy69E2gQPhhrovYTcGIeKUflNUOMPdwzm7WiszPAlLb8VkeMLSoaojacq9edSV4egs800GAfPO9J8JhJhHS11Y1RNvXfMZnD8YQuQiimtF99nLrjP50PbGOxQ0lIEpBpQtOiUPOxScCJjqFRW4i7IMdr4h8J1JacHH1L+F5K2Y0E2kdEIDvOaWOnoei6cp+XdXQ7eVrF8awmM+LVKxtav4WV1ucF95nm3aVcP2flF+QQz0F5fzaBzjUFQhFHMGT6iXL5PCoIxfGOxeRToIYehOCTg3mYfSFOxq8m9CcVOlQPjBRkkOqSDvvjJQxlju4+WGSAPt/wKk+Wj1wvC5zhJm2Cg4/iITGw8uXaUwm9P2hxUhIV5tfIHiigZytj3hyqPVzqFJr/8W6VIntwYC1xpt79msHwdxZTT6t+rJCD2y0buHyF9gRHKHdCxHaYv7BS2QttYUDgOxBjfR7X7GmAO1gxZ7AoKSlQabqBz+kPps6MycDrRwVtq0ReeZTDG2I4z84epGOhS/pv6Sv3Ge6OWUx2eOzTshypvVTqzQs4o5zT1loqwYusYx2l9NgONvhKR/q35HHcagI2Jh3IGMks/3qLl4GwVCLSwFdyf/t2u9CO1EjhT+j8p5FO8wSbH6ZP/XwyffQ4NNsFArwrKOOT1HKJ0PKyPCFBIBO5mV9cXPLV56zFU2hUMFPxpSqf/OTCCx5T/tR5OB3mIkc6q5H7ckQPncYf6GZ0clcZfuwqkqkgjTontPrzB9pA+6Kt3OP2PK71v7FN0T1R6p/5K5QOONvg+MtRnXUv4FAfrhg6U1o9nxcDLQPdy8jwF6ALR/JNiQ4ZKSX4nhagN4/SO2NDQFc0cdANtDuqM2NADDA9CdZrKgrUfVBrmua9X9w9p1gH98lvbobSlTvYSghr6nUtTjIUdHsWiUw+pCFJXY6JFDM75Sr88HWqUp6aUgy5BJHtFrNxjmOP4hlYOvseadKWJBoFnZevDlmIntBl9DHTbGwubDjlP0hv81Pqgwis/CHpfhSVXN+a1yr3mn1rMyR8kyMOS750KjAJnD/ycOsL5Rnzv+CDCrmOIgSblsIoDHwOpwD7pTwzzVeqvg4MobYehy0CfCBABTb2t3w+I2HAwl7hyrv9CC4wX57yuqNNDX0idbUJfVoH5jCnAVUAuY5S7pWn+0691wquLF2lRbjlA7gMH5BfHyj2EYCaXVvGwHmvbka6SY5rZbDgYIje5bsiBcpB10MHBEOXtFex4v6xlo33YuFXpLahNgUN3e9NqZmZmZuYg8H+SagHEWZT3kgAAAABJRU5ErkJggg==>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACcAAAAZCAYAAACy0zfoAAABEUlEQVR4Xu2VoW4CQRCGfwKiogkORKt4gCJ4AQSmpoLwFFgEFsMbkIoaHoV5lHMIbKuatJ3pBtgb7sgwyUJI9ks+M3Ob/W/3dg/IZDInNNkOO2Qn5dbteWV/I2/BCzvVxRgJRrpo4FkXjIxQXhQqdSMeEB5Y6oYBYh910cATwqckY8+Gk7fYsj3dMEDwhYs5G24O/yQE37iY2nCyvAX7pupWCAnDyZb+wLelAiFhONlSaXonIPjH7qkNV8B2vy3Yjwp37Lqi/v4/ykZtuG+Ek+qFkGjlWjjeb3IgNuW2CUKicF2ExphdIWzdpRAShRO+2E92xjZUzwLBF26AEKrKA/Lr6seFCyH4wl2Fti5kMvfIH6NTRwDiwE8pAAAAAElFTkSuQmCC>

[image18]: <data:image/png;base64,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>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAABaUlEQVR4Xu2UMS9EURCFj4iEICESotCIRqUQHdFQatR+gG0UEvED0Kj8A9EodAqNkk7jN1BQaSQURDjHzMTNZO9mxVb4kpN9b867c+fNnbfAP3+G/nQ/THW7BpPXip4cCMapBvVOnVCjsORaMEa9ujfj94HWbVC31CpsTZV5WJLtbJAbmNebDTJLTeVgM1SdkhxlAxaXcruWqYsUq6LXVZKzFFd1L+7pmZJzWGvaQtUpyWURW6CuYW8lb7rwJmDtaZsBWBL1Ozil1vG1QSTsonb891soyaNfj8B6LHTw8lb8/tBjzRhCi42V5MmvVXnM9aZ7sYEOVgVkVFxt2j6JaTmAtSBQYsVV9RrVV3glS7CBqBIbXMEOMZijnqlj92rsUfc5WBIb5NfX9DxQd2j9USm5Nqmi/r/lIGz+NV21gw3UHrWpyi41mYOwQ9vKwYSe0UdaPeCfosqlxWx0Co2yzmc/G50k/uZ/ER+J1kZumRa9JgAAAABJRU5ErkJggg==>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACkAAAAZCAYAAACsGgdbAAACYUlEQVR4Xu2Wz6tNURTHl1BE+ZFI0itJyczPlDIxeUUp6Q38AYiBKAzfxFzKRCQjE4bKwOAmgxcTg/cykDJ4P0bMGJAf38/b+7jrrrOPe59zyeB+6tu9e+2zz15nrx/nmI0YMeKfsVK6I62LEy04KC1EYxuuSrPROAQmLDnbmtXSG2lLnBgS36Rr0bhUDkuT0ThEOpYO4Y/ZJr2XNgT7MDkifZVWxAnPMumEtC9OiGPSd0vXNEEasH55Hm+XTtrv13hY/y7/1uAm5MPjPKZyf2RV3A1jD9dPWqp8+CR9lDZK45Yqd0ee68cDSw9a45Q0LW12Nhz64MZPsq3EFes6CDjFtWukZ5bShHQZhBvS5WgkJG+li8HOJq/cuGPphCKbpJfBRlrgKLl1UzpvvSHnMNa6sYfqrlU4juCQX0RxYDvjbB0rOxlZZWktJ9LEZ+lANGaKThLSGEYKB4d8AXWyrR+ElZOk0JqYsRSBEkUnCQtlX0FY7mf5EJHQ8WFgq/TCuk5xguQh+QiE3G+6WzrrxpFi4Ty13s15PVHpPtRAMpecZEPs/FLlU5Y2qjgk7XRj7ks/LFEVWqkF2lFLG5ErF/L/2LSbmvmYpc4wn3/3SF8stSBS6Xj30sW11EBTs66aed++SphLJ8aNH1p6NUZo3jTgqg1RhHvduIJQ+7YWIS1qe++X5qRdzkbSN33ltP3AuGWpaK5bOvEIp3guGnmL+ComdDzJpV9X1CElajcaEPLtkaVoxe9RDuB5wb4YptvSa+medLp3ukjbj9710WB/4aMXcJSHayqApcIba5AD+j/5CV8Sd2vwPwuZAAAAAElFTkSuQmCC>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAXCAYAAAALHW+jAAABPElEQVR4Xu2TTytEURjGH6EoJSVZjNha27JRlpaKzJ58ABZ8AgsLGyWWYuM7TFbKWookpZSyVZI/z9M79zrzuM1cMyvlV7+ae573nnPPe+YA//wZFughPafbtLcxLk8/XadPdIvO093681JSV4oK/aT7HpBBeobISn2tilR8Q8csy5hBLLjqQRHHiOIpDwzVyJY8IgqHPTBeEHV9HjgfiMJRemKZ2EFs9QFRp343JdvKAK0lz6kb9L7+u1VrfkyoA0q9Q/MJR2h3OvDbHmrhjCF6RLuSsY5OeZKu2Bh66B69pRONUc4cYrKqjesmFe5Mt0EvHHhAxuk1YtH0pmi7F4gPKkSN1Z19p6eIe12jV3T2uyxnmb75YBHqpSbTCa/BGp6g7V76YCc8Iw5k04N2eaXTdNGDdtEB6U+d8wWH4EqBptOQlAAAAABJRU5ErkJggg==>

[image22]: <data:image/png;base64,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>

[image23]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAZCAYAAABD2GxlAAAB8klEQVR4Xu2WPyiFURjGX6GIUhQZLLIoJpPBIgaDlMVgtBhMKFEmkzLIKAaDlAxKQhluFsWqFMtVStkN5N/zdO53nft+59x7vq+b7nB/9SzP+37nvOd8559IlSpVUlMDLUDH0CW0Whh2si0mNymJvxuG7qELaAZah16hQ6jVyrPhgJ6hHh2wWISatQlaoCuoXgc0bPwOWoZqVYzcQD9iirFphE6gDuX3i2mP33xCT1BnQcYfbOMR6tIBmwdxFxAxJybep/xB6F15hLM9AbVBS1K8QMK22UcMFrQGfUGjKmYzAL1BO8rPQrfK04QUeCTugeY7Phcz1T6ivD3lf0P7ytOEFDgvZhZjcHEz0K0DihUxeVvKpzepPE1IgdEExDYSO6CadEBxKiZv2vIact6Y5bkIKZAT9CLxzZYvsBT8mHm9lsfRctQcfTFCCmTMmRNSIAthjl7EFVMg1xhzNpX/LwXy1mDnvh3Mq4hx3y3B2Lg2FSEFejfJrJhOXGcgi2aMg/DBOAsoRkiB3GjeP8l78ENMAmdsV8z5di3+myUiK+6DOvpl0RKy5RoQD2rOoBceGSNiCuRZxysqhAOJb540ZKGM8sqC77GQBLbBh0W7DpSLITFrOS1c/1PaLDeJH5450n6XGG60DahOB0pwJv6HcGXzC7n7fjUiA+OUAAAAAElFTkSuQmCC>

[image24]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAXCAYAAAB0zH1SAAACI0lEQVR4Xu2WO2gVURCGR4ygGFCIoAElFxtBsAgRbawE0wRCIEXSW1gKCgr2FrGwCISABIKIYCFCQBHEIkmTQBoDeVQWipWgYqGFD8z/efaws+fu5T7T7Qcf7Jyz92bu7JzZmFVUVFSUMC5/y8PyurwmDxTuCKzKxXRxP7ksH8hZOZrswZZcz6775D+5ZuGHwEl5R76XtWxtXzkm38ojWXzaQtW+yUvxJguJ3nMxP4J7I/yYeXnIrXUMX9YMHvtXKz72EQuJxgoD8V0Xv7NwX2RSbru4bQblnHwhzyd7ZTy3kBQt4mENfbzgYtrkRHZNlenrjqt9Tv619g7HmPwpLybraeK0zisX8zfiU6LatXyrdfiCN/KPhYp3CweOpH+4NQ4eUwXo/eHsekhuZtctcVBeld/lzWSvGzgXtMRHeTbZAwrT72KSJnkgJ1rvWb5dhF76ID9bPg16BY+ddvMTpRE1C/cDOTFVbmfWfX7CQtLT1sVhaACV27DW+nXKwmyP0P9fXLwir7j4PzySXettxeM8P+XWjrvrFEZfrDZ8slDQyGN5y8V18MZjNM3IgWSvVXhyT+RRt0Z/77jYQ7VjX0c4yD7x+1ac+6UwVV7LX9b+VOGzTIyn8pGTYvhEIjUrf9GkFef7bri4KST+0MILiLnejGXLZ3YqjzuFKeJf9RH+CeNQUwhcsg7bmCry0uglZywkWAbJMk0Yhy/lheJ2RUVT9gA7+GbAHkqVfwAAAABJRU5ErkJggg==>

[image25]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAaCAYAAACD+r1hAAAAsklEQVR4XmNgGAWjgCjQDcQ/gHglED8F4lIgZkVRgQQ0gfg9ELtD+SCF/4G4DMr3AGJuKJthF1SSBSYABQ+h4uJAfBxZAiT4D1kACg4wQORygHgVTBCkGyS4BiaABA4wQOQmADEjTFASKlgOE0ACBxggcmLIgjAbgpAFoWAHA0QOA4BC5xYS3wqITwBxGgPEb/xAXIkkz2AKxHeA+AkUnwJiJwZI0M4C4ndAfBGuehQQAABIMyWU5xwOsQAAAABJRU5ErkJggg==>