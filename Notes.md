# Notes of Statistical Mechanics

## Part 1 经典统计

### 1. Hamilton力学回顾

在有势体系下，Hamilton量的定义为：$ H(p, q) = \sum\limits_{i=1}^{3N}p_i\dot{q_i} - L(q, \dot{q})$，式中$p_i = \dfrac{\partial L}{\partial \dot{q_i}}$。运动方程由正则方程

$\dot{q_i} = \dfrac{\partial H}{\partial p_i},\quad\dot{p_i} = -\dfrac{\partial H}{\part q_i}$描述。Hamilton量不显含时间也说明能量守恒。

Hamilton量的意义告诉我们，粒子的状态可以由$\pmb{p}_i,\pmb{r}_i$参量来描述。因此定义**相空间**中的6N维矢量$\vec{x} = (\pmb{p}_i, \pmb{r}_i)$来描述一群（N个）粒子的状态。若在能量守恒的条件下（亦即$H(\vec{x}) = C$），则矢量 $\vec{x}$ 终点在相空间中画出一个(6N-1)维超平面。



### 2. 系综及相关概念

考虑大量的系统，它们之间有相同类型的作用力，有相同的宏观性质。但它们的运动可能有不同的初始条件从而有不同的时间演化。这样系统的集合称为**系综 (Ensemble)**。

**微观状态 (microstate) **表示一个系统在任何时刻下的位置和动量的集合。自相平面的视角观之，它表示一个系统任意时刻的**相空间矢量**。在能量守恒的情况下，求微观态即求超平面上的点。

**系综平均 (ensemble averaging)**的概念则表示为$\color{\purple} A = \langle a \rangle_{\mathrm{ensemble}} = \dfrac{1}{\mathcal{N}} \sum\limits_{i=1}^{\mathcal{N}} a (\vec{x}_{i}) $，式中$A$为系综的某个宏观状态，$a(\vec{x_i})$为第$i$个系统与该宏观状态相对应的微观态函数，$\mathcal{N}$为系综内的系统数。

**时间平均 (time averaging)**则表示为$$\color{purple} A = \langle a \rangle_{\mathrm{ensemble}} = \lim\limits_{T\rightarrow \infty}\int_{0}^{T}a (\pmb{x}_i(t))dt$$。

**遍历假设 (ergodic hypothesis)**认为，系综平均与时间平均相等，因为它们都已经消除了系综的微观细节。



### 3. 相空间分布函数与Liouville定理

若研究系综中的一个特定系统，它在某一个相空间体积元$\mathrm{d} \pmb{x}$中出现的概率为$f(\pmb x, t)d\pmb{x}$，$f(\pmb{x}, t)$称**相空间分布函数 (phase space distribution function)**或**相空间概率密度**。相空间分布函数满足以下性质：$f(\pmb{x},t) \ge 0, \quad \int f(\pmb x, t)d\pmb{x} = \mathcal{N}$。

**Liouville定理**：

(1): **相空间分布函数是守恒量**，即$\color{green} \dfrac{\mathrm{d}f}{\mathrm{d}t} = 0$。证明要略如下：

> 由二阶偏导可交换的性质，可知$\sum\limits_i (\dfrac{\part^2H}{\part p_i \part r_i} - \dfrac{\part^2 H}{\part r_i \part p_i}) = 0$。代入正则方程整理得到$\nabla \cdot \pmb{\dot{x}} = 0$。
>
> 由$\nabla\cdot (f(\pmb{x},t)\pmb{\dot{x}}) = \pmb{\dot{x}} \cdot \nabla f(\pmb{x},t) + f(\pmb{x},t) \nabla \cdot \pmb{\dot{x}}$，可得$\nabla(f(\pmb{x},t)\pmb{\dot{x}}) -\pmb{\dot{x}} \cdot \nabla f(\pmb{x},t) = 0$。
>
> 对体系内相空间中给定的区域$\Omega$，单位时间从其表面流出该区域的系统数表示为$\int f(\pmb{x},t) \pmb{\dot{x}}\cdot \mathrm{d} \pmb{S} = \int \nabla(f(\pmb{x},t)\pmb{\dot{x}}) \mathrm{d}\pmb{x}$。它等于区域内相空间分布函数积分的减少速率$-\int \dfrac{\part f(\pmb{x},t)}{\part t} \mathrm{d} \pmb{x}$。该规律对任意区域适用，因此$\dfrac{\part f}{\part t} + \nabla \cdot (f(\pmb{x},t)\pmb{\dot{x}}) = 0$ 。
>
> 代入上面，即有$\color{purple} \dfrac{\mathrm{d} f}{\mathrm{d} t} = \dfrac{\part f}{\part t} + \pmb{\dot{x}} \cdot \nabla f(\pmb{x},t) = 0$，得证。 

备注：

1. 该规律只对相空间适用，而非$(r_i, \dot{r}_i)$构成的位形空间。

2. 定义Liouville算符$\color{deeppink}iL \overset{\mathrm{def}}{=}\pmb{\dot{x}}\cdot \nabla$，式中乘以$i$为了保证$L$是厄米算符。则表示为$\color{purple}\dfrac{\part f}{\part t} + iLf = 0$。

3. Poisson括号表述下表示为$\color{purple}\dfrac{\part f}{\part t } + [f, H] = 0$。

(2): **相空间中的体积元在时间演化之下不变**，即$\color{green}\mathrm{d}\pmb x(0) = \mathrm{d}\pmb x(t)$。证明要略如下：

> 设关联两个6N维体积元的Jacobi矩阵为$M$，则有$\dfrac{\mathrm{d} |M|}{\mathrm{d}t}= |M|\sum\limits_i \sum\limits _j (M^{-1})_{ij}\dfrac{\mathrm{d}M_{ji}}{\mathrm{d}t} $。
>
> 以矩阵元$M_{ji} = \dfrac{\part x_j(t)}{\part x_i(0)}$为例，其中$x$可表示动量或位置。则有$\dfrac{\mathrm{d}M_{ji}}{\mathrm{d}t} = \dfrac{\part \dot{x_j}(t)}{\part x_i(0)} = \sum\limits_{k}\dfrac{\part \dot{x_i}(t)}{\part x_k(0)}\dfrac{\part x_k(t)}{\part x_j(0)}   $。
>
> 交换求和并整理，最终有$\dfrac{\mathrm{d} |M|}{\mathrm{d}t} = |M|\,\nabla \cdot \pmb{\dot{x}}$。由Liouville定理证明过程可得$\dfrac{\mathrm{d} |M|}{\mathrm{d}t} = 0$。
>
> 即Jacobi行列式等于初始值1。故而始末体积元相等，得证。

结合相空间分布函数不变的性质，最终有$\color{red}f(\pmb x, t) \mathrm{d}\pmb x(t) = f(\pmb x, 0) \mathrm{d}\pmb x(0)$。

若研究的系统**不是保守系统**，例如与热源或粒子源接触，则Liouville定理的等式不一定成立。此时$\kappa =\nabla \cdot \pmb{\dot{x}} \ne 0$。令$\kappa$的原函数为$W(\pmb x(t))$由微分方程$\dfrac{\mathrm{d} |M|}{\mathrm{d}t} = |M|\,\nabla \cdot \pmb{\dot{x}}$得到$|M|(t) = \mathrm{e}^{W(\pmb{x}(t)) - W(\pmb{x}(0))}$。因此守恒律表示为：$\mathrm{e}^{-W(\pmb x(t))}\mathrm{d}\pmb x(t) = \mathrm{e}^{-W(\pmb x(0))}\mathrm{d}\pmb x(0)$。

广义的Liouville定理表示为$\dfrac{\part f\mathrm{e}^{-W}}{\part t} + \nabla \cdot (\pmb{\dot{x}} f \mathrm{e}^{-W}) = 0$，可以通过$W$的定义最终推出$\color{purple} \dfrac{\mathrm{d} f}{\mathrm{d} t} = \dfrac{\part f}{\part t} + \pmb{\dot{x}} \cdot \nabla f(\pmb{x},t) = 0$，它与Hamilton系统的Liouville定理有一致的形式。这说明Liouville定理是不同系统的普适规律。



 ### 4. 微正则系综

#### 4.1 平衡系综概述

**平衡系综 (equilibrium ensemble)**指相空间分布函数**不显含时间**的系综。由$\dfrac{\part f}{\part t} = 0$可得$[f, H] = 0$。因此，$f$一定是$H$的函数$F(H(\pmb x))$。

分布函数在相空间上面的积分$\color{deeppink}\mathcal{F} \overset{\mathrm{def}}{=} \int F(H(\pmb x)) \mathrm{d} \pmb x$称作**配分函数(partition function)**，等于微观状态数总和。

一个宏观物理量$A$的测量值则为**系综平均**$\color{purple}\langle A\rangle = \dfrac{1}{\mathcal{F}} \int A(\pmb x)F(H(\pmb x)) \mathrm{d} \pmb x$。

配分函数和系综平均是系综理论的核心概念。

#### 4.2 基本概念与配分函数

**等概率假设(postulation of equal a priori probabilities)**：处于平衡的**孤立**宏观系统中，所有的微观状态都是以等可能的概率出现的。它是下面讨论微正则系综的出发点。

**微正则系综**指在系统处于能量$E$、粒子数$N$和体积$V$时，其中的系统在相空间的分布函数为$\textcolor[rgb]{0.54,0,0.08} {F(H(\pmb x)) = \delta (H(\pmb x) - E)}$的系综。假设能量处于$E$和$E+\Delta E$的薄球壳之间，其配分函数为$\color{purple}\Omega(N,V,E)  = \dfrac{1}{N! h^{3N}} \int\limits_{H \in [E,E+\Delta E]}d\pmb x$ $= \dfrac{\Delta E}{N! h^{3N}} \int_{\Omega}\delta (H(\pmb x)- E) d\pmb x $ 。

> 关于配分函数前系数的说明：
>
> $N!$是因为交换系统内任意两个粒子，所观察到的状态不发生变化；
>
> $h^{3N}$是因为量子力学认为每个**可观测的微观状态**占据$(\Delta r\Delta p)^{3N} = h^{3N}$的相空间体积。
>
> 因此观测到的系统数+1对应着$\dfrac{1}{N!h^{3N}}\Delta \pmb x$。

#### 4.3 热力学量

由微正则系综的配分函数可定义系统的**熵**$\color{deeppink} S \overset{\mathrm{def}}{=} k\ln \Omega(N,V,E)$，也由此定义两个系统之间只交换热量后相等的热力学量--**温度**：$\color{purple} T = (\dfrac{k\,\part \ln \Omega}{\part E})_{N,V} = (\dfrac{\part S}{\part E})_{N,V}$。过程要略如下：

> 设有粒子数和体积分别固定为$N_1、N_2$和$V_1、V_2$的系统，它们可互相传能，但总能量保持为$E_1 + E_2 = E$。
>
> 则总的配分函数需要考虑两个系统所处的各微观状态，因此$\Omega(E) = C\Delta E_1 \sum\limits_{i=1}^{P}  \Omega_1(E_1)\Omega_2(E-E_1) $。
>
> 数量级上，由配分函数定义易知$\Omega \sim V^{N}$，又因能量为广延性质，故$E \sim N$。
>
> 设满足$ \Omega_1(E_1)\Omega_2(E-E_1) $最大的$E_1$为$\bar{E_1}$，则推导得$\ln \Omega(E) = \ln \Omega_1(\bar{E_1}) + \ln \Omega_2(E-\bar{E_1}) + \ln E + C'$
>
> 总能量不变时，有$0 = \dfrac{\mathrm{d}{\ln\Omega} }{\mathrm{d} t}= \dfrac{\part \ln \Omega_1}{\part \bar{E_1}} \dot{\bar{E_1}} + \dfrac{\part \ln \Omega_2}{\part \bar{E_2}} \dot{\bar{E_2}}$。由于$\dot{\bar{E_1}} = -\dot{\bar{E_2}}$，则有$\dfrac{\part \ln \Omega_1}{\part \bar{E_1}} = \dfrac{\part \ln \Omega_2}{\part \bar{E_2}} $  。

其它讨论可以得到化学势、压强的表示，与热力学公式$\mathrm{d}S = \dfrac{1}{T}\mathrm{d}E + \dfrac{P}{T}\mathrm{d}V - \dfrac{\mu}{T}\mathrm{d}N$一致。



### 5. 经典Virial定理

令$x_i,x_j$为相空间中的坐标，则**微正则系综平均值**$\color{purple}\langle x_i \dfrac{\part H}{\part x_j} \rangle = kT\delta_{ij}$。证明要略如下：

> 根据热力学量系综平均值概念，有
>
> $\langle x_i \dfrac{\part H}{\part x_j}\rangle = \dfrac{C}{\Omega(E)}\int x_i \dfrac{\part H}{\part x_j}\delta (H(\pmb x)- E) d\pmb x = \dfrac{C}{\Omega(E)}\dfrac{\part}{\part E}\int x_i \dfrac{\part H}{\part x_j}\eta (E-H(\pmb x)) d\pmb x $
>
> 由于分部积分可知$x_i \dfrac{\part H}{\part x_j} = x_i \dfrac{\part (H-E)}{\part x_j} = \dfrac{\part (x_i (H-E))}{\part x_j} - \delta_{ij}(H-E)$，因此有
>
> 原式$= \dfrac{C}{\Omega(E)} \dfrac{\part}{\part E}(\oint\limits_{H=E}x_i(H-E)dS_j + \int\delta_{ij}(H-E)\eta(H-E)d\pmb x)$。
>
> 前一项表示在$H=E$超平面积分，显然为0。因此$\langle x_i \dfrac{\part H}{\part x_j}\rangle = \dfrac{C\delta_{ij}}{\Omega(E)}\int_{H<E}\mathrm{d}\pmb x$。
>
> 此式涉及**均匀系综**的相关结论，可以导出它和温度的关系。$\square $

通过Virial定理，可知$\langle E_{ki}\rangle = \langle \dfrac{1}{2}p_i\dot{r}_i\rangle = \dfrac{1}{2}kT$，因此N个分子的总动能为$\dfrac{3}{2}NkT$。



### 6. 正则系综

#### 6.1 基本概念

将内能$E(N,V,S(T))$进行Legendre变换得到$\color{deeppink} A(N,V,T)\overset{\mathrm{def}}{=}E(N,V,S)-TS$，即**Hemholtz自由能**，它是下面要考虑的正则系综的基本能量，有$\mathrm{d}A = -p\mathrm{d}V - S\mathrm{d}T + \mu\mathrm{d}N$。

**正则系综 (Canonical Ensemble)**是系统与大热源之间传递能量，维持温度稳定的系综。控制变量为$N,V,T$。

其相空间分布函数满足$\textcolor[rgb]{0.54,0,0.08} {f(\pmb x) = C\mathrm{e}^{-H(\pmb x)/kT}}$。过程要略如下：

>考虑两个系统，相空间矢量为$\pmb x_1,  \pmb x_2$，则总的配分函数为$C\iint \delta(H_1(\pmb x_1)+H_2(\pmb x_2)-E) \mathrm{d}\pmb x_1 \mathrm{d} \pmb x_2$。
>
>研究$\pmb x_1$，其相空间分布函数自然为$f(\pmb x_1) = C\int \delta(H_1(\pmb x_1) + H_2(\pmb x_2)-E)\mathrm{d}\pmb x_1$。
>
>$\pmb x_2$为大热源，则利用一阶近似有$\ln f(\pmb x_1) = \ln f(\pmb 0) + H_1 \dfrac{\part }{\part H_1}\ln\int\delta(H_1(\pmb x_1) + H_2(\pmb x_2)-E) \mathrm{d} \pmb x_2$
>
>进一步等于$\ln f(\pmb 0) - H_1\dfrac{\part}{\part E}\ln\int f(\pmb 0)$，其中$\ln f(\pmb 0) = \int\delta(H_2(\pmb x_2)-E)\mathrm{d}\pmb x_2 = \dfrac{S_2}{k}$
>
>再利用$\dfrac{\part S_2}{\part E} = T$可得$f(\pmb x_1) \propto \exp(-H_1/kT)$。$\square$

#### 6.2 配分函数与热力学量

正则系综的配分函数为$\color{purple} Q(N,V,T) = \dfrac{1}{N!h^{3N}}\int \mathrm{e}^{-H(\pmb x)/kT} \mathrm{d} \pmb x$。

Hemholtz自由能用正则配分函数表示为$\color{red} A(N,V,T) = kT\ln Q(N,V,T)$

令$\beta = \dfrac{1}{kT}$，则其它热力学量均可从配分函数来表示：

$E =- \dfrac{\part \ln Q}{\part \beta}, P = kT\dfrac{\part \ln Q}{\part V}, S=k \ln Q + \dfrac{E}{T}, C_V = k\beta^2\dfrac{\part^2\ln Q}{\part \beta^2} $

正则和微正则系综的配分函数可用Laplace变换关联：$\Omega(N,V,E)\fallingdotseq \tilde{\Omega}(N,V,\beta) = Q(N,V,T)$。

#### 6.3 正则系综的Virial定理

**正则系综平均下的Virial定理**最终表达式为$\color{darkblue}\langle x_i \dfrac{\part H}{\part x_j} \rangle = kT\delta_{ij} + \int x_i \mathrm{e}^{\frac{-H(\pmb x)}{kT}}|^{x_j=\infty}_{x_j = -\infty} \mathrm{d}S_j$，右边的积分不总为0，但一般研究的有界体系会满足$x_i \rightarrow \pm \infty,H(\pmb x)\rightarrow +\infty$（即势阱），所以微正则系综Virial定理的表述大都适用。

#### 6.4 正则系综的能量涨落

从正则系综满足的条件可知，其能量并非不变，而有**涨落 (perturbation)**。我们以能量的标准差来描述正则系综能量的不确定性，即$\color{deeppink} \Delta E \overset{\mathrm{def}}{=} \sqrt{\langle H^2 \rangle - \langle H \rangle^2}$。用系综平均的定义与能量和配分函数关系即有$\color{purple}  \Delta E = \sqrt{kT^2 C_V}$。

热容和能量均为广延量，因此$\dfrac{\Delta E}{E} \sim N^{-1/2}$。可见粒子数越多，相对涨落越来越小。



### 7. 温度和压力的微观意义

令$N$个分子总动能$\color{darkblue} K (\pmb p) = \sum\limits_{i=1}^{N}\dfrac{\pmb p_i^2}{2m}$，由Virial定理易得温度的统计意义$\color{purple} T = \dfrac{2}{3Nk}\langle K(\pmb p_i)\rangle$。

令函数$\color{darkblue} \Pi(\pmb p, \pmb r) = \dfrac{1}{3V}\sum\limits_{i=1}^{N}[\dfrac{\pmb p_i^2}{2m_i} + \pmb r_i \cdot \pmb F_i(\pmb r)]$，则压力的统计意义为$\color{purple} P = \langle \Pi(\pmb x) \rangle$。推导要略如下：

> 相空间中引入变换$\pmb s_i = V^{-1/3}\pmb r_i, \pmb \pi_i = V^{1/3} \pmb p_i$。可知$\pmb s_i$表示粒子的相对位置，在体积发生微小变化时，$\pmb s_i$不发生变化。由Liouville定理可知相体积元$\mathrm{d}\pmb \pi \mathrm{d}\pmb s = \mathrm{d}\pmb r\mathrm{d}\pmb p$在演化中不变，因此$\pmb \pi_i$也不变。
>
> 因此重新改写$Q(N,V,T) = C_N\iint\exp(-\dfrac{1}{kT}[\sum\limits_{i=1}^{N}\dfrac{V^{-2/3}\pmb \pi_i^2}{2m_i}]+U(V^{1/3}\pmb s))\mathrm{d}\pmb \pi \mathrm{d}\pmb s$。
>
> 代入压力的表达式$P = kT\dfrac{\part \ln Q}{\part V}$，注意$\pmb F_i =-\dfrac{\part U}{\part \pmb r_i}$，最终可得$P = \langle \Pi(\pmb x) \rangle = \langle  -\dfrac{\part H}{\part V} \rangle$。$\square $

备注：

1. 上面提到的$K(\pmb p)$和$\Pi (\pmb x)$等微观函数称为**估计值 (estimator)**，其系综平均则为物理量的观测值。
2. 推导可得$\sum\limits_{i=1}^{N} \pmb r_i \cdot \pmb F_i = \dfrac{1}{2}\sum\limits_{i\neq j}\pmb r_{ij} \cdot \pmb F_{ij}$，式中$\pmb r_{ij}$是粒子的相对位移。可见这是表示相互作用的量。

### 8. 等温等压系综

#### 8.1 基本概念、配分函数与热力学量

对$A(N,V,T)$做Legendre变换得到$\color{deeppink} G(N,P,T) \overset{\mathrm{def}}{=} A(N,V,P(T)) + PV$，即**Gibbs自由能**。它是下面讨论的等温等压系综的基本能量。热力学公式有$\mathrm{d}G = -S\mathrm{d}T + P\mathrm{d}V + \mu\mathrm{d}N$。

仿照正则系综的导出方式，**等温等压系综 (Isothermal-isobaric Ensemble)**是在正则系综基础上用活塞保持同外压平衡得来。能量通过改变$PV$来交换，可类似用**自变量为$PV$，参数为$\beta$的Laplace变换**得到配分函数：

$\color{green} \Delta (N,P,T) = \dfrac{1}{V_0}\int_0^{\infty}Q(N,V,T)\exp(-\beta PV)\mathrm{d}V$，其中$V_0$为具有体积量纲的常数。

$\color{purple} \Delta (N,P,T)=\dfrac{1}{N!h^{3N}}\int_0^{\infty}\int \mathrm{e}^{-\beta (H(\pmb x)+ PV)} \mathrm{d}\pmb x \mathrm{d}V $。

Gibbs自由能用配分函数表示为$\color{red} G(N,P,T) = -kT \ln \Delta(N,P,T)$

其它热力学量：$\color{forestgreen} V=kT\dfrac{\part\ln \Delta}{\part P},\bar{H}=-\dfrac{\part}{\part \beta}\ln \Delta,C_P = k\beta^2 \dfrac{\part ^2 \ln \Delta}{\part \beta^2},S=k\ln\Delta + \dfrac{\bar{H}}{T}$。

其中， $\color{deeppink} \bar{H} \overset{\mathrm{def}}{=} \langle H\rangle + P\langle V\rangle$是等温等压系综中的焓变定义。

与正则系综能量涨落类似，等温等压系综平均下存在焓变涨落$\color{purple}  \Delta \bar{H} = \sqrt{kT^2 C_P}$。

#### 8.2 压力Virial定理和功Virial定理

以下考虑**等温等压系综平均**。令$-\dfrac{\part H}{\part V} = P_{int}$为压力在微正则和正则系综中的估计值。

**压力Virial定理**：$\langle P_{int} \rangle = P $，说明内外压强的测量值相等。

**功Virial定理**：$\langle P_{int} V \rangle +\color{darkblue} kT $ $= \langle P_{int} \rangle \langle V\rangle$。可认为在新加的一个自由度$V$上满足能量均分原理。



### 9. 理想气体

#### 9.1 系综处理热力学体系方法概论

> 1. 列出体系Hamilton量与系综对应的配分函数；
> 2. 将配分函数并最终化为相应自变量的表达式；
> 3. 通过配分函数、Virial定理或直接待入系综平均积分表达式等方法，求出热力学量和关系。

#### 9.2 微正则系综方法

下面以[微正则系综](#5. 微正则系综)方法处理理想气体为例，分析统计系综处理热力学问题的思路。

**1.体系Hamilton量：**$H = \sum\limits_{i=1}^{3N}\dfrac{p_i^2}{2m}$；

**微正则系综配分函数：**$\Omega(N,V,E) = \dfrac{E_0}{N!h^{3N}}\int \delta (\sum\limits_{i=1}^{3N}\dfrac{p_i^2}{2m}-E)\mathrm{d}^{3N}p_i\mathrm{d}^{3N}r_i$

**2.分析配分函数**

> 由于Hamilton量不显含位置坐标，则体积可以从中分离出来。
>
> 再令$\dfrac{p_i}{\sqrt{2m}} = x_i$，则配分函数表示为$\dfrac{E_0}{N!h^{3N}}V^{N}(2m)^{3N/2}\int\delta(\sum\limits_{i=1}^{3N}x_i^2-E)\mathrm{d}^{3N}x_i$。
>
> 令$\sum\limits_{i=1}^{3N}x_i^2 = R^2$，该问题实质上为N维球。体积微分$\mathrm{d}^{3N}x_i=\dfrac{(2\pi)^{3N/2}}{\Gamma({\frac{3N}{2}})}R^{3N-1}\mathrm{d}R$。
>
> 又由delta函数性质$\delta(R^2 - E)= \dfrac{1}{2\sqrt{E}}[\delta(R-\sqrt{E})+\delta(R +\sqrt{E})]$，得到最终表达式。

$\color{darkblue} \Omega (N,V,E)= \dfrac{E_0}{\Gamma({\frac{3N}{2}})N! E}[\dfrac{V}{h^3}(2\pi mE)^{3/2}]^{N}$

**3.得出热力学关系**

利用**Sterling公式**$\color{green} \ln{N!} \approx N\ln N - N$，整理得$\color{purple} S=k\ln \Omega = Nk\ln[\dfrac{V}{h^3}(\dfrac{4\pi mE}{3N})^{3N/2}]+\dfrac{3}{2}Nk-\color{green} k\ln N!$

标绿色的项是全同粒子交换不增加状态数的假设下引入的，去除此项为经典观点下的熵表达式。

由此可得其它热力学量。例如$\dfrac{1}{T} = \dfrac{\part k \ln \Omega}{\part E}=\dfrac{3kN}{2E};\, P = T\dfrac{\part k\ln \Omega}{\part V}=\dfrac{NkT}{V};\,C_V = \dfrac{\part E}{\part T}=\dfrac{3Nk}{2}$。

#### 9.3 正则系综方法

[正则系综](#7. 正则系综)方法处理理想气体时，配分函数的形式格外简单：$Q(N,V,T)=\dfrac{1}{N!h^{3N}}\iint\exp(-\beta\sum\limits_{i=1}^{3N}\dfrac{p_i^2}{2m})\mathrm{d}^{3N}p_i\mathrm{d}^{3N}r_i = \color{darkblue} \dfrac{1}{N!}[\dfrac{V}{h^3}(2\pi m k T)^{3/2}]^{N}$

代入$S=k\ln Q + \dfrac{E}{T}$即得到$\color{purple} S(N,V,T) = Nk\ln[\dfrac{V}{h^3}(2\pi mkT)^{3/2}] + \dfrac{3}{2} Nk-k\ln N!$。下同。

#### 9.4 等温等压系综方法

[等温等压系综](#9. 等温等压系综)方法中，配分函数$\Delta (N,P,T) = \dfrac{1}{V_0}\int_0^{\infty}Q(N,V,T)\exp(-\beta PV)\mathrm{d}V = \color{darkblue} [\dfrac{1}{h^3}(\dfrac{2 \pi m}{\beta})^{3/2}\dfrac{1}{\beta P}]^{N}\dfrac{1}{\beta P V_0}$

 代入$\bar{H} = -\dfrac{\part \ln \Delta}{\part \beta}$得到$\bar{H} = E + P\langle V \rangle = \dfrac{5N +1}{2}kT ,\,P\lang V\rang = (N+1)kT$。

由[功Virial定理](#9.2 压力Virial定理和功Virial定理)可得理想气体状态方程在等温等压系综中表达为$\lang P_{int} V\rang = NkT$。



### 10. 巨正则系综

#### 10.1 基本概念

**巨正则系综  (Grand Canonical Ensemble)**是控制变量为**化学势$\mu$、体积$V$和温度$T$**的系综。对$A(N(\mu),V,T)$做Legendre变换为$\tilde{A}(\mu,V,T)=A-\mu N = A - G = \color{deeppink} -PV$。$PV$是巨正则系综中的基本自由能。没有特殊名称。热力学公式为$\mathrm{d}(PV) = P\mathrm{d}V + S\mathrm{d} T + N \mathrm{d} \mu$。

定义巨正则系综分布函数为$\textcolor[rgb]{0.54,0,0.08} {\rho (\pmb x, N)=\mathrm{e}^{-\beta PV}[\dfrac{1}{N!h^{3N}}\mathrm{e}^{-\beta(H(\pmb x, N)-\mu N)}]}$，过程要略如下：

> 考虑两个系统$\pmb x_1, \pmb x_2$。它们温度均为$T$。粒子数和体积均满足$N_1 \ll N_2, V_1 \ll V_2$，即$\pmb x_2$为大粒子源。
>
> 若系统不交换粒子，则总的正则系综配分函数为$\tilde{Q}(N_1, N_2, V, T)=\dfrac{1}{N!h^{3N}}\int \mathrm{e}^{-\beta H_1(\pmb x_1, N_1)}\mathrm{d}\pmb x_1 \int \mathrm{e}^{-\beta H_2(\pmb x_2, N_2)}\mathrm{d}\pmb x_2 =\dfrac{N_1!N_2!}{N!}Q_1Q_2$。
>
> 但$\pmb x_1$是开放系统，则$Q(N,V,T) = \sum\limits_{N_1=0}^{N}\dfrac{N_1!N_2!}{N!}\tilde{Q} = \sum\limits_{N_1=0}^{N}Q_1(N_1,V_1,T)Q_2(N_2,V_2,T)$
>
> 定义$\rho_1(\pmb x_1, N_1) = \dfrac{Q_2(N_2, V_2, T)}{Q(N,V,T)}\dfrac{\exp(-\beta H_1(\pmb x_1, N_1))}{N_1!h^{3N_1}}$，易知它**满足归一化关系**$\sum\limits_{N_1=0}^{N} \int \rho \mathrm{d} \pmb x = 1$。
>
> 由$A=\dfrac{\ln Q}{\beta}$与小量近似条件，可得$ \dfrac{Q_2(N_2, V_2, T)}{Q(N,V,T)}= \mathrm{e}^{\beta(A(N,V,T)-A(N-N_1,V-V_1,T))}=\mathrm{e}^{\beta(\mu N_1 - PV_1)}$。
>
> 将其代入$\rho_1$的表达式即得出分布函数。$\square$

#### 10.2 配分函数与热力学量

定义巨正则系综配分函数为$\color{purple} \mathcal{Z}(\mu,V,T) = \sum\limits_{N=0}^{\infty}\dfrac{1}{N!h^{3N}}\mathrm{e}^{\beta \mu N}\int \mathrm{e}^{-\beta H(\pmb x ,N)}\mathrm{d}\pmb x = \exp(\beta PV)$。

因此核心自由能$PV = kT\ln \mathcal{Z}(\mu,V,T)$，由此导出$k\ln \mathcal{Z}-k\beta(\dfrac{\part}{\part \beta}\ln \mathcal{Z}) $。

定义**逸度**$\color{deeppink} \zeta \overset{\mathrm{def}}{=} \mathrm{e}^{\beta \mu}$，$\dfrac{\part}{\part\mu} = \beta\zeta\dfrac{\part}{\part\zeta}$。则配分函数写作$\color{purple} \mathcal{Z}(\zeta, V,T)=\sum\limits_{N=0}^{\infty}\zeta^NQ(N,V,T)$。

由此导出内能：$E = \lang H \rang = -\dfrac{\part \ln \mathcal{Z}(\textcolor{red}{\zeta},V,T)}{\part \beta}$，粒子数$\lang N\rang=kT\dfrac{\part \ln \mathcal{Z}(\mu, V,T) }{\part \mu} = \zeta\dfrac{\part \mathcal{Z}(\zeta, V,T)}{\part \zeta}$。

#### 10.3 巨正则系综的粒子数涨落

巨正则系综的粒子数存在涨落$\Delta N = \sqrt{\lang N^2\rang-\lang N\rang^2}=\color{purple} kTV\dfrac{\part^2P}{\part\mu ^2}=\sqrt{\lang N \rang kT\rho\kappa}$

式中$\rho$为粒子数密度，$\kappa$为等温压缩系数$-\dfrac{1}{V}\dfrac{\part V}{\part P}$。当$\lang N \rang \rightarrow \infty$时，$\dfrac{\Delta N}{\lang N \rang}\rightarrow 0$。



### 11. 真实气体与液体

#### 11.1 基本讨论

**体系Hamilton量：**$H = \sum\limits_{i=1}^{N}\dfrac{\pmb p_i^2}{2m} + U(\pmb r_1,\cdots\,\pmb r_n)$；

**配分函数：**令**热力学波长**$\color{deeppink} \lambda \overset{\mathrm{def}}{=}\sqrt{\beta h^2/2\pi m}$，**构型配分函数**$\color{deeppink} Z_n \overset{\mathrm{def}}{=} \int\mathrm{e}^{-\beta U(\pmb r_1,\cdots \pmb r_n)}\mathrm{d} \pmb r_1 \cdots \mathrm{d}\pmb r_n$，

则容易得到正则系综配分函数$\color{darkblue} Q(N,V,T)=\dfrac{1}{N!\lambda ^{3N}}Z_N$。

#### 11.2 径向分布函数$g(r) $

考察条件概率密度$P^{(n)}(\pmb r_1,\cdots \pmb r_n)=\dfrac{1}{Z_N}(\int  \mathrm{e}^{-\beta U(\pmb r_1\cdots \pmb r_N)}\mathrm{d}\pmb r_{n+1}\cdots  \mathrm{d}\pmb r_N) $，它表示特定的粒子1~n分别处于坐标$\pmb r_1 \cdots \pmb r_n$的概率密度。

定义**广义关联函数**$g^{(n)}(\pmb r_1,\cdots \pmb r_n)\overset{\mathrm{def}}{=}\dfrac{V^{n}N!}{Z_N(N-n)!N^{n}}P^{(n)} = \dfrac{V^{n}N!}{Z_N(N-n)!N^{n}}\lang \prod\limits_{i=1}^{n}\delta(\pmb r_i - \pmb r_i') \rang$。括号内的式中对积分参数$\pmb r_1',\cdots \pmb r_N'$进行正则系综平均。它与**任意粒子**在$\pmb r_1\cdots \pmb r_n$出现的概率有关。

考虑$n=2$的情况，$g^{(2)}(\pmb r_1,\pmb r_2)=\dfrac{N(N-1)}{\rho^2}\lang \delta(\pmb r_1 - \pmb r_1')\delta(\pmb r_2 - \pmb r_2') \rang$，$\rho$为粒子数密度。

做变换$\pmb r = \pmb r_1 - \pmb r_2; \pmb R = \dfrac{1}{2}(\pmb r_1 + \pmb r_2)$为满足$\mathrm{d}\pmb r_1 \mathrm{d}\pmb r_2 = \mathrm{d}\pmb r\mathrm{d}\pmb R$。$\pmb r$是两个粒子的相对位移。

令$\tilde{g}^{(2)}(\pmb r, \pmb R)= g^{(2)}(\pmb r_1, \pmb r_2)$，定义**径向分布函数 (radial distribution function)**$\textcolor{deeppink}{g(r)}=\dfrac{1}{4\pi V}\int_{0}^{\pi}\int_{0}^{2\pi}\sin\theta\mathrm{d}\theta\mathrm{d}\varphi\int\tilde{g}^{(2)}(\pmb r, \pmb R)\mathrm{d}\pmb R = \color{deeppink} \dfrac{N-1}{4\pi\rho}\lang \dfrac{\delta(r-r')}{rr'} \rang_{r',\theta',\varphi',\pmb R',\pmb r_3'\cdots\pmb r_n'}$，

式中角度参量是对$\pmb r$在球坐标中对应的角度。因此最后结果只是两个粒子之间距离的函数。

易知径向分布函数满足$\color{purple} 4\pi\rho \int_0^{\infty}r^2g(r)\mathrm{d}r=N$。

#### 11.3 热力学量

由正则系综中[内能的表达式](#7.2 配分函数与热力学量)易得$E = \dfrac{3}{2}NkT + \lang U \rang$，因此着重考察势能项。

我们在此假设**仅有两个分子之间才存在势能**，即$U(\pmb r_1,\cdots \pmb r_n)=\sum\limits_{i\ne j}u(|\pmb r_i-\pmb r_j|)$，

则$\lang U\rang = \dfrac{N(N-1)}{2Z_N}\int u(|\pmb r_1-\pmb r_2|)\mathrm{e}^{-\beta U}\mathrm{d}^N \pmb r_i=\dfrac{\rho^2}{2}\int u(|\pmb r_1 - \pmb r_2|)g^{(2)}(\pmb r_1,\pmb  r_2)\mathrm{d}\pmb r_1\mathrm{d}\pmb r_2$

进一步利用前面定义的径向分布函数，最终得到内能$\color{red} E =\dfrac{3}{2}NkT + \dfrac{N^2}{2V}\int_0^{\infty}4\pi r^2u(r)g(r)\mathrm{d}r$。

再看压力。结合[正则系综对压力的讨论](#8. 温度和压力的微观意义)和理想气体的结论，可知$P=\dfrac{N}{\beta V}+\dfrac{1}{3V}\lang \sum\limits_{i=1}^{N} \pmb r_i \cdot \pmb F_i(\pmb r) \rang$

相互作用力的平均项$\lang \sum\limits_{i=1}^{N} \pmb r_i \cdot \pmb F_i(\pmb r) \rang = \lang  \dfrac{1}{2}\sum\limits_{i\neq j}\pmb r_{ij} \cdot \pmb F_{ij} \rang = \dfrac{1}{2Z_N}\int\mathrm{e}^{-\beta U}\sum\limits_{i\neq j}\pmb r_{ij} \cdot \pmb F_{ij}\mathrm{d}\pmb r_1 \cdots \mathrm{d}\pmb r_N$。

类似于内能项的讨论得到$\lang \sum\limits_{i=1}^{N} \pmb r_i \cdot \pmb F_i(\pmb r) \rang =\rho^2 \int\pmb F_{12}\cdot \pmb r_{12}g^{(2)}(\pmb r_1, \pmb r_2)\mathrm{d}\pmb r_1 \mathrm{d}\pmb r_2$，其中$\pmb F_{12} = -\dfrac{\mathrm{d}u}{\mathrm{d}r}\mathrm{\hat{\pmb e}}_{12}$

利用径向分布函数，最终给出压力表达式$\color{red} P = \rho k T - \dfrac{\rho^2}{6}\int_0^{\infty}4\pi r^3\dfrac{\mathrm{d}u}{\mathrm{d}r}g(r)\mathrm{d}r$。



### 12. Van der Vaals方程

#### 12.1 微扰理论初步

假设讨论[实际气体](#12. 真实气体与液体)时粒子间的势能可以写成$U_0(\pmb r_i)+U_1(\pmb r_i)$，其中$U_1$是比$U_0$远小的量，称为**微扰项**。

令构型配分函数的主要部分$Z_N^{(0)} = \int\mathrm{e}^{-\beta U_0(\pmb r_1,\cdots \pmb r_N)}\mathrm{d} \pmb r_1 \cdots \mathrm{d}\pmb r_N$，则$Z_N = Z_N^{(0)}\lang \mathrm{e}^{-\beta U_1} \rang_0$。下标0表示在Hamilton量中势能取为$U_0$下的正则系综平均。此时Hemholtz自由能写为$A(N,V,T) = -\dfrac{1}{\beta}\ln(\dfrac{Z_N^{(0)}}{N!\lambda^{3N}})-\dfrac{1}{\beta}\ln \lang \mathrm{e}^{-\beta U_1} \rang_0$。对微扰项做Taylor级数展开保留到三阶得到下面的结果：

$\color{purple} A = -\dfrac{1}{\beta}\ln(\dfrac{Z_N^{(0)}}{N!\lambda^{3N}}) + \lang U_1\rang_0 -\dfrac{\beta}{2}(\lang U_1^2\rang_0-\lang U_1\rang_0^2) + \dfrac{\beta^2}{6}(\lang U_1^3\rang_0-3\lang U_1 \rang_0  \lang U_1^2 \rang_0 + 2\lang U_1\rang _0^3) + \cdots$

#### 12.2 微扰理论对Van der Vaals方程的推导

若上一节中的$U_0$和$U_1$均为成对势能，则可推出$\color{darkblue} \lang U_1\rang_0 =\dfrac{\rho^2 V}{2}\int_0^{\infty}4\pi r^2u_1(r)g_0(r)\mathrm{d}r$。 

若分子半径为$\sigma$，则$u_0$在$|\pmb r_i - \pmb r_j|\le \sigma$时为$\infty$，其余为$0$。则考虑体积修正，$Z_N^{(0)}=(V-Nb)^N$。式中$b=\dfrac{2}{3}\pi \sigma^3$。

与$u_0$相应的径向分布函数$g_0(r)=\begin{cases}0,r\le\sigma\\1,r>\sigma\end{cases}$。代入一阶微扰项得到$A^{(1)}=-aN\rho$，其中$a=-2\pi\int_\sigma^{\infty}u_1(r)r^2\mathrm{d}r$。

由压强$P=-\dfrac{\part A}{\part V}$得到Van der Vaals方程$P=\dfrac{NkT}{V-Nb}-\dfrac{aN^2}{V^2}$。

低密度近似下有$\color{green}\dfrac{P}{kT}\approx \rho + \rho^2B_2(T)$，式中$B_2(T)=b-\dfrac{a}{kT}=\dfrac{2\pi\sigma^3}{3}+\dfrac{2\pi}{kT}\int_\sigma^{\infty}u_1(r)r^2\mathrm{d}r$称**第二Virial系数**。

#### 12.3 临界现象初步

对Van der Vaals气体做不同的等温线，可见它们有些会出现$P-V$图不单调的情况。体积随压力不单调的变化说明已经发生了相变。变化$T$时，从发生相变到不发生相变的拐点叫**临界温度**$T_c$。$P$对$V$一阶导和二阶导都为0的点$(P,V)$为**临界点 (critical point)**。

对临界现象可有三个**平均场指数**来描述，定义如下：

$\color{darkblue}\alpha$：临界点时$C_V \sim |T-T_c|^{-\alpha}$；$\color{darkblue}\gamma$：临界点时$\kappa_T= -\dfrac{1}{V}\dfrac{\part V}{\part P}\sim |T-T_c|^{-\gamma}$；$\color{darkblue} \delta$：临界点时$\dfrac{P}{kT}\sim C+|T-T_c|^{\delta}$。

可以证明Van der Vaals气体的$\alpha = 0,\gamma = 1,\delta = 3$。



### 13. 自由能的计算

#### 13.1 微扰方法

考虑$\mathcal{A}$和$\mathcal{B}$两个微观状态，反应由$\mathcal{A}$到$\mathcal{B}$。例如$\mathcal{A}$表示未相互作用的酶和底物，$\mathcal{B}$表示它们的复合物。

反应的Hemholtz自由能为$A_{\mathcal{A}\mathcal{B}}=-RT\ln(\dfrac{Q_{\mathcal{B}}}{Q_\mathcal{A}})=\color{purple} -kT\lang \mathrm{e}^{-\beta (U_\mathcal{B}-U_{\mathcal{A}})}\rang_{\mathcal{A}}$，该式称**自由能微扰公式**。它表示我们从$\mathcal{A}$的正则系综中选择状态，然后简单地将势能换成$U_\mathcal{B}-U_\mathcal{A}$。该方法的缺点是如果二者势能相差太大，则系综平均内的指数项将过小，不能满足计算需求。所以适合势能差较小的体系。

其改进措施是构造多个中间过程$\mathcal{A}\rightarrow \alpha \rightarrow \mathcal{B}\,\,(\alpha = 1,2,\cdots M)$，$A_{\mathcal{A}\mathcal{B}}=-kT\sum\limits_{\alpha =1}^{M-1}\ln \lang \mathrm{e}^{-\beta\Delta U(\alpha,\alpha+1)}\rang_{\color{blue} \alpha}$，将中间过程取合适可尽可能减少指数项过小的问题。注意每次要更换系综平均的参考状态$\color{blue} \alpha$。

#### 13.2 浸渐变换方法

由于自由能是状态函数，我们可以参考微扰法的改进，将状态缓慢从$\mathcal{A}$变成$\mathcal{B}$。设$U(\pmb r,\lambda) = f(\lambda)U_\mathcal{A}(\pmb r)+ g(\lambda )U_\mathcal{B}(\pmb r)$，其中$\lambda$从0变为1时，$U$从$U_\mathcal{A}$变为$U_\mathcal{B}$。由此对$f,g$的限制条件是$f(0)=1, f(1)=0;g(0)=0,g(0)=1$，除了这些端点条件外$f$与$g$任意连续变化。

简单推导可知自由能变化为$\color{purple} A_{\mathcal{A}\mathcal{B}}=\int_0^1\lang \dfrac{\part U}{\part \lambda}\rang_{\color{blue} \lambda} \mathrm{d}\lambda$，其中下标为$\color{blue} \lambda$表示对能量$H(\pmb r,\textcolor{blue} {\lambda})$进行系综平均。

特别地，当$f(\lambda) = 1-\lambda,g(\lambda) = \lambda$时，$A_{\mathcal{A}\mathcal{B}}=\int_0^1\lang U_{\mathcal{B}}-U_{\mathcal{A}}\rang_{\lambda} \mathrm{d}\lambda$。该式是等容可逆过程下Hemholtz自由能变是做的最小功的统计表述。

#### 13.3 Jarzynski等式与非平衡方法

考虑从$\mathcal{A}$转移到状态$\mathcal{B}$的系综，其中对每一个微观态$\pmb x$施加了功$\mathcal{W_{AB}}(\pmb x)$。由于微观状态在过程中任何时候的变化均取决于初始状态$\pmb x_0$，则总功$W_{\mathcal{AB}}=\lang \mathcal{W_{AB}}(\pmb x_0)\rang_\mathcal{A} \ge A_{\mathcal{AB}}$。虽然不可逆过程看似阻碍了用于自由能这一状态函数的计算，但是Jarzynski证明了$\mathrm{e}^{-\beta A_{\mathcal{AB}}}=\mathcal{\lang \mathrm{e}^{-\beta W_{AB}}\rang _A}$。这是非平衡态计算自由能方法的前提。

#### 13.4 反应坐标

根据研究反应的需求，可以选取体系的少部分广义坐标$q_i=f_i(\pmb r_1,\cdots \pmb r_N)(i=1,2,\cdots n)$来描述过程的进程。

此时反应坐标处在$q_i = s_i$时的概率密度$P(s_1,\cdots,s_n)= \dfrac{C_N}{Q(N,V,T)}\int \prod\limits_{i=1}^{n}\delta(f_{i}(\pmb r)-s_i)\mathrm{e}^{-\beta H(\pmb r,\pmb p)}\mathrm{d}^N\pmb r\mathrm{d}^N\pmb p$。

这些坐标下的Hemholtz自由能超平面方程自然为$A(s_1,\cdots s_n)=-kT\ln P(s_1,\cdots,s_n)$。

反应坐标的实例：

> 多肽折叠拉式构象图中的二面角$\phi,\psi$；
>
> 简单解离反应$AB\rightarrow A+B$里的两分子距离$r=|\pmb r_1-\pmb r_2|$；
>
> 若特殊条件改变解离反应的机理，相对位移的角度$\theta$和$\varphi$也会成为有作用的反应坐标。

#### 13.5 反应坐标系综方法

只有一个反应坐标时，$P(s)= \dfrac{C_N}{Q(N,V,T)}\int \delta(q_1(\pmb r)-s)\mathrm{e}^{-\beta H(\pmb r,\pmb p)}\mathrm{d}^N\pmb r\mathrm{d}^N\pmb p$，$A(s)=-kT\ln P(s)$。

则从起始反应坐标$s^{(i)}$到终止反应坐标$s^{(f)}$，自由能变化为$\Delta A=\int_{s^{(i)}}^{s^{(f)}}-\dfrac{kT}{P(s)}\dfrac{\mathrm{d}P(s)}{\mathrm{d}s}$。

对相空间的坐标进行**正则变换**，变换后的**广义坐标中恰有一个反应坐标**，即$\mathrm{d}^N\pmb p\mathrm{d}^N\pmb r=\mathrm{d}^{3N} p\mathrm{d}^{3N} q$，$H(\pmb p, \pmb r)\rightarrow \tilde{H}(p,q)$。最终得到的结论是$A(s^{(f)})=A(s^{(i)})+\int_{s^{(i)}}^{s^{(f)}}\lang \dfrac{\part \tilde{H}}{\part q_1} \rang_s^{\mathrm{cond}}\mathrm{d}s$。其中$\color{deeppink}\lang \dfrac{\part \tilde{H}}{\part q_1} \rang_s^{\mathrm{cond}}\overset{\mathrm{def}}{=}\dfrac{\lang \dfrac{\part \tilde{H}}{\part q_1}\delta(q_1-s)\rang}{\lang \delta(q_1-s)\rang}$定义了新的系综。

由于正则动量寻找与重新表示Hamilton量的复杂性，可以考虑分离动量项，只关注广义坐标的变换和势能项的表示。令$q=q(\pmb r)$，关联二者的Jacobi行列式为$J(q) = \dfrac{\part (\pmb r_1\cdots\pmb r_N)}{\part (q_1,\cdots q_{3N})}$，$\tilde{U}(q)=U(\pmb r)$。则推导可得$A(s^{(f)})=A(s^{(i)})+\int_{s^{(i)}}^{s^{(f)}}\lang \dfrac{\part \tilde{U}}{\part q_1}-kT\dfrac{\part \ln J(q)}{\part q_1} \rang_s^{\mathrm{cond}}\mathrm{d}s$。
