## Notes of Statistical Mechanics

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



### 4. 平衡系综概述

**平衡系综 (equilibrium ensemble)**指相空间分布函数**不显含时间**的系综。由$\dfrac{\part f}{\part t} = 0$可得$[f, H] = 0$。因此，$f$一定是$H$的函数$F(H(\pmb x))$。

分布函数在相空间上面的积分$\color{deeppink}\mathcal{F} \overset{\mathrm{def}}{=} \int F(H(\pmb x)) \mathrm{d} \pmb x$称作**配分函数(partition function)**，等于微观状态数总和。

一个宏观物理量$A$的测量值则为**系综平均**$\color{purple}\langle A\rangle = \dfrac{1}{\mathcal{F}} \int A(\pmb x)F(H(\pmb x)) \mathrm{d} \pmb x$。

配分函数和系综平均是系综理论的核心概念。



 ### 5. 微正则系综

#### 5.1 基本概念与配分函数

**等概率假设(postulation of equal a priori probabilities)**：处于平衡的**孤立**宏观系统中，所有的微观状态都是以等可能的概率出现的。它是下面讨论微正则系综的出发点。

**微正则系综**指在系统处于能量$E$、粒子数$N$和体积$V$时，其中的系统在相空间的分布函数为$\textcolor[rgb]{0.54,0,0.08} {F(H(\pmb x)) = \delta (H(\pmb x) - E)}$的系综。假设能量处于$E$和$E+\Delta E$的薄球壳之间，其配分函数为$\color{purple}\Omega(N,V,E)  = \dfrac{1}{N! h^{3N}} \int\limits_{H \in [E,E+\Delta E]}d\pmb x$ $= \dfrac{\Delta E}{N! h^{3N}} \int_{\Omega}\delta (H(\pmb x)- E) d\pmb x $ 。

> 关于配分函数前系数的说明：
>
> $N!$是因为交换系统内任意两个粒子，所观察到的状态不发生变化；
>
> $h^{3N}$是因为量子力学认为每个**可观测的微观状态**占据$(\Delta r\Delta p)^{3N} = h^{3N}$的相空间体积。
>
> 因此观测到的系统数+1对应着$\dfrac{1}{N!h^{3N}}\Delta \pmb x$。

#### 5.2 热力学量

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



### 6. 经典Virial定理

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



### 7. 正则系综

#### 7.1 基本概念

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

#### 7.2 配分函数与热力学量

正则系综的配分函数为$\color{purple} Q(N,V,T) = \dfrac{1}{N!h^{3N}}\int \mathrm{e}^{-H(\pmb x)/kT} \mathrm{d} \pmb x$。

Hemholtz自由能用正则配分函数表示为$\color{red} A(N,V,T) = kT\ln Q(N,V,T)$

令$\beta = \dfrac{1}{kT}$，则其它热力学量均可从配分函数来表示：

$E =- \dfrac{\part \ln Q}{\part \beta}, P = kT\dfrac{\part \ln Q}{\part V}, S=k \ln Q + \dfrac{E}{T}, C_V = k\beta^2\dfrac{\part^2\ln Q}{\part \beta^2} $

正则和微正则系综的配分函数可用Laplace变换关联：$\Omega(N,V,E)\fallingdotseq \tilde{\Omega}(N,V,\beta) = Q(N,V,T)$。

#### 7.3 正则系综的Virial定理

**正则系综平均下的Virial定理**最终表达式为$\color{darkblue}\langle x_i \dfrac{\part H}{\part x_j} \rangle = kT\delta_{ij} + \int x_i \mathrm{e}^{\frac{-H(\pmb x)}{kT}}|^{x_j=\infty}_{x_j = -\infty} \mathrm{d}S_j$，右边的积分不总为0，但一般研究的有界体系会满足$x_i \rightarrow \pm \infty,H(\pmb x)\rightarrow +\infty$（即势阱），所以微正则系综Virial定理的表述大都适用。

#### 7.4 正则系综的能量涨落

从正则系综满足的条件可知，其能量并非不变，而有**涨落 (perturbation)**。我们以能量的标准差来描述正则系综能量的不确定性，即$\color{deeppink} \Delta E \overset{\mathrm{def}}{=} \sqrt{\langle H^2 \rangle - \langle H \rangle^2}$。用系综平均的定义与能量和配分函数关系即有$\color{purple}  \Delta E = \sqrt{kT^2 C_V}$。

热容和能量均为广延量，因此$\dfrac{\Delta E}{E} \sim N^{-1/2}$。可见粒子数越多，相对涨落越来越小。



### 8. 温度和压力的微观意义

令$N$个分子总动能$\color{darkblue} K (\pmb p) = \sum\limits_{i=1}^{N}\dfrac{\pmb p_i^2}{2m}$，由Virial定理易得温度的统计意义$\color{purple} T = \dfrac{2}{3Nk}\langle K(\pmb p_i)\rangle$。

令函数$\color{darkblue} \Pi(\pmb p, \pmb r) = \dfrac{1}{3V}\sum\limits_{i=1}^{N}[\dfrac{\pmb p_i^2}{2m_i} + \pmb r_i \cdot \pmb F_i(\pmb r)]$，则压力的统计意义为$\color{purple} P = \langle \Pi(\pmb x) \rangle$。推导要略如下：

> 相空间中引入变换$\pmb s_i = V^{-1/3}\pmb r_i, \pmb \pi_i = V^{1/3} \pmb p_i$。可知$\pmb s_i$表示粒子的相对位置，在体积发生微小变化时，$\pmb s_i$不发生变化。由Liouville定理可知相体积元$\mathrm{d}\pmb \pi \mathrm{d}\pmb s = \mathrm{d}\pmb r\mathrm{d}\pmb p$在演化中不变，因此$\pmb \pi_i$也不变。
>
> 因此重新改写$Q(N,V,T) = C_N\iint\exp(-\dfrac{1}{kT}[\sum\limits_{i=1}^{N}\dfrac{V^{-2/3}\pmb \pi_i^2}{2m_i}]+U(V^{1/3}\pmb s))\mathrm{d}\pmb \pi \mathrm{d}\pmb s$。
>
> 代入压力的表达式$P = kT\dfrac{\part \ln Q}{\part V}$，注意$\pmb F_i =-\dfrac{\part U}{\part \pmb r_i}$，最终可得$P = \langle \Pi(\pmb x) \rangle = \langle  -\dfrac{\part H}{\part V} \rangle$。$\square $

备注：

1. 上面提到的$K(\pmb p)$和$\Pi (\pmb x)$等函数称为**估计值 (estimator)**，其系综平均则为物理量的观测值。
2. 推导可得$\sum\limits_{i=1}^{N} \pmb r_i \cdot \pmb F_i = \dfrac{1}{2}\sum\limits_{i\neq j}\pmb r_{ij} \cdot \pmb F_{ij}$，式中$\pmb r_{ij}$是粒子的相对位移。可见这是表示相互作用的量。

### 9. 等温等压系综

#### 9.1 基本概念、配分函数与热力学量

对$A(N,V,T)$做Legendre变换得到$\color{deeppink} G(N,P,T) \overset{\mathrm{def}}{=} A(N,V,P(T)) + PV$，即**Gibbs自由能**。它是下面讨论的等温等压系综的基本能量。热力学公式有$\mathrm{d}G = -S\mathrm{d}T + P\mathrm{d}V + \mu\mathrm{d}N$。

仿照正则系综的导出方式，**等温等压系综 (Isothermal-isobaric Ensemble)**是在正则系综基础上用活塞保持同外压平衡得来。能量通过改变$PV$来交换，可类似用**自变量为$PV$，参数为$\beta$的Laplace变换**得到配分函数：

$\color{green} \Delta (N,P,T) = \dfrac{1}{V_0}\int_0^{\infty}Q(N,V,T)\exp(-\beta PV)\mathrm{d}V$，其中$V_0$为具有体积量纲的常数。

$\color{purple} \Delta (N,P,T)=\dfrac{1}{N!h^{3N}}\int_0^{\infty}\int \exp (-\beta (H(\pmb x)+ PV)) \mathrm{d}\pmb x \mathrm{d}V $。

Gibbs自由能用配分函数表示为$\color{red} G(N,P,T) = -kT \ln \Delta(N,P,T)$

其它热力学量：$\color{forestgreen} V=kT\dfrac{\part\ln \Delta}{\part P},\bar{H}=-\dfrac{\part}{\part \beta}\ln \Delta,C_P = k\beta^2 \dfrac{\part ^2 \ln \Delta}{\part \beta^2},S=k\ln\Delta + \dfrac{\bar{H}}{T}$。

其中， $\color{deeppink} \bar{H} \overset{\mathrm{def}}{=} \langle H\rangle + P\langle V\rangle$是等温等压系综中的焓变定义。

与正则系综能量涨落类似，等温等压系综平均下存在焓变涨落$\color{purple}  \Delta \bar{H} = \sqrt{kT^2 C_P}$。

#### 9.2 压力Virial定理和功Virial定理

以下考虑**等温等压系综平均**。令$-\dfrac{\part H}{\part V} = P_{int}$为压力在微正则和正则系综中的估计值。

**压力Virial定理**：$\langle P_{int} \rangle = P $，说明内外压强的测量值相等。

**功Virial定理**：$\langle P_{int} V \rangle +\color{darkblue} kT $ $= \langle P_{int} \rangle \langle V\rangle$。可认为在新加的一个自由度$V$上满足能量均分原理。



### 10. 系综应用1：理想气体

#### 10.1 系综处理热力学体系方法概论

> 1. 列出体系Hamilton量与系综对应的配分函数；
> 2. 将配分函数并最终化为相应自变量的表达式；
> 3. 通过配分函数、Virial定理或直接待入系综平均积分表达式等方法，求出热力学量和关系。

#### 10.2 微正则系综方法

下面以微正则系综方法处理理想气体为例，分析统计系综处理热力学问题的思路。

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

#### 10.3 正则系综方法

正则系综方法处理理想气体时，配分函数的形式格外简单：$Q(N,V,T)=\dfrac{1}{N!h^{3N}}\iint\exp(-\beta\sum\limits_{i=1}^{3N}\dfrac{p_i^2}{2m})\mathrm{d}^{3N}p_i\mathrm{d}^{3N}r_i = \color{darkblue} \dfrac{1}{N!}[\dfrac{V}{h^3}(2\pi m k T)^{3/2}]^{N}$

代入$S=k\ln Q + \dfrac{E}{T}$即得到$\color{purple} S(N,V,T) = Nk\ln[\dfrac{V}{h^3}(2\pi mkT)^{3/2}] + \dfrac{3}{2} Nk-k\ln N!$。下同。

#### 10.4 等温等压系综方法

配分函数$\Delta (N,P,T) = \dfrac{1}{V_0}\int_0^{\infty}Q(N,V,T)\exp(-\beta PV)\mathrm{d}V = \color{darkblue} [\dfrac{1}{h^3}(\dfrac{2 \pi m}{\beta})^{3/2}\dfrac{1}{\beta P}]^{N}\dfrac{1}{\beta P V_0}$

 代入$\bar{H} = -\dfrac{\part \ln \Delta}{\part \beta}$得到$\bar{H} = E + P\langle V \rangle = \dfrac{5N +1}{2}kT ,\,P\lang V\rang = (N+1)kT$。

由功Virial定理可得理想气体状态方程在等温等压系综中表达为$\lang P_{int} V\rang = NkT$。



### 11. 巨正则系综

#### 11.1 基本概念

**巨正则系综  (Grand Canonical Ensemble)**是控制变量为化学势$\mu$、体积$V$和温度$T$的系综。对$A(N(\mu),V,T)$做Legendre变换为$\tilde{A}(\mu,V,T)=A-\mu N = A - G = \color{deeppink} -PV$。$PV$是巨正则系综中的基本自由能。热力学公式为$\mathrm{d}(PV) = P\mathrm{d}V + S\mathrm{d} T + N \mathrm{d} \mu$。

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

#### 11.2 配分函数与热力学量

定义巨正则系综配分函数为$\color{purple} \mathcal{Z}(\mu,V,T) = \sum\limits_{N=0}^{\infty}\dfrac{1}{N!h^{3N}}\mathrm{e}^{\beta \mu N}\int \mathrm{e}^{-\beta H(\pmb x ,N)}\mathrm{d}\pmb x = \exp(\beta PV)$。
