# MWD–地震 物理约束岩石强度融合

通过融合两路**独立反演、互补观测**的信息，为露天矿爆破/岩体刻画构建三维岩石强度场 \(S(x,y,z)\)：

- **MWD**（随钻测量）——钻孔沿线、分辨率高的力学响应 → 单轴抗压强度 UCS。
- **三维地震**——区域连续的结构信息 → 波阻抗 \(\mathrm{AI}=\rho\cdot V_p\)。

```
        MWD ─▶ PG-GPR UCS ─▶ 三维回归克里金 ─▶ S_MWD, σ_MWD
                                                              ╲
 三维地震 ─▶ 波阻抗反演 ─▶ AI→UCS 标定 ─▶ S_Z, σ_Z
                                                              ╱
     共定位协克里金（Doyen 贝叶斯更新 / 外漂移克里金） ─▶ S_fused, σ_fused
                                              │
                          三维强度体 ─▶ 水平切片（如 z = −20 m）
```

这**不是**端到端黑盒网络，而是**双分支独立反演 + 物理约束的后期融合**：公开数据中不存在共定位的 `MWD + UCS + 三维地震 + AI`。每一分支先在可复现的公开风格数据上验证，融合再在共定位合成矿山真值上定量评价。

完整公式、推导与钻孔数量实验见 **[docs/技术说明.md](docs/技术说明.md)**。图件保存在 [`docs/images/`](docs/images/)。

## 为什么用共定位协克里金，而不是把两张图画平均？

稀疏准的钻孔和全区连续的地震，是储层建模里同一类问题。Xu 等（SPE 24742）的做法是**外漂移克里金**（井为硬数据，地震为漂移）；Doyen 等（SPE 36498）把共定位协克里金写成对井克里金的贝叶斯更新——只需要克里金方差和井–地震相关系数 \(\rho\)，孔上 \(\sigma_M\to 0\) 时估计自动等于井，远处缩向地震回归。这与把 \(S_M\)、\(S_Z\) 当成独立观测做逆方差混合不同：两路共用同一批标定井，独立假设会把孔上真值冲掉。

实现上：两口井及以上用 Doyen 更新（主变量 \(S_M\)，共定位次变量 \(S_Z\)，\(\rho=\mathrm{corr}(\mathrm{UCS},\mathrm{AI})\)）；单口井时 GP 标定会过拟合，改用波阻抗本身做外漂移克里金。孔轨迹体素始终写回 MWD 点估计。

## 项目结构

```
├── datasets/        # 合成三维模型 + 共定位合成矿山基准
├── dataio/          # 五模块数据仓加载器（MWD-UCS / MWD 空间 /
│                    #   Marmousi2 / Penobscot / 合成矿山）
├── mwd/             # MWD 特征（Teale 比能）+ PG-GPR UCS 模型
├── inversion/       # 子波、反射系数、正演、模型基 / 稀疏脉冲 /
│                    #   PyLops 叠后反演
├── preprocessing/   # SEG-Y (segyio)、LAS (lasio)、深度↔时间标定
├── geostats/        # 三维普通克里金与回归克里金（均值 + 方差）
├── fusion/          # AI→UCS 标定（GP）+ 共定位协克里金（Doyen / KED）
├── validation/      # R²、RMSE、MAE、盲井检验
├── visualization/   # 切片、剖面、融合面板、PyVista 三维
├── examples/        # 基准、可视化、文档出图脚本
├── docs/            # 技术说明 + 已跟踪的图件（docs/images/）
├── scripts/         # download_datasets.py（真实数据，需手动下载）
└── tests/           # pytest 套件
```

## 安装

```bash
python3 -m pip install --user --break-system-packages -e ".[dev]"
```

（Cloud Agent 环境的 `install` 步骤即为此命令。本地也可：`python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"`。）

中文图题需要 CJK 字体。可选安装：`scripts/setup_fonts.sh`（需 sudo），或把 `AI_INVERSION_CJK_FONT` 指到某个 `.ttf`/`.otf`。未安装时图仍能出，汉字可能显示为方框。

## 运行

端到端双分支融合基准（完全离线，带共定位真值）：

```bash
python3 examples/run_fusion_benchmark.py --outdir results
```

在 `results/` 中生成：

- `S_fused.npy`、`sigma_fused.npy`、`S_true.npy`
- `figures/fusion_panels.png` — (a) MWD 强度，(b) 地震波阻抗，(c) 阻抗导出强度，(d) 融合强度，(e) 融合不确定度
- `figures/fused_strength_elev_{-8,-20,-32}.png` — 水平切片
- 四种方案的指标表（仅 MWD、仅地震、简单加权、共定位协克里金）以及 **空间盲孔** MWD 留出得分

再生成本仓库跟踪的中文技术图件（融合优势对比 + 1→12 口钻孔系列）：

```bash
python3 examples/run_docs_figures.py --outdir docs/images
```

地震单分支 Stage I 验证（`AI → 地震 → AI`，带真值）：

```bash
python3 examples/run_stage1_synthetic.py --outdir results
```

## 可视化套件

```bash
python3 examples/run_visualizations.py --outdir results/viz
```

在 `results/viz/` 中生成：

- `impedance_slice.png`、`mwd_strength_slice.png`、`fused_strength_slice.png` —
  报告风格水平切片（viridis 填充 + 虚线等值线 + 钻孔靶心 + 科学色标）。
- `*_3d.png` — 静态三维渲染（PyVista，离屏）。
- `interactive/*.html` — **可交互**三维页面（Plotly）：旋转/缩放/切片，浏览器中打开即可。

## 交互式网页（无后端，纯静态）

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fc76d3656e%2FAcoustic-impedance-inversion&root-directory=frontend&project-name=mine-fusion-viz&repository-name=mine-fusion-viz)

`frontend/` 是 Vite + React + TypeScript + WebGL2 应用，随仓库发布**固定数据集**，全部在浏览器中计算：可拖动 X/Y/Z 切片、半透明三维体 + 可移动剖面、多井测井曲线、自定义强度色标，以及 **Pyodide + matplotlib 在浏览器内出中文出版图**。部署是 100% 静态前端（Root Directory = `frontend`），无后端、无 serverless。

```bash
python3 scripts/export_frontend_dataset.py   # 重新导出固定数据集（确定性）
npm --prefix frontend install
npm --prefix frontend run dev                # http://localhost:5173
```

## 测试

```bash
python3 -m pytest
```

## 两阶段数据路线

| 阶段 | 数据 | 用途 | 状态 |
| --- | --- | --- | --- |
| **算法验证** | Marmousi2 / 合成 | 地震→AI，MWD→UCS→克里金 | 可离线运行 |
| **共定位基准** | 合成矿山（50×80×40 m，约 10 个台阶） | 双分支融合 vs 真值 | 已实现 |
| **现场数据** | Penobscot 3D + Hansen MWD + MWD-UCS | 真实数据验证 | I/O 与加载器已实现 |

公开数据在物理上是**拆开的**（有 MWD 无地震，或有地震无 MWD）。因此各分支在类真实公开数据上验证，融合在物理约束的共定位合成基准上评价，并保留现场数据加载器。获取真实数据：

```bash
python scripts/download_datasets.py --dataset marmousi2 --dest data/marmousi2
python scripts/download_datasets.py --dataset penobscot --dest data/penobscot
```

五个数据模块及来源见 `data/README.md`。
