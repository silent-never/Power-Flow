# Power-Flow 报告资料库

本目录用于集中管理报告正文、候选图片、最终图片、实验快照、参考文献及编译产物。旧的 `text.tex`、`text.pdf` 等文件暂时保留，不参与新目录结构的编译。

## 目录结构

```text
Latex/
├─ README.md
├─ report/                         # 报告源码
│  ├─ main.tex
│  ├─ sections/
│  └─ styles/
├─ assets/                         # 报告素材
│  ├─ figures/
│  │  ├─ selected/                # 正文最终采用的图片
│  │  ├─ candidates/              # 等待比较的候选图片
│  │  └─ external/                # 外部图片及其来源说明
│  ├─ tables/
│  │  ├─ selected/
│  │  └─ candidates/
│  └─ diagrams/                   # 手工流程图和结构图
├─ experiments/                   # 按运行批次保存原始结果
│  └─ _template/                  # 新实验目录模板
├─ references/                    # 论文、书籍、标准和阅读笔记
│  └─ references.bib
├─ provenance/                    # 图片和文献的来源索引
└─ build/                         # PDF、aux、log 等编译产物
```

## 编译方法

在 `Latex/report` 目录执行：

```powershell
latexmk -xelatex "-outdir=../build" main.tex
```

生成的 PDF 位于 `Latex/build/main.pdf`。辅助文件也集中在 `build/`，不会污染正文目录。

清理编译产物：

```powershell
latexmk -C "-outdir=../build" main.tex
```

## 图片选择流程

1. 每次运行程序时，复制 `experiments/_template`，重命名为：

   ```text
   YYYYMMDD_HHMMSS_算例_算法或目的
   ```

   例如：

   ```text
   20260721_120500_ieee118_nr_fdlf
   ```

2. 将本次运行的全部图片、数据、配置快照和控制台输出保存在该目录中。
3. 有使用价值但尚未确定的图片复制到 `assets/figures/candidates/`。
4. 最终选定的图片复制到 `assets/figures/selected/`。
5. 在 `provenance/figure_manifest.csv` 登记最终图片的来源、配置和选择理由。
6. 报告正文只引用 `selected/`，不直接引用某次实验目录或程序临时输出目录。

## 图片命名建议

推荐使用稳定的英文语义名称：

```text
fig_01_algorithm_framework.pdf
fig_02_all_cases_summary.png
fig_03_ieee118_nr_fdlf_comparison.png
fig_04_ieee118_robustness.png
```

不要在最终文件名中加入 `final`、`new`、`latest2` 等难以追踪的字样。版本差异应由实验目录和来源清单记录。

## 参考文献流程

1. PDF 分别放入 `references/papers/`、`books/` 或 `standards/`。
2. 文献的 BibTeX 信息写入 `references/references.bib`。
3. 阅读笔记放入 `references/notes/`，文件名建议与引用键一致。
4. 在 `provenance/reference_index.md` 记录文献用途和报告章节。
5. 正文使用 `\cite{引用键}`，不要手工输入参考文献编号。

## 编写约定

- `main.tex` 只负责封面、摘要、目录、章节装配和文献表。
- 正文内容写在 `report/sections/`。
- 页面、字体、图表和图片路径统一在 `report/styles/report-style.sty` 中配置。
- `label` 使用 `sec:`、`fig:`、`tab:`、`eq:` 前缀。
- `build/` 可以随时清理；`experiments/` 中的原始记录不应被编译脚本覆盖。
