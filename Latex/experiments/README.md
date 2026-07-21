# 实验归档说明

`_template/` 是运行批次模板，不代表一次真实实验。开始新实验前，将它复制并重命名为：

```text
YYYYMMDD_HHMMSS_算例_目的
```

例如：

```text
20260721_120500_ieee118_nr_fdlf
```

真实实验目录建立后：

1. 用实际配置完整覆盖 `config.yaml`；
2. 将图像和原始表格分别写入 `figures/` 与 `tables/`；
3. 将标准输出和错误输出写入 `console.log`；
4. 填写 `metadata.yaml` 和 `notes.md`；
5. 归档目录不再被后续程序运行覆盖。

