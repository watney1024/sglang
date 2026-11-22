# SGLang 源码教程

以 `category.md` 为大纲、以仓库源码为唯一事实来源的 19 讲教程。

## 版本

本教程锚定 commit `450fb9da33`。SGLang 迭代很快，升级仓库后请先跑:

```bash
python lesson/tools/check_anchors.py
```

输出 `全部通过` 说明文档里的 file:line 仍然对得上；否则按提示逐条修订。

## 阅读方式

顺序读 03 打底，之后按需跳转。每篇开头有 **考点地图**，末尾有 **一页速记**；
赶时间可以只读这两节，再回头补代码主线。

## 统一模板

每篇基本按同一套九段组织（少数课按主题微调，例如 01 用"必知参数/常见坑"替代取舍表），便于横向对照:

1. TL;DR
2. 考点地图
3. 最小概念集 + 全景图
4. 代码主线（`file:line` + 短摘录）
5. 设计取舍表
6. 动手验证
7. 易错点
8. 面试问答（一句话 / 展开 / 追问）
9. 一页速记

## 目录

| 模块 | 课程 | 状态 |
|---|---|---|
| 01 基础与请求主链 | [01 快速上手](module-01/01-quickstart.md) / [02 体系架构](module-01/02-architecture.md) / [03 请求生命周期](module-01/03-request-lifecycle.md) | 完成 |
| 02 缓存、加载与调度 | [04 RadixAttention](module-02/04-radix-attention.md) / [05 模型加载](module-02/05-model-loading.md) / [06 Scheduler](module-02/06-scheduler.md) / [07 Chunked Prefill](module-02/07-chunked-prefill.md) | 完成 |
| 03 模型结构与精度压缩 | [08 量化基础](module-03/08-quantization-basics.md) / [09 AWQ](module-03/09-awq.md) / [10 Attention 变体](module-03/10-attention.md) | 完成 |
| 04 图执行与推测解码 | [11 CUDA Graph](module-04/11-cuda-graph.md) / [12 torch.compile](module-04/12-torch-compile.md) / [13 Triton](module-04/13-triton.md) / [14 推测解码](module-04/14-speculative.md) | 完成 |
| 05 分布式与 MoE | [15 并行策略](module-05/15-distributed.md) / [16 EP](module-05/16-moe-ep.md) / [17 EPLB](module-05/17-eplb.md) | 完成 |
| 06 服务化能力 | [18 PD 分离](module-06/18-pd-disaggregation.md) / [19 约束解码](module-06/19-constrained-decoding.md) | 完成 |

全书共 25 张配图，prompt 在 `assets/prompts/`，图未生成时只显示 alt 文本。

## 目录结构

```
lesson/
  category.md                 课程大纲
  module-01/ ... module-06/   19 篇正文
  assets/
    README.md                 图片规范（统一风格、语义配色）
    prompts/03.md             第 03 课配图 prompt
    03/                       生成的图片放这里
  appendix/repo-map.md        源码目录地图
  appendix/glossary.md        术语表
  appendix/interview-index.md 面试题反查表
  tools/check_anchors.py      锚点校验
```

## 配图

图不在仓库里，只存 prompt。每课一个 `assets/prompts/NN.md`，每条 prompt 对应一张图：

1. 复制 prompt 整段（已含统一风格前缀）到出图工具。
2. 按 prompt 下的「标签清单」在 Figma / draw.io 里校对或覆盖文字。
3. 存为 `assets/NN/fig-MM-<name>.png`，正文里的图片引用已经写好，放进去即可显示。

风格约定见 `assets/README.md`，核心是浅底、2px 描边、正交连线和一套语义配色。

## 写作约定

- 代码摘录不超过 15 行，只保留与结论相关的分支，省略处用 `...`。
- 引用格式 `path/to/file.py` 加 `:行号`；同一文件多次引用可简写为 basename + 行号。
- frontmatter 的 `anchors` 带符号名，用于检测代码漂移；正文引用只校验行号范围。
- 概念对比一律用表格，不用长段落。
