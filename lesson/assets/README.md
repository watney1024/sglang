# 图片规范

图由用户自行生成，本目录只存放 **prompt 与落盘位置**，不存二进制以外的中间产物。

## 目录约定

```
assets/
  README.md              本文件
  prompts/03.md          第 03 课的图 prompt（与课号同名）
  03/fig-01-process-map.png
  03/fig-02-overlap-vs-normal.png
  ...
```

- 每课一个 `prompts/NN.md`，每张图一条 prompt，文件名与 `assets/NN/` 下的图片一一对应。
- 图片未生成时 Markdown 只显示 alt 文本，不影响阅读顺序，所以可以先写引用后补图。

## 统一风格前缀

所有图共用同一段风格描述，已原样内联在每条 prompt 的 `STYLE:` 行（25 条完全一致），保证 19 讲视觉一致：

```
Flat vector technical diagram for an engineering textbook, light background #FAFAFA,
uniform 2px strokes, rounded rectangles, orthogonal connectors, no gradients,
no drop shadows, no 3D, no skeuomorphism, no human figures, no decorative patterns,
generous whitespace, sans-serif labels, crisp high-contrast lines, 16:9, 2x resolution.
```

## 语义配色

同一含义在全书用同一颜色，读者第二次见到就不用重新解码：

| 颜色 | 色值 | 含义 |
|---|---|---|
| 蓝 | #2563EB | 控制流：请求、调度、决策 |
| 橙 | #EA580C | 内存：KV Cache、token 预算 |
| 绿 | #059669 | 输出：token、文本、响应 |
| 紫 | #7C3AED | GPU 计算：forward、kernel |
| 灰 | #6B7280 | 可选路径、旁路、已废弃 |

## 关于文字

图像模型对文字渲染不可靠。两条应对：

1. prompt 里只放 3–6 个短英文标签（大写、2 个词以内）。
2. 中文标签与公式后期用 Figma / draw.io 覆盖，所以每条 prompt 下面都附了「标签清单」，按清单补即可。

## 反向提示

```
no gradients, no glow, no 3D, no photorealistic, no clipart, no emoji,
no gibberish text, no watermark, no dense paragraphs of text
```
