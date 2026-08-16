# DESIGN.md - 视觉系统(锁定自原片,禁止 AI 默认审美)

> 所有 token 以原片画面为唯一权威。禁:AI 紫渐变、Inter+slate 默认、玻璃拟态滥用、装饰性 status dot、em-dash(—/–)出现在任何可见文案、滚动提示("Scroll ↓")、section 编号 eyebrow。

## Dials

- `DESIGN_VARIANCE: 8` - 每页构图服从原帧,不对称为主
- `MOTION_INTENSITY: 6` - 氛围动效(粒子/雨/扫描线/辉光呼吸),transform/opacity only
- `VISUAL_DENSITY: 4` - 画面主导,UI 元素克制

## 色板(7 个色调家族,按页锁定)

| Token | Hex | 用途 |
|---|---|---|
| `--t-grn` | `#3dff7a` | 终端绿字/猫清单/boot |
| `--t-grn-dim` | `#1f8a4c` | 终端次级文字 |
| `--bg-term` | `#040705` | 终端黑场(非纯黑) |
| `--w-blu` | `#7ea8d8` | 病房冷蓝高光 |
| `--w-blu-deep` | `#0d1622` | 病房深色部 |
| `--c-org` | `#ff9a3c` | 夕阳/晨光/大钟暖橙 |
| `--c-org-deep` | `#b8501f` | 暖橙暗部 |
| `--f-cyn` | `#46d4e8` | 花田荧光蓝/数据涡旋 |
| `--f-cyn-deep` | `#0a2733` | 荧光蓝暗部 |
| `--alert-red` | `#ff2d3f` | 倒计时/glitch 警示红 |
| `--chat-green` | `#95ec69` | 希达气泡(右侧) |
| `--chat-blue` | `#8fc6ee` | Z-ERO 气泡(左侧) |
| `--chat-bg` | `#f7f7f5` | 聊天白底(非纯白) |
| `--ink` | `#0a0a0a` | 标题卡近黑 |
| `--paper` | `#f4f4f2` | 标题卡近白 |

规则: 一页一个色调家族;跨页不混用 accent;终端页绿、病房页蓝、夕阳页橙、花田页青、警示页红、聊天页白、标题/名单/升华页黑白。

## 字体

- 终端/HUD/字幕英: `"JetBrains Mono", "Cascadia Mono", ui-monospace, Consolas, monospace`
- 聊天/正文 CJK: `-apple-system, "PingFang SC", "Microsoft YaHei", "Noto Sans SC", sans-serif`
- 标题卡中文: 思源黑体/系统黑体加粗 + 英文 mono 大写字距拉开 (`letter-spacing: 0.5em`)
- 不用 Inter 作为"默认选择";不引 Google Fonts link(本地自托或系统栈)

## 形状

- 聊天气泡: `border-radius: 18px`(单侧小角),全站唯一圆角尺度;其余元素直角(radius 0)
- 终端面板/HUD: 直角 + 1px 内描边 `border: 1px solid color-mix(in srgb, accent 40%, transparent)`

## 材质 / FX 语言(共享组件)

- `FilmGrain` - fixed inset-0 pointer-events-none 噪点层(SVG feTurbulence,opacity 0.06)
- `Scanlines` - 终端/监控页水平扫描线(repeating-linear-gradient, 3px 周期,opacity 0.08)
- `CRTFlicker` - 终端页亮度微闪(opacity 0.97↔1, 4s 随机)
- `GlitchRGB` - 红色故障页 RGB 通道分离(clip-path 错层 + translate,偶发 200ms)
- `SubtitlePlate` - 原片字幕板: 底部居中,中文白 + 英文 mono 小号,黑场页带 `text-shadow` 描边
- `LetterboxBars` - 2.39:1 黑边(仅"电影画幅"页:空镜/全景),UI 全屏页(终端/聊天/HUD)不用
- `Typewriter` - 终端文字逐字上屏
- `PetalsLoader` - 菊花瓣 loading(白瓣旋转环,用于 13/52 页)
- `CountdownDigits` - 红色等宽数字跳秒

## 动效曲线

- 标准: `cubic-bezier(0.16, 1, 0.3, 1)` (easeOutExpo 系)
- 氛围循环: `ease-in-out`,幅度小(辉光 opacity 0.6↔1,漂浮 translateY ±8px)
- glitch: `steps(2, end)` 硬切
- 全部循环动效 gated `@media (prefers-reduced-motion: no-preference)`;reduce 时静帧

## z-index 尺度

`0` 画面层 / `10` 氛围层(雨/粒子) / `20` UI 内容 / `30` 字幕板 / `40` 导航 / `50` grain+scanline 覆盖

## 导航(唯一 chrome)

- `/` 胶片索引页: 61 格缩略图网格(黑场,mono 编号+场景名)
- 场景页: 底部细进度条 + 左/右箭头热区;键盘 ←/→ 切换,Esc 回索引
- 导航元素 opacity 0.35,hover 时 1;不抢画面

## 每页构建顺序

构图(布局/画幅/主体位置) → 色彩(锁定家族色板) → 字体文案(原片文字逐字) → 动效(最后,可裁)
