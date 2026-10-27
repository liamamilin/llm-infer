# 验证记录

`runs/` 是实验脚本每次运行的原始输出，默认被 git 忽略（可能含设备信息）。
确认无隐私信息后，把代表性的结果复制到 `public/`，章节正文只引用 `public/` 里的文件。

| 实验 | 状态 | 证据 |
|---|---|---|
| ch00 smoke | PASS（Apple M4 Pro / MPS） | [`public/ch00-smoke.json`](public/ch00-smoke.json) |
| ch01 assets | PASS | [`public/ch01-assets.json`](public/ch01-assets.json) |

其余章节：`NOT_RUN`。
