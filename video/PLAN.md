# 视频计划（Video，占 10%）

## 硬性要求

- [ ] **≤ 5 分钟**（与队伍人数无关）。目标时长 4:40，留出余量。
- [ ] **每个成员都出镜讲解，讲话时露脸**。
- [ ] **≤ 300 MB**。`package_deliverables.py` 按 300,000,000 字节检查。
- [ ] 放进队伍 zip。`video/` 下**只能有一个**视频文件（.mp4 / .mov / .mkv，含子目录），否则 `package_deliverables.py` 会报错。原始录像放在项目外的文件夹，只把最终压缩版放进来，命名为 `video\<TEAM_NAME>.mp4`。
- 语言：英文（TODO：确认课程是否有要求）。

材料（2026-11-01 由 Claude Code 起草，成员改写后使用）：讲稿 `video/SCRIPT.md`（3 人、2 人、1 人三个版本，按实际字数测的时间，每个数字的出处）；幻灯片 `video/slides.pdf`（源文件 `video/slides.tex`，在开头的 EDIT HERE 处填队名、成员、扩展负责人，关掉没人负责的扩展）；录制、拼接、压缩和最终检查 `video/RECORDING.md`。

## 5 分钟分镜（3 人版；2 人时合并相邻段落）

| 时间 | 内容 | 画面 | 讲者 |
|---|---|---|---|
| 0:00–0:20 | 队名、成员、Kaggle team name，一句话说明我们做了什么 | 全员同框或依次露脸 | 全员 |
| 0:20–1:00 | 问题：250 行、150 个特征、ROC AUC；为什么难（噪声多，public LB 只有约 395 行） | 1 张数据概览图 | A |
| 1:00–1:50 | 验证协议：5-fold × 20 repeats、选特征放在 pipeline 内、配对比较；E1 的关键图 | `reports/figures/e1_*` | E1 负责人 |
| 1:50–2:40 | 提交历程：CV vs public 散点图，2–3 个转折点和教训，一个与预期不符的地方（B003 的解释或 B005 的点预测；B004 只是被模型判为过于自信，private 还没公布，不能说被 LB 证伪） | `core_` 图，submission history 表 | B |
| 2:40–4:00 | 扩展：每个负责人约 25–40 秒，讲 Question → 一张图 → 结论 | `reports/figures/e2_*` / `e3_*` / `e4_*` | 各扩展负责人 |
| 4:00–4:35 | 最终方案，以及为什么按预先确定的规则选它；分数 | 最终模型的 config 摘要 | C |
| 4:35–4:50 | 一句话总结每人学到的东西 | 全员露脸 | 全员 |

扩展段只讲有人负责的扩展（每人一个，所以 1–3 个，每个 25–40 秒）；没人负责的扩展不单独讲，只作核心报告的支撑材料。E1 有人负责时紧接在验证协议后面讲（如上表），其余扩展放在提交历程之后。各段的实际时间和 2 人、1 人版本以 `video/SCRIPT.md` §1 为准。

英文语速约 130 词/分钟，全片讲稿控制在 600 词以内。讲稿在 `video/SCRIPT.md`（最长的 3 人版本 533 词；文本文件可以进 git）。

## 录制建议

逐步操作（Zoom/Teams、PowerPoint 录制、Clipchamp、OBS，拼接、压缩、最终检查清单）见 `video/RECORDING.md`。

- 最省事的方式：Zoom / Teams 会议录制，打开摄像头并共享幻灯片，每人讲自己那段。或者用 OBS（屏幕 + 摄像头画中画），或 PowerPoint 的"录制"功能（可以带摄像头）。
- 分段录制，再拼接；每段先按计时器排练一遍。
- 幻灯片字号要大，每页一个信息点；图直接用 `reports/figures/` 里的。
- 用耳机麦克风，在安静的房间录；正式录之前先试录 10 秒，检查音量。
- 录屏前关掉无关的浏览器标签页和通知。**不要让 Kaggle Settings / API token 页面出现在画面里**。
- 1080p 或 720p 都可以；屏幕内容以文字为主时，720p 就够清楚。

## 压缩 / 降分辨率（ffmpeg）

这台机器在 setup 时（2026-10-22）**没有安装 ffmpeg**。由成员自行决定是否安装，例如 `winget install --id Gyan.FFmpeg -e`，或从 ffmpeg.org 下载 Windows build。不想装的话，Windows 11 自带的 Clipchamp 或 HandBrake 也可以导出 720p。

降到 720p、30 fps，H.264 + AAC：
```powershell
ffmpeg -i D:\video_raw\final_edit.mp4 -vf "scale=-2:720,fps=30" -c:v libx264 -preset slow -crf 26 -c:a aac -b:a 128k -movflags +faststart video\TODO_TEAM_NAME.mp4
```
- 文件仍然太大：把 `-crf` 调到 28–30（数字越大文件越小）。300 MB / 300 s ≈ 8 Mbit/s 是上限，720p 屏幕录制一般只有 1–3 Mbit/s（约 40–110 MB）。
- 拼接多段编码参数相同的录像：先写 `D:\video_raw\list.txt`，每行一条 `file 'part1.mp4'`，然后运行
  ```powershell
  ffmpeg -f concat -safe 0 -i D:\video_raw\list.txt -c copy D:\video_raw\final_edit.mp4
  ```
  （编码参数不同时去掉 `-c copy`，让 ffmpeg 重新编码。）

检查时长和大小：
```powershell
ffprobe -v error -show_entries format=duration,size -of default=noprint_wrappers=1 video\TODO_TEAM_NAME.mp4
(Get-Item video\TODO_TEAM_NAME.mp4).Length
```
要求：duration ≤ 300.0（秒），size ≤ 300000000（字节）。最后再完整播放一遍，确认每个成员都露脸、声音清楚。

## 时间点（见 `docs/TIMELINE.md`）

- 11-22 前：讲稿定稿（AI 初稿 2026-11-01 已写好，每人改写自己的部分）
- 11-24 – 11-26：录制、剪辑、压缩（与 `docs/TIMELINE.md` 一致）
- 11-27：`.venv\Scripts\python.exe scripts\package_deliverables.py` 预检能识别到视频
