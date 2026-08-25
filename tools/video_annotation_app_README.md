# QV-M2 视频标注工具

这是一个纯 Python 标准库实现的轻量 Web 标注器，用于生成 QVHighlights/QV-M2 风格的 JSONL：

```json
{"qid": 3158, "query": "...", "duration": 150, "vid": "..."}
```

如果标注了高亮片段，导出的 `train`、`val` 会额外包含：

```json
"relevant_windows": [[0, 16]],
"relevant_clip_ids": [0, 1, 2, 3, 4, 5, 6, 7],
"saliency_scores": [[4, 4, 4], [4, 4, 4]]
```

`test` 默认会去掉 `relevant_windows`、`relevant_clip_ids`、`saliency_scores`，保持和 `data/highlight_test_release.jsonl` 一样的发布格式。

## 启动

```powershell
python tools\video_annotation_app.py --host 127.0.0.1 --port 7860
```

浏览器打开：

```text
http://127.0.0.1:7860
```

数据库默认保存在：

```text
annotation_workspace/annotations.sqlite3
```

## 导入已有 JSONL

可以在网页里直接点击文件选择框选择 `.jsonl` 文件导入；也可以填 `data/highlight_test_release.jsonl` 后点击导入，或者启动时导入：

```powershell
python tools\video_annotation_app.py --import_jsonl test:data\highlight_test_release.jsonl
```

如果本地有视频目录，并且视频文件名能匹配 `vid.mp4` 或 `{youtube_id}.mp4`，可以加：

```powershell
python tools\video_annotation_app.py --video_root D:\videos --import_jsonl test:data\highlight_test_release.jsonl
```

## 多人标注与审核

多人在同一个局域网使用时，把服务监听到 `0.0.0.0`：

```powershell
python tools\video_annotation_app.py --host 0.0.0.0 --port 7860
```

页面顶部填写用户名并选择角色，然后点击“登录/创建账号”。系统内置两个初始账号：

```text
annotator1 / 标注员
reviewer1 / 审核员
```

标注员权限：

- 领取任务
- 编辑自己的标注
- 保存草稿
- 提交审核

审核员权限：

- 导入 JSONL 和视频
- 修改任务基础信息
- 划分训练/验证/测试集合
- 导出 JSONL
- 查看所有标注员提交的结果
- 点击“载入修正”修改某个标注员的窗口和备注
- 通过或驳回标注

同一个 `qid` 可以有多个标注员各自提交一份标注；导出时默认只使用 `approved` 标注。

## 选择视频文件

页面提供两种点击式视频选择方式：

- 在左侧“导入”区域选择多个视频文件：工具会上传到 `annotation_workspace/uploads`，并按文件名自动匹配已有任务的 `vid` 或 `{youtube_id}_...`。
- 在某个任务详情里点击“选择视频”：该视频会直接绑定到当前 `qid`。

浏览器不能直接读取任意本地绝对路径，所以点击选择的视频会复制到 `annotation_workspace/uploads` 中，之后标注页面会从这个目录播放。

## 发布任务包

推荐的新流程是不依赖现成 JSONL：

1. 审核员登录。
2. 审核员选择视频文件或视频目录上传。
3. 在“每个任务包含的视频数”中填写数量，例如 `5`。
4. 点击“按视频数发布任务”。
5. 系统会把尚未发布的视频按数量打包成任务包。
6. 标注员登录后在“任务包”列表中看到每个任务是否已被接取。
7. 标注员只能接取未被接取的任务包，接取后逐个视频标注并提交审核。
8. 审核员打开任务包，载入标注员提交内容后可以修正、通过或驳回。

JSONL 文件不是前置条件，而是最后审核完成后导出的结果。

## 划分训练、验证、测试集合

在页面里设置 `train_ratio`、`val_ratio`、`seed`，点击“按视频比例划分”。划分按视频基础 ID 聚合，避免同一原视频的不同 query 被拆到不同集合。

没有完成标注时也可以先划分集合；完成并审核后再导出，导出器会只写入通过审核的样本。

## 导出

页面里设置输出目录，例如：

```text
annotation_workspace/exports
```

点击“导出 release JSONL”，会生成：

```text
highlight_train_release.jsonl
highlight_val_release.jsonl
highlight_test_release.jsonl
```

导出来源可以选择“全部任务”或“仅审核通过”。如果还没有做标注，选择“全部任务”可以先得到只包含 `qid/query/duration/vid` 的 train/val/test 文件；标注和审核完成后，选择“仅审核通过”即可导出带高亮窗口的训练/验证集。

如果要直接作为项目数据使用，可以把输出目录设为 `data/my_annotations`，再在训练脚本中指向对应 JSONL。
# 当前任务包工作流说明

最新版本以“任务包”为主线：

1. 审核员选择视频文件或视频目录，点击“确认视频”后，才能发布任务。
2. 审核员设置“每个任务包的视频数”和任务名前缀，点击“按视频数发布任务”。
3. 标注员只能接取未被接取的任务包。接取后，右侧会显示该任务包内全部视频。
4. 任务包右侧会统计：视频总数、已标注、剩余、已提交审核、已通过。
5. 标注员可以保存单个视频草稿，也可以提交当前视频审核。
6. 标注员也可以在任务包面板中勾选部分视频提交审核，或一键提交该任务包内全部已标注视频。
7. 审核员打开任务包后，可以看到该包内所有视频及待审核数量，点击视频后在“审核记录”里查看、修正、通过或驳回。

账号与权限：

- 打开页面后先进入登录/创建账号界面，登录成功后才显示主页面。
- 创建账号时需要填写账号、密码，并选择角色：标注员或审核员。
- 账号不能重复；已有账号必须用密码登录。
- 顶部会显示当前账号和角色，可以点击“退出登录”切换账号。
- 标注员只能看到未被接取的任务包和自己接取的任务包，不能查看其他标注员已经接取的任务包。
- 审核员可以看到所有发布、接取、提交审核和审核通过的任务包。
- 默认账号用于本地快速测试：`annotator1 / annotator1`，`reviewer1 / reviewer1`。

显著性分数逻辑：

- 人工先标连续时间段 `relevant_windows`。
- 每个时间段有一个“整段默认分数”。
- 如需细化，可展开“clip 评分”，对每个 2 秒 clip 单独打分。
- 导出 QVHighlights/QV-M2 格式时，系统会生成 `relevant_clip_ids` 和每个 clip 对应的 `saliency_scores`。
- 原始 QVHighlights 的 `[a,b,c]` 表示三个标注者对同一个 clip 的评分；本工具新标注默认按最终审核分数导出为 `[score, score, score]`，用于兼容现有训练格式。
# 当前标注逻辑

- 一行 JSONL 是一个 `qid` 标注项，也就是一个 `query + vid` 的检索样本，不等于一个完整视频任务。
- 同一个 `vid` 可以出现多次，并对应不同的 `query`。
- 每个 `query` 都有自己独立的 `relevant_windows`，所以标注时需要先选择视频，再在该视频下选择具体 query 单独标注片段。
- 任务包按视频发布和接取；进入任务后，右侧会按 `vid` 汇总视频，编辑区会列出该视频下所有 query。
- 标注员只能保存和提交自己的标注；审核员可以查看已提交标注、载入修正、通过或驳回。

