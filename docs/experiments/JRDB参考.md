# JRDB 数据与 T1 实现参考

本文维护既有JRDB参考资源的字段、对齐、T1输入协议与工具操作。已有冻结JRDB开发窗口用于VLM接口与延迟对照；原始JRDB数据不作为检测和实机资源迁移的前提，仅在重新生成输入或审核源标签时按需准备。来源划分及人工评测见[数据与评测](数据与评测.md)，研究任务语义见[系统方案](../系统方案.md)，远程服务操作见[VLM部署与调用](../decision/VLM部署与调用.md)。

JRDB T1使用P0标注几何，不能当检测器的P1结果；框架或文档整理不改变原始数据、冻结输入、人工标签及用户结果。

## 1. 参考数据范围

默认入口为 `data/datasets/jrdb`，实际链接目标由本机记录维护。以下统计对应既有**JRDB 2022训练部分**参考资源，不是全量train/test或独立Social新录像；新机器仅在使用相应工具时准备。配置与重定位见[开发指南](../开发指南.md#数据入口与迁移)。

已有 `train_images.zip`、`train_pointclouds.zip`、`train_calibration.zip`、`labels.zip`、`train_reasoning.zip`；此前已通过CRC和解压清单一致性核验。本次整理不修改这些包或解压源数据。

| 目录 | 文件数 | 内容 |
|---|---:|---|
| `images/` | 167,358 | 5单相机＋1拼接视角JPEG |
| `pointclouds/` | 55,786 | 上下两路Velodyne PCD |
| `calibration/` | 8 | 标定YAML、单相机到拼接映射NPY |
| `labels/` | 2,548 | 人框、动作、社交、头框、姿态、全景及伪3D标签 |
| `labels_2d_stitched_reasoning/` | 27 | 拼接视角Reasoning标注JSON |

共27序列、27,893时间帧；每时刻6张图和2个PCD，不把文件数当独立时间帧。单相机752×480，拼接3760×480，原RGB名义15fps。当前部分按帧数约31分钟，单序列约29–116秒；这些是帧号换算，不是精确时间戳。没有另下载timestamps/rosbag或测试部分。

主要路径：

```text
images/image_0|image_2|image_4|image_6|image_8/<seq>/<frame>.jpg
images/image_stitched/<seq>/<frame>.jpg
pointclouds/lower_velodyne|upper_velodyne/<seq>/<frame>.pcd
labels/labels_2d_activity_social/<seq>_image0.json
labels/labels_2d_activity_social_stitched/<seq>.json
labels/labels_3d/<seq>.json
labels_2d_stitched_reasoning/<seq>.json
```

序列是采集、轨迹和划分的基本单位，六位帧号只在各自序列中有意义。JRDB并非全是静止咖啡店：当前开发采用clark交叉口、gates通道和memorial庭院，可按第6节生成前视＋BEV审核视频；它们仍是校园机器人视角，不等于目标轮椅分布。

### 1.1 序列与时间

既有27训练序列共27,893帧；单序列430～1,735帧，均值约1,033帧。名义15fps换算总计约31分钟，播放时长N/15与首末采样时间差(N-1)/15不同，均不是精确时间戳。完整序列计数由源审计维护，不在手册重复逐序列表。

序列名中的2019是采集日期，不代表发布版本；“移动场景”依据背景变化选择，不据此得到可靠自车位姿。

### 1.2 版本、扩展与目录职责

基础JRDB提供传感器和人物检测/跟踪标注，Act/Social/Pose/PanoTrack/Reasoning增加任务或标签。多数扩展复用录像，但下载包的组织、传感器标定、覆盖帧和标注版本仍可能变化；不能笼统认为“所有扩展都是只有一份JSON且完全一致”。当前2022训练下载包已包含多种同源标签，不需要再下载一套同序列RGB来使用它们。

`labels/`中的实际目录如下；目录数与下面文件数说明数据形态，不代表每个任务有同样多有效标注。

| 目录 | 文件数 | 文件粒度 |
|---|---:|---|
| labels_2d_activity_social | 135 | 27序列×5单相机 |
| labels_2d_activity_social_stitched | 27 | 每序列一个 |
| labels_3d | 27 | 每序列一个 |
| labels_2d_head / labels_2d_head_stitched | 135 / 27 | 单相机/拼接COCO JSON |
| labels_2d_pose_coco / labels_2d_pose_stitched_coco | 135 / 27 | 单相机/拼接COCO JSON |
| labels_2d_panoptic_CW / labels_2d_panoptic_stitched_CW | 135 / 27 | 单相机/拼接稀疏COCO JSON |
| labels_3d_pseudo_panoptic_CW | 1,873 | 每序列、每标注帧一个.bin |

单相机人物/姿态/头框文件后缀如 `_image0.json`，全景分割如 `_camera_0.json`；stitched不带单相机后缀。同名帧号、COCO image_id和人物ID跨序列不唯一。所有索引至少带序列、视角、帧和所属任务。

## 2. T1相关的原始字段与标签边界

### 2.1 2D人物及Social

一个JSON覆盖一个序列/视角，顶层为 `labels`字典；使用外层帧键关联图像：

```json
{"labels":{"000000.jpg":[{"label_id":"pedestrian:6","box":[2962,63,299,397],"social_group":{"cluster_ID":7,"cluster_stat":1}}]}}
```

这是拼接视角格式节选，不是完整记录。单相机对应的框坐标应在自身图像范围。

| 字段 | 格式与用途 |
|---|---|
| `label_id` | 序列内人物身份，如 `pedestrian:6`；输入另映射中性整数PID |
| `box` | `[x,y,width,height]`，像素；不直接当xyxy |
| `attributes` | 遮挡、截断、interpolated、no_eval等；部分布尔与字符串混用 |
| `action_label` | 动作名→数值，可多标签；不是归一化概率 |
| `social_group` | `cluster_ID`和`cluster_stat`；成员候选/原生诊断，不是完整REL真值 |
| `social_activity` | 群体活动名→数值，可能有浮点值；帧内同组一致性需检查，不能盲目广播 |
| `H-interaction` | 列表，每项含另一人的 `pair`、`box_pair`及 `inter_labels` |
| `group_info` | 场所、目的、上下文等嵌套字段；不是排队顺序或拍照角色 |
| `demographics_info` | 个体属性，当前任务不需要，也不作为VLM输入 |

HHI样例：`{"pair":"pedestrian:7","box_pair":[3359,106,236,366],"inter_labels":{"conversation":1}}`。它描述两两交互，可能跨群组。HHI不是同组等价关系，不能通过对称化、传递闭包或连通分量直接生成群组真值。

当前T1目标是可观测成员＋w/t/q/p/o/u活动＋结构k；JRDB Social的语义字段用于检索候选和审核辅助，不直接机械映射完整目标答案。

完整形态节选如下，显示内部file_id与外层键是两个独立字段；省略字段不表示原数据没有它们：

```json
{
  "labels": {
    "000000.jpg": [
      {
        "label_id": "pedestrian:6",
        "file_id": "000002.jpg",
        "box": [2962, 63, 299, 397],
        "action_label": {"sitting": 1, "talking to someone": 1},
        "social_group": {"cluster_ID": 7, "cluster_stat": 1},
        "H-interaction": [
          {"pair": "pedestrian:7", "box_pair": [3359,106,236,366],
           "inter_labels": {"conversation": 1}}
        ]
      }
    ]
  }
}
```

动作可多标签；group_info可含venue、aim、inter、BPC、SSC、location_pre等嵌套词典，例如aim里的 `socializing&eating/ordering_food`。这些上下文不是队列顺序或摄影角色。social_activity虽然描述群体活动，却存在各成员记录内；帧内同组的缺失、数值或类别不一致应保留，不能未经审核把某个人的值广播给全组。

统计群组人数应区分“逐帧cluster实例”“序列内群组轨迹”和“独立社会事件”；逐帧累计会让长时间静止组权重很大。只看前视时群组会被视野截断，单个可见成员不证明该人本来独行。正式T1使用可观测补标，而不是把原生cluster人数分布当目标类别分布。

### 2.2 3D人物框与标定

`labels_3d/<seq>.json`采用 `labels["000000.pcd"]`，通过同帧 `label_id`与2D配对：

```json
{"label_id":"pedestrian:14","box":{"cx":-4.20404,"cy":0.95873,"cz":0.11048,"h":2.08097,"l":1.04,"w":0.55,"rot_z":-1.63424}}
```

中心和尺寸是米制，rot_z为弧度朝向，不是视线；部分3D记录也带动作/群组字段，渲染输入只白名单读取几何。当前工具读取 `calibration/lidars.yaml`，对RGB去畸变，点云按相应矩阵转换；不能原样套用为实机标定。缺少同人物3D位置时保留null，不虚构BEV点。

原始PCD含x/y/z/intensity/ring并采用 `binary_compressed`，不是裸float32 `.bin`。当前T1不需要把点云传给VLM，也不将点云等同于自由空间地图。

3D记录还可含 `observation_angle` 和attributes中的distance、num_points、interpolated、no_eval。记录单位和坐标约定来自数据与标定；工具将上下雷达点云按矩阵变换至ego绘制，3D框按JSON数值绘制，不能仅凭字段名就断言投影绝对无误。

| 文件 | 内容与读取约定 |
|---|---|
| cameras.yaml | stitching和sensor_*的K、D、R、T；相机顺序重要 |
| lidars.yaml | upper2ego、lower2upper及相机/雷达/ego相关矩阵；确认方向再使用 |
| defaults.yaml | 标定默认参数 |
| indi2stitch_mappings/*.npy | 五路单相机到拼接视图映射，不是相机内参 |
| lower/upper_velodyne/*.pcd | PCD 0.7；SIZE为4/4/4/4/2，TYPE为F/F/F/F/U，压缩布局不是连续XYZ数组 |

原始标签中的bool与字符串bool需规范化后使用；不能把 `"false"` 用Python bool直接当真。no_eval和interpolated需留在审核元数据中；是否排除应由任务评测规约决定，不由读取器偷偷丢弃。

## 3. 其他已下载标签：保留审核能力，暂不加入主输入

| 标签目录 | 数量/关联 | 格式与注意事项 |
|---|---|---|
| `labels_2d_head[_stitched]` | 135＋27 COCO JSON | annotations.image_id→本文件images.id，bbox为xywh，track_id按序列管理 |
| `labels_2d_pose[_stitched]_coco` | 135＋27 COCO JSON | 17组 `[x,y,v]`，长度51；关节顺序读categories，不假定有bbox/area |
| `labels_2d_panoptic[_stitched]_CW` | 135＋27 COCO JSON | 类别、COCO RLE掩码、tracking_id；约1fps稀疏标注，不假定有bbox |
| `labels_3d_pseudo_panoptic_CW` | 1,873 `.bin` | 2D投影产生的伪3D逐点标签；已核验样例每点4字节，位域/拼点顺序不能只凭尺寸猜测 |
| `labels_2d_stitched_reasoning` | 27 JSON | 逐帧HHI/HOI等；不是现成QA样本，HOI数字字符串需官方实体映射 |

同一COCO文件内关联image_id，跨文件不当全局ID。姿态可支持动作观察，Reasoning可支持辅助诊断，但目前不自动导出T1答案。原生与伪3D标签质量分开，不为了“利用所有标签”而扩大输入。

### 3.1 COCO关联：头框与姿态

典型顶层为images、annotations、categories、info、licenses。关联顺序是本文件annotations.image_id→images.id→file_name对应的序列/帧；不能把COCO image_id直接当帧号，也不能未经核验把track_id与pedestrian:N当同一个编号。

头框记录含id、image_id、track_id、category_id、bbox、area、is_crowd。bbox仍为xywh。本地bytes-cafe样例categories只声明id=2/head，但annotations同时有category_id=1和2，且存在近似框；160/162个头框JSON有类别表外id=1。不能将全部记录相加直接当独立头人数，需按实际类别和去重规则审核。

姿态记录含image_id、track_id、category_id、num_keypoints、keypoints、is_crowd；本地抽样没有bbox/area，读取器允许缺省。keypoints为17组x/y/v，v=0不可见、1遮挡但可推断、2可见；顺序和骨架读该文件categories.keypoints/skeleton，不套其他关节排序。

### 3.2 PanoTrack的RLE与伪3D

```json
{
  "image_id": 1,
  "category_id": 68,
  "segmentation": {"size": [480,3760], "counts": "<压缩COCO RLE>"},
  "area": 24486,
  "tracking_id": 58,
  "attributes": {"occluded": false, "is_crowded": false, "tracking_id": "58"}
}
```

这是格式节选，不是可解码完整样本。RLE按size解码成二维掩码，不是多边形；抽样条目没有bbox，必要时从mask推导。类别读取当前JSON的categories；网页介绍71类而抽样数组有72项，不能硬编码71。顶层tracking_id与attributes.tracking_id可能分别为整数/字符串，应规范化并检查冲突。

伪3D.bin是投影生成的逐点标签，不是PCD。bytes-cafe首帧142,856字节，对应上下雷达17,297＋18,417=35,714点、每点4字节。全量长度一致只能证明布局长度；32位高低位、上下点序、未知码与跨帧实例约定仍需官方转换代码及投影复核。

当前pseudo3d渲染并排展示两种候选拼点顺序的原始编码，只帮助检查，不声称已经正确恢复语义类别，也不用于可靠的3D训练/评测标签。

### 3.3 Reasoning格式与实体关联

Reasoning仍为 `labels[帧.jpg] → 人物数组`。HHI元素形态与H-interaction类似；HOI可出现 `{"pair":["134","62"],"inter_labels":{"sitting":1}}`。这些数字字符串不自动等于pedestrian:N，需按官方场景图/生成代码确定物体实体映射。

下载的是逐帧标注，不是可直接训练/评测的完整VQA题。要派生QA须明确观察跨度、问答模板和真值来源；当前代码只显示HHI与HOI原始信息，不给未知物体ID伪造位置。

## 4. 已核验的对齐风险

以下保留此前全量源审计的结论，历史`all.json`不作为当前工作区已保留的产物入口；当前资源位置以本机记录为准：

1. 2D/Social六视角的1,932,611记录和Reasoning的843,133记录中，内部 `file_id`均比外层帧键大2；3D的861,033记录一致。当前按外层键叠图，不静默平移2帧。
2. 头框/姿态stitched JSON的27,877个image条目写width=752，但真实拼接图宽3760。裁剪、归一化读取真实尺寸。
3. 头框/姿态每视角有16个序列缺末帧；全景分割是稀疏标签。缺标不表示无人或负例，视频不补画虚假标签。
4. stitched人物与Reasoning逐帧ID集合一致，但与3D在14,255帧存在差异；按同帧ID配对，不按数组顺序，不丢掉缺3D的2D人物。
5. COCO file_name存在旧路径/重复序列名；用当前文件内image_id及唯一帧映射解析。头框类别映射、姿态缺bbox、RLE类别数均以实际JSON为准。
6. ID不是跨序列全局身份。划分按完整序列/地点；姿态插值和全序列处理不能被默认称为因果在线感知。

上述数据统计来自此前本机审计，本次没有重下载数据。需要复核时运行数据审计命令，不覆盖既有报告。

### 4.1 审计字段与数值边界

| 报告字段 | 解释 |
|---|---|
| frame_key_differences | 逐帧标签外层键与源帧集合的对称差，不是某类标签准确率 |
| image_view_frame_differences / lidar_frame_differences | 多视角图像、两路雷达帧名是否一致 |
| file_id_alignment | 内部file_id减外层键的偏移；+2是元数据现象，不自动修改输入帧 |
| person_id_alignment | 同帧人物ID集合差异；只取ID交集配对，不按数组下标 |
| coco | 悬空image_id、类别缺项、尺寸元数据与路径异常 |
| pseudo3d | 稀疏帧名及每点4字节长度检查，不验证编码语义 |

同帧2D-only人物累计11,994实例，3D-only累计29,894实例，属于逐帧实例累计而非独立人数。每视角Panoptic有1,873标注帧，325个COCO file_name是非标准导出路径；按本文件image_id与唯一basename解析，不直接拼旧目录。

当前原始传感器与逐帧标签外层帧集合完全一致，不代表每个人在每项任务都有标签。姿态/头框缺末帧与Panoptic稀疏标注都必须记录为“该任务无标签”，不是空场景。

审计检查索引/长度，不证明全部像素、关键点、类别或3D位置准确。此前只目视抽查部分首帧、中间无稀疏标签帧和末帧；3D标定/框投影还应额外复核。不得把本报告称为全数据人工验收。

## 5. 当前输入、Prompt与输出

当前唯一主协议是 `object`：三帧真实前视RGB＋共范围BEV，默认1280×544、无人物图库、无真值颜色/连线。必要小标签只包含PID、相对时间和坐标说明，不显示原始标注明细。

动态字段为 `v,dt,ts,e,o,d`，无动作、群组、HHI、原始人物身份或人工答案。近距离表是几何提示，不是同组真值；无可靠自车位姿时不提供假速度。固定前缀见 [Prompt](../../decision/vlm/configs/prompts/jrdb_t1_rel.txt)，内容沿用此前修正格式后验证的对象版，不在本次整理中凭空修改决策语义。

```json
{"g":[{"m":[0,2],"r":"w","k":0}],"u":[5]}
```

任务语义、unknown与角色顺序统一按[整体方案的输出规约](../系统方案.md#5-输出与校验)维护。动态Schema约束当前PID，客户端再查整数、成员互斥、角色、完整流和非思考状态；下文聚焦真实字段和复现链，不另立一套标签定义。

此前24开发窗口的72图已从旧测试包无损迁到 [固定输入](../../data/benchmarks/jrdb_t1_dev/manifest.json)，没有真值或凭证。旧样本保持原图字节、几何、Prompt和Schema，便于结果追溯；它们不含本地重建可视化所需的完整源审计manifest。要新做补标/预测对照，应使用 `jrdb_t1 prepare`生成带源审计的新版bundle，不能为旧样本伪造source指纹。

### 5.1 几何表的精确定义

| 字段 | 精确定义 |
|---|---|
| `v` | 末帧有效前视人物的整数PID；只有这些PID允许出现在REL响应中 |
| `dt` | 三张真实图相对末帧的时间，秒，严格递增且末项为0 |
| `ts` | 时间来源；当前为名义帧号换算，不是精确传感器时间戳 |
| `e` | 三帧共用BEV范围 `[xmin,xmax,ymin,ymax]`，米 |
| `o` | 每人 `[pid,xy三项,b三项,h三项]`；下面解释坐标与框 |
| `d` | 近距离候选对 `[pid_a,pid_b,d0,d1,d2]`；下面解释筛选 |

编号在窗口内固定，不按真值组排序/着色。三帧BEV使用共范围，可超过旧版12m窗口以覆盖有效前视人物；这是局部三帧范围，不是全序列BEV ALL。

o每项为 `[pid, xy三项, b三项, h三项]`。xy是各时刻ego坐标米制x/y；b是对应原前视图的xyxy框归一化至0–1000，不是输出拼图的像素坐标；h为该RGB框的像素高度。去畸变后的RGB与框在同一图像坐标中使用。

d每项为 `[pid_a,pid_b,d0,d1,d2]`，米制距离保留到两位小数；任一时刻距离≤3.5m才列出。没有列出的配对不自动代表“不同组”，这个筛选只节约几何文本。某时刻缺位置时相应距离为null。

时间按目标末帧往前选择约1s、0.5s、0s的三张真实帧；15fps的半秒不落整数帧，实际dt可能约-0.533或-0.467s。不足真实历史或三时刻不递增则报错，不复制首帧。P0可能使用插值/全序列关联，不能因选帧因果就称完整P1。

### 5.2 输入、标签与指纹隔离

新bundle包含transport文件清单与每case本地审计。人工模板最初为 `{"g":[],"u":当前v}` 的全未知draft；这只是待审核占位，不是自动生成的准确答案。reviewed入口校验指纹、审核字段和REL约束，审核图可以使用颜色/箭头，但不能回流到HTTP输入。

references/<case>.native.json只保存前视native成员及来源hash，标记NOT_reviewed_REL_gold；它没有完整关系、结构或可观测unknown。compare画预测与native并排，不导出gold分数，也不自动画人工答案对比。

存在两条指纹链：case本地manifest的input_fingerprint绑定人工标签与源重建；HTTP加载器重新计算的input_fingerprint绑定实际传输metadata/Prompt/图片，用于日志与compare。两者序列化定义不同，不要求字符串相等，不能拿其中一项冒充另一项。修改图片、PID表、Prompt或Schema后需重新生成对应输入和标签绑定。

运输加载器要求每个被引用的Prompt、metadata和三张图片均在文件SHA-256清单中；少写一项也拒绝加载，不只校验清单里碰巧存在的文件。三图须为三个不同资产路径，case_id不允许目录穿越。哈希用于发现版本混用或意外修改，不是签名认证，不能对恶意重写整个manifest提供真实性保证。

重建审核/比较图前，`load_window`还校验保存的`implementation_sha256`与当前`jrdb_t1_inputs.py`、`jrdb_t1_common.py`一致。缺项或代码版本不一致直接报错，不用新渲染器悄悄替换旧画面。需要审阅旧输入时使用对应源码版本；使用当前代码则在新目录重新生成并人工复核标签。该检查不锁定Pillow、OpenCV或字体版本，严格复现还需记录环境。已有24运输样本仍可原样测速，不依赖源重建检查。

使用默认别名的新case还保存`dataset_root_ref=jrdb`，重建时从当前仓库配置找到JRDB链接，不再依赖生成机器的物理数据目录；root字段仍保留当时路径作审计。所有源文件hash与几何签名继续检查，更换链接目标为不同数据会失败。显式外部root生成的输入和缺少逻辑引用的旧case仍按旧绝对路径读取，不自动猜测或改写历史manifest。

portable打包仅复制system_prompt、每case model_input与三张图片及最小客户端，不复制源人物ID、native、模板和已有私有配置。新包另携带经哈希校验的公共测速默认值快照；其配置优先级见[客户端配置](../decision/VLM部署与调用.md#仓库客户端)。故可以不带JRDB源进行远程测速，但不能只凭便携包重建源审核图。

## 6. 当前代码与人工审核

以下入口维护JRDB解析与审核，命令见本文[工具使用](#工具使用)；远程请求和打包由VLM手册维护：

| 入口 | 用途 |
|---|---|
| [jrdb_visualize.py](../../decision/vlm/jrdb_visualize.py) | 数据对齐审计、9类标签PNG/MP4、3D透视渲染 |
| [visualize_jrdb_group_relations.py](../../decision/vlm/visualize_jrdb_group_relations.py) | 前视＋局部BEV群组ID、同组关系与HHI审核；默认无BEV ALL |
| [jrdb_t1.py](../../decision/vlm/jrdb_t1.py) | 当前三图prepare、指纹绑定的reviewed、预测/native并排compare |
| [benchmark_vlm.py](../../decision/vlm/benchmark_vlm.py) | 手动SGLang端点；dry-run、GET检查、串行/按节奏HTTP测试 |
| [package_vlm_test.py](../../decision/vlm/package_vlm_test.py) | 白名单复制非敏感输入并生成便携ZIP |

`prepare`输出的annotation_template是全未知draft，占位符不是gold；`reviewed`要求输入指纹、人工状态、人员、时间和provenance，否则只允许显式draft预览。模型输入与请求不读取模板或references。`compare`只是预测与native成员图，不导出完整T1金标准分数。

媒体工具可生成咖啡店9类标签及3D透视视频、clark/gates/memorial三场景前视＋BEV视频和抽样PNG。
历史示例曾输出到`outputs/jrdb/videos/`，不据此推断当前机器仍保留媒体；实际资源位置见本机记录。
原图15fps与稀疏标签1fps区分；稀疏类缺帧时不补标，不在画面旁滚动打印标注文本。

兼容的旧P0标签查看器和几何核心保留，用于审核已有输入；默认流程不再暴露legacy/spatial多种实验profile。

### 6.1 九类图层如何解读

| 图层 | 显示内容与边界 |
|---|---|
| people | 人物框与源身份 |
| actions | 个体动作、群体活动；数值不解释为概率 |
| social | 原生群组颜色、交互及上下文；不是模型输入 |
| head | 头框及原类别；注意类别表缺项/近重框 |
| pose | 17点骨架、关键点可见性；插值不能当原始人工帧 |
| panoptic | COCO RLE类别/实例mask，只有标注关键帧 |
| reasoning | HHI连线、HOI原始ID，不伪造HOI对象坐标 |
| view3d | 上下点云＋立体人框的3D透视，非BEV |
| pseudo3d | 透视点云上两种候选点序的原始编码，语义未证实 |

默认all包含view3d/pseudo3d；旧bev3d/pseudo3d_bev只能显式选择。透视虚拟相机位于约(-16,-18,13)m，x前/y左/z上；它是检查视角，不是重新标定结论。拼接缝关系线走短路径，避免跨整幅图的假长连线。

PNG/MP4不再显示不断刷新的原始字段侧栏，只保留必要目标短标签、原始帧号及固定图例。详细字段在JSON；不能为画面清爽而删除用于审计的底层记录。

### 6.2 原始帧率与稀疏标签视频

bytes-cafe原始000000–001725共1,726帧。people/actions/social/reasoning/view3d各1,726帧；head/pose各1,725帧，缺原末帧；panoptic/pseudo3d各116真实关键帧、独立视频约1fps。

九格overview_15fps.mp4按全部原始帧号同步，以15fps播放；稀疏类无标签时显示NO LABEL FOR THIS SOURCE FRAME，不沿用上一帧mask。15fps总览不等于分割具有15Hz标注密度。

video的stride=1保留原始时序，stride=15为约1fps抽帧；稀疏类只编码有标签帧，并按源帧间隔估播放频率。若间隔不规则，恒定fps视频不能代替精确对时，应读manifest.source_frame_ids。max-frames是每层上限，做同步小预览时优先限制相同源时间区间。

各层MP4、总览和首帧PNG由同次生成目录中的manifest索引，历史咖啡店目录不再作为当前保留结果入口。
需要新媒体时按[工具使用](#工具使用)生成到新目录；文档调整不要求重新渲染或恢复历史产物。

### 6.3 前视＋局部BEV群组/HHI视频

该工具显示原生标签供人工检查，默认front-near两面板、1920×1024，不显示BEV ALL。它比当前模型输入范围宽：前视外有3D的人也可出现在局部BEV；因此审核视频不能直接作为T1模型输入。

| 画面标记 | 含义 |
|---|---|
| P6 / G7 | 源pedestrian:6 / 原cluster7，不是模型短PID |
| 稳定群组颜色与细线 | 序列内组色与全部同组成员对，不是HHI语义边 |
| 带短类型的粗线 | HHI类型；同一对可多标签，不执行T1活动优先级 |
| 红色虚线 | 跨组HHI，不据此合并群组 |
| BEV实心/空心点 | 当前前视有效框内/前视外；不表示置信度或遮挡级别 |
| G7! / G? | 来源组信息冲突 / 缺组信息；缺失不当独行负例 |
| NO3D / OutsideBEV | 缺3D位置 / 有位置但超局部画幅，两种情况分开 |

人/组信息优先stitched、再使用可用前视/3D来源，并记录冲突；前视框使用前视标签，不把全景box_pair像素坐标直接画到前视。HHI双向重复可合并为显示边，原方向/标签和难度仍保存在JSON。

BEV x向上、y向左、单位米，R为机器人参考；FOV线和点云只是观测，不证明自由空间。默认local-span=24对应x[-10,14]/y[-12,12]，现有移动场景用32对应x[-14,18]/y[-16,16]。超过范围的人和边保留在记录，但不画入局部面板。

逐帧JSON记录drawn_bev_ids、outside_rendered_bev_ids及hhi_outside_rendered_bev，manifest记录HHI短名映射和统计。若显式启用ALL，边界可能使用全序列几何，是离线审核而非因果输入；当前主线不使用。

此前三场景参考媒体共2,820帧，均15fps、H.264、1920×1024；下表记录历史生成目录与源帧对应，不是当前保留清单：

| 历史生成目录 | 源序列 | 帧数 |
|---|---|---:|
| mobile_scenes_front_bev_15fps/clark_center_intersection | clark-center-intersection-2019-02-28_0 | 1,007 |
| mobile_scenes_front_bev_15fps/gates_to_clark | gates-to-clark-2019-02-28_1 | 724 |
| mobile_scenes_front_bev_15fps/memorial_court | memorial-court-2019-03-16_0 | 1,089 |

这些场景比静止咖啡店更适合当前开发，但只有3个根序列、且已反复用于调试，不能当独立盲测。

## 7. 历史远程结果参考

历史AutoDL应用的原始结果保留为[远程results1.zip](../../outputs/decision/sglang_jrdb_comparison/historical_autodl/results1.zip)和[服务器内results2.zip](../../outputs/decision/sglang_jrdb_comparison/historical_autodl/results2.zip)。应用来源由用户确认为[AutoDL应用279](https://www.autodl.art/app/market/279?search=qwen3.8)，但其权重与完整运行配置未独立验证；公开页面也不证明历史实例的实际启动参数。

历史串行与paced结果、当前双4090的MTP／DSPARK对照、请求绑定及计时限制统一由[耗时汇总](../../outputs/decision/sglang_jrdb_comparison/report.md)维护，本参考不重复成绩表。
历史包按2秒离线重算的完成数不等于0.5Hz实测，旧1Hz到达、不同协议和硬件的结果也不能直接解释为当前配置的收益。
历史与当前请求摘要不匹配时只作参考，不修改摘要或输入来强行配对；旧服务、本地／远程差值不能进一步当作纯RTT或GPU耗时。

当前MTP和默认DSPARK已有冻结窗口的串行与短时0.5Hz调度测试记录，均与连续社会状态回放分开。
交错三个场景测延迟不验证同一序列的身份保持、群组连续性或结果年龄；人工REL金标准、隔离盲测、T2/T3、P1和实机闭环仍待完成。
合法响应不等于群组正确，旧native F1只是成员诊断；预热或缓存命中也不能代替这些证据。

延迟字段、请求调度、默认值覆盖和日志文件含义见[VLM部署与调用](../decision/VLM部署与调用.md#请求结果与耗时)。
服务是否正在运行以本机记录及实际启停为准，历史测试不证明当前端点在线。

## 8. 参考入口

数据语义与格式参考 [JRDB官方工具包](https://github.com/JRDB-dataset/jrdb_toolkit)、[原始论文](https://arxiv.org/abs/1910.11792)、[传感器说明](https://download.cs.stanford.edu/downloads/jrdb/Sensor_setup_JRDB.pdf)、[JRDB-Social](https://jrdb.erc.monash.edu/dataset/social)、[JRDB-Pose](https://jrdb.erc.monash.edu/dataset/pose)、[JRDB-PanoTrack](https://jrdb.erc.monash.edu/dataset/panotrack)、[JRDB-Act论文](https://arxiv.org/abs/2106.08827)、[PanoTrack论文](https://arxiv.org/abs/2404.01686)和[Reasoning代码](https://github.com/sjahangard/JRDB-Reasoning)。这是既有来源索引，本次未重新联网核验网页更新；本地格式/计数优先于简化示例。

研究方案的历史演变见[旧版方案](../archive/旧版方案.md)，可复核的调用原包和对照由上节耗时汇总定位。
当前工具以本文及实际代码为准，不把旧机器的源码／结果归档作为运行前提。

## 工具使用

以下从仓库根目录运行，使用具备渲染依赖的解释器及新的输出目录。已有媒体/审计仍适用时不重新生成。

### 原生标签查看

```bash
python -m decision.vlm.jrdb_visualize audit \
  --sequence gates-to-clark-2019-02-28_1 \
  --output outputs/jrdb/audits/gates_new.json
python -m decision.vlm.jrdb_visualize render \
  --sequence gates-to-clark-2019-02-28_1 --frame 000075 --view 0 \
  --layers people,social,view3d --output-dir outputs/jrdb/preview_new
python -m decision.vlm.visualize_jrdb_group_relations \
  --sequence gates-to-clark-2019-02-28_1 --layout front-near \
  --out outputs/jrdb/videos/gates_new
```

`jrdb_visualize video`支持原序列各类MP4，`montage`按源帧号合成总览。开发预览优先限定源时间或帧数，宽高取偶数。稀疏分割只在原标注关键帧显示，不能补成15Hz人工标签。显式数据根参数 `--root` 放在 audit/render/video 子命令前。

### 生成输入与人工审核

```bash
python -m decision.vlm.jrdb_t1 prepare \
  --scenes clark gates memorial --frames 75 --out outputs/t1/dev_new
python -m decision.vlm.jrdb_t1 reviewed \
  --input-dir outputs/t1/dev_new/cases/clark_000075 \
  --labels outputs/t1/dev_new/cases/clark_000075/annotation_template.json \
  --allow-draft --out outputs/t1/draft_preview_new
```

该 prepare 生成三个窗口；不传frames时默认三个场景各八个末帧，共24例。也可传完整序列名，仍需先决定根来源split；新生成不等于盲测。默认不重复生成contact sheet，需要时显式添加 `--contact-sheet`。

人工正式流程：复制annotation_template到独立labels目录，只看同一模型输入进行审核，填写真实reviewer、UTC reviewed_at和label_provenance，仲裁分歧，再去掉 `--allow-draft` 导出gold。draft仅导出预览，状态字段不能证明真实人工审核；不在标注前读取模型预测造成锚定。

### 预测与原生诊断对照

```bash
# 先按VLM手册对同一个bundle取得预测。
python -m decision.vlm.jrdb_t1 compare \
  --bundle outputs/t1/dev_new --requests outputs/t1/smoke_new/requests.jsonl \
  --out outputs/t1/compare_new
```

compare校验输入指纹、case重复、合法预测和渲染版本；只作预测/native对照，不生成完整T1金标准成绩。prepare/reviewed/compare不能输出到JRDB源目录，后两者也不能输出到输入bundle内部，软链接按真实目标检查。

### 输入产物

```text
bundle/
  manifest.json                   运输资产、case列表及非gold声明
  system_prompt.txt
  cases/<case_id>/
    model_input.json              三图、几何字段和动态Schema
    image_0.png image_1.png image_2.png
    manifest.json                 源帧、PID映射、源hash及几何签名
    annotation_template.json      draft审核起点，不进入HTTP
    system_prompt.txt             本地重建/审核使用
    input_contact_sheet.png       仅显式启用时生成
  references/<case_id>.native.json 原生成员诊断，不进入HTTP
```

冻结 `jrdb_t1_dev` 只含运输资产，适合接口/延迟参考；没有完整case源manifest时不能直接重建或伪造人工标签绑定。更换数据链接只允许相同源字节重定位，源文件/几何/渲染源码仍须匹配；旧源码绑定输入用匹配实现审核，或重新生成并复核。
