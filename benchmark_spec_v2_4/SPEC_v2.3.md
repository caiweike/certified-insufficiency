# 候选题目筛选与分层抽样脚本开发与验收技术规范 v2.4

适用入口：`01_select_candidates.py`。规则版本：`ob-candidate-rules-2.4.0`。整理日期：2026-11-16（Asia/Shanghai）。

本文件与同包的 `reference_rules_v2_3.py`、`audit_overrides.v2.3.json`、`reference_baseline/` 共同构成开发交付契约。无需查阅 v2.1、v2.2 或聊天记录。参考模块提供选题纯函数的精确定义；本文件规定输入、输出、CLI、发布协议、验收和阶段二接口。参考模块不是已经完成发布与故障恢复的生产脚本。

本版参考规则文件的原始字节 SHA-256 为 `fe99089623cef007070158605e058deb0063f6e6f064be938bc900f7118786f8`。开发交付须保留其原始字节或明确升级规则版本。

本版已对固定源数据执行参考核心复算和独立进程一致性检查。生产 CLI 的发布故障注入、其他操作系统/解释器、阶段二数学凭证验证仍是后续开发验收义务，不能据此交付包宣称已经完成。

## 1. 目标、边界与术语

1. 阶段一从 1,648 条指定来源记录中，依次执行三个个体过滤阶段、一个合格池聚簇阶段、一个分层抽样阶段，产生 100 道暂定测试题、20 道暂定开发题、完整替补池和全量账本。
2. 阶段一全部离线。运行时不访问 Hugging Face、在线模型、搜索服务或外部随机服务，不自动下载缺失文件。依赖安装与数据取得是运行前准备。
3. `estimated_clause_count` 仅是固定文本切分函数的子句数；`provisional_mechanism` 仅是暂定机制；v2.4 起移除仅针对 CEE 的解答长度与步数代理门槛，对 COMP 与 CEE 统一仅应用子句数结构门槛。
4. 本阶段不证明条件必要性、答案正确性、真实难度或绝对无同源题。聚簇断言只覆盖本版算法识别出的簇。
5. 只有阶段二完成凭证、原文对应、集合配额等验证后，才能发布正式冻结测试集。阶段一 manifest 必须标记 `PROVISIONAL_NOT_FROZEN`。
6. 下文 `STAGE_1` 至 `STAGE_5` 指阶段一脚本内部的五步；“阶段二”专指后续 `02_shard_and_verify.py`，两者不得混淆。

## 2. 官方来源与固定输入

### 2.1 上游身份

数据仓库：`Hothan/OlympiadBench`。固定 revision：

```text
91184b52131e7fc9455fef848035173aea8cc01a
```

| 来源 | repository_path | 记录数 | 文件原始字节 SHA-256 |
|---|---|---:|---|
| COMP | `OlympiadBench/OE_TO_maths_zh_COMP/OE_TO_maths_zh_COMP.parquet` | 408 | `b9e461bd6b0ea9c91939d4ecacf27e45f8a76e86e76153c5fd728a31efc5f9d9` |
| CEE | `OlympiadBench/OE_TO_maths_zh_CEE/OE_TO_maths_zh_CEE.parquet` | 1240 | `1a0164d2179c88be45c831d0c70dd0b95cf904d7b2d0f10f5f954f6c86e0dac3` |

上述哈希已与固定 revision 的 [COMP 文件](https://huggingface.co/datasets/Hothan/OlympiadBench/blob/91184b52131e7fc9455fef848035173aea8cc01a/OlympiadBench/OE_TO_maths_zh_COMP/OE_TO_maths_zh_COMP.parquet) 和 [CEE 文件](https://huggingface.co/datasets/Hothan/OlympiadBench/blob/91184b52131e7fc9455fef848035173aea8cc01a/OlympiadBench/OE_TO_maths_zh_CEE/OE_TO_maths_zh_CEE.parquet) 的公开元数据核对。

官方分类中 OE 为 Open-ended，TP 为证明题，TO 为 Text-only，COMP 为竞赛，CEE 为中国高考。不得仅因 `question_type == "Open-ended"` 而排除题目。参见 [官方命名及字段说明](https://github.com/OpenBMB/OlympiadBench#data-process)。

官方提示部分参考解含图片，Hugging Face 版本未完整提供解答图片；因此题干纯文本不自动意味着参考解完全自足。参见 [固定版本数据卡](https://huggingface.co/datasets/Hothan/OlympiadBench/blob/91184b52131e7fc9455fef848035173aea8cc01a/README.md)。

### 2.2 源记录的严格类型

使用 `pyarrow.parquet.read_table(...).to_pylist()` 得到原生 Python 值。不得用 `str(None)`、`bool(nan)` 等方式把错误输入转成有效记录。

| 字段 | 准入前类型/约束 |
|---|---|
| `id` | 原生 `int`，非 bool，非负；各来源内唯一；不得用行号重建 |
| `question` | 非空 `str`，原文不 strip、不改换行 |
| `solution`、`final_answer` | `list[str]`；列表为空属于记录级过滤，列表/元素类型错误属于输入错误 |
| `subfield`、`answer_type` | `str`；学科标签必须在第 5 节映射内 |
| `is_multiple_answer` | 原生 `bool` 或 `None`；不把 None 当 False |
| `unit`、`error` | `str` 或 `None`，原样保留；不将 error 擅自当作源题错漏标志 |
| `subject`、`language` | 固定为 `Math`、`Chinese` |
| `modality`、`question_type` | 固定为 `Text-only`、`Open-ended` |
| `context`、`image_1`…`image_9` | 本固定来源快照均为 None；若不符，报输入版本/结构错误 |

检查源哈希、记录数和类型后才执行选题。整文件哈希异常不允许“尽量继续”。从已验证的同一份字节快照解析 Parquet，避免哈希后文件被替换而读取另一份内容。

### 2.3 主键和排序

COMP 的 `source_rank=0`，CEE 的 `source_rank=1`。

```text
item_id = "ob-comp-" + str(original_id)   # COMP
item_id = "ob-cee-"  + str(original_id)   # CEE
primary_key = (source_rank, original_id)
```

规范 ID 无前导零。所有记录级排序使用上述整数二元组。桶名按字符串字典序排序；这不等于对题目 ID 做字符串排序。

## 3. 完整审计规则附件

### 3.1 基准文件与真实哈希

使用同包的 `audit_overrides.v2.3.json`，其完整内容另以易读形式写入 `AUDIT_DECISIONS.md`。实际 JSON 是 UTF-8、无 BOM、紧凑排序序列化并以一个 LF 结束。

```text
SHA-256 = 90b962f5ea7d5b0219b9808e6f4942e840934d280bbc9f3ebc82c280ee1e4b41
```

这里计算文件原始字节，包括末尾 LF；不对输入文件重新序列化后再验哈希。`.sha256` 附件便于人工核对，但脚本应与发布常量比较，不能让可随意修改的 sidecar 自己授权另一份审计表。

本版包含 1 条标志修正和 5 条排除：

| ID | 操作 | 原因 |
|---|---|---|
| `ob-cee-3106` | `True → False` | 求极值点个数，答案为单个整数 2 |
| `ob-comp-212` | 排除 | c=6 已由其他定量条件推出 |
| `ob-comp-470` | 排除 | 竖直平移参数的定值条件不影响单调递减区间 |
| `ob-comp-239` | 排除 | 所求曲线转坐标不依赖题干另给的直线和点 |
| `ob-comp-254` | 排除 | 所求曲线转坐标不依赖题干另给的直线 |
| `ob-comp-226` | 排除 | 当前发布文本存在错字、残缺与定界问题；不自动猜改 |

**移除旧版对 `ob-comp-230` 的排除。** 当前源题是 AB=3、AC=1、夹角 60°、D 为 BC 中点，求 AD。由中线公式 AD²=(2·3²+2·1²−7)/4=13/4；该题文没有旧表所称“前问答案并入”的证据。取消静态排除不承诺其一定通过其余筛选或阶段二。

审计依据来自本轮源文本和数学核查，并明确标为助手复核；不得写成已完成独立人工双人复核。文本长度检查只检查数据完整性，不证明证据正确。

### 3.2 严格 schema

顶层恰好四个键：

```text
schema_version: "audit_overrides_v2.3"
review_method: str
flag_corrections: dict[item_id, Correction]
exclusions: dict[item_id, Exclusion]
```

`Correction` 恰好包含：

```text
source_question_sha256: 64 位小写十六进制字符串
expected_source_is_multiple_answer: bool | null
override_is_multiple_answer: bool
evidence: str，strip 后长度至少 15
```

`Exclusion` 恰好包含 `source_question_sha256`、`reason_code`、`evidence`。原因仅允许：

```text
EXCL_S2_AUDIT_OVERDETERMINED
EXCL_S2_AUDIT_NON_CONSTRAINING
EXCL_S2_AUDIT_DEAD_OBJECT
EXCL_S2_AUDIT_SOURCE_TYPO
```

拒绝重复 JSON 键、NaN/Infinity、未知题目、未知字段、未知原因码，以及同一 ID 同时出现在两张表。每个条目的题干哈希必须匹配原文；标志修正还必须匹配原标志。修正只能影响多答案标志，不能绕过其他 Gate。

`--audit-json` 允许选择文件位置，不允许在本规则版本内替换内容。修改审计内容须提升规则版本、更新固定哈希并重建基准，不能默默沿用旧版本号。

## 4. 运行环境、CLI 与错误处理

### 4.1 环境和确定性范围

- 参考环境：Windows，CPython 3.11.9，pyarrow 19.0.1；依赖清单见 `requirements-reference.txt`。
- 实现语法下限 Python 3.10。其他解释器/pyarrow/操作系统必须通过基准验证后才能声明支持；本包不声称这些环境已经实测。
- 阶段一只需 pyarrow 和标准库，不需要 pandas、NumPy 或 SymPy。后续凭证验证的依赖属于阶段二。
- 选题不得使用 `random`、Python 内置 `hash()`、时间、PID、系统路径或随机 UUID 作为排序依据。
- 临时目录名可由标准库安全产生，其值不参与选题或六个核心文件。
- 相同输入字节、参数、参考规则文件，在已通过验收的环境中，六个核心文件必须逐字节相同。

### 4.2 入口参数

```bash
python 01_select_candidates.py \
  --comp-parquet ./OE_TO_maths_zh_COMP.parquet \
  --cee-parquet ./OE_TO_maths_zh_CEE.parquet \
  --audit-json ./audit_overrides.v2.3.json \
  --out-dir ./output_benchmark_v2_3 \
  --test-seed 2026 \
  --dev-seed 2027
```

上式反斜线为 POSIX shell 续行；PowerShell 可写为单行，不得把 `\` 当作 PowerShell 续行符。

| 参数 | 必需 | 默认/约束 |
|---|---|---|
| `--comp-parquet` | 是 | 本地文件路径 |
| `--cee-parquet` | 是 | 本地文件路径 |
| `--audit-json` | 是 | 本地文件路径，哈希固定 |
| `--out-dir` | 是 | 第 11 节定义的版本化结果仓库根目录 |
| `--test-seed` | 否 | 2026，十进制整数，0≤seed<2^63 |
| `--dev-seed` | 否 | 2027，同上；允许等于 test_seed |
| `--prefer-retained` | 否 | 旧版 100 题 JSONL 的路径，传入即启用继承 |
| `--overwrite` | 否 | 允许更新已有结果仓库的 CURRENT；不删除旧 release |

不提供 `--test-size`、`--dev-size`、动态阈值或自动配额放宽。总量固定为 100/20。

旧题文件必须含恰好 100 条非空 JSONL 记录、100 个不同的 `id`，且均属于当前源数据。每条可有其他旧字段；若带 `question` 或 `question_sha256`，必须与当前源记录一致。不能用模糊题干匹配恢复 ID。记录原文件字节哈希与按整数主键排序的 ID 列表的规范 JSON 哈希。继承题仍需通过所有个体 Gate、聚簇和阶段二，不因旧题身份豁免。

### 4.3 退出码

| 码 | 场景 |
|---:|---|
| 0 | 成功发布；CURRENT 指向已验证 release |
| 1 | 已有有效 CURRENT，但未传 --overwrite |
| 2 | CLI 参数非法或不支持的运行配置 |
| 3 | 至少一个桶不足测试+开发配额，打印逐桶缺口 |
| 4 | 生成文件/结果仓库校验失败 |
| 5 | 仅阶段二：替补耗尽，禁止发布冻结集 |
| 10 | 输入不存在、源哈希/审计哈希/类型/行数错误、未知学科、非法审计或旧题文件 |
| 11 | 结果仓库已有其他写进程占有发布锁 |
| 12 | 文件 I/O 或发布失败 |
| 13 | 非预期程序故障；阶段二亦用于验证器崩溃/环境错误/运行超时 |

使用显式 `verify_or_raise`/异常，关键校验不得使用可被 `python -O` 移除的 `assert`。异常须给出稳定错误名和有用定位信息；不能因代码错误将题目改为 EXCLUDED。除成功发布外，已有 CURRENT 不得被切换到不完整输出。

## 5. 学科映射与全部记录预计算

完整覆盖固定数据中的 16 个标签：

| macro_subfield | 原始 subfield 精确值 |
|---|---|
| G1_Func_Calc_Trig | Elementary Functions；Derivative；Trigonometric Functions |
| G2_Alg_Ineq | Algebra；Inequality；Complex Numbers；Set Theory；Logic |
| G3_Seq_Comb_NT | Sequence；Combinatorics；Probability and Statistics；Number Theory |
| G4_Geom_Analytic | Plane Geometry；Geometry；Conic Sections；Polar Coordinates and Parametric Equations |

不做大小写猜测、模糊匹配或自动归入其他类。未知标签报 `UnknownSubfieldError`。

对全部 1,648 条记录预先计算以下字段，即使稍后提前淘汰也保留：

```text
macro_subfield
estimated_clause_count = len(split_clauses_raw(question))
provisional_mechanism = classify_provisional_mechanism(question)
question_sha256 = SHA256(question.encode("utf-8"))
is_retained_v1 = item_id in retained_ids
```

其中 count 可为 0，但不能为空；计数不会充当真实分片数。

## 6. 文本函数的精确定义

本节全部函数的可执行定义在 `reference_rules_v2_3.py`；开发可以直接复用，也可实现等价版本，但必须得到相同基准字节。以下规则不得以另一套“更聪明”的临时分词器替换而不提升版本。

### 6.1 三种视图

1. **源文本视图**：原始 question、solution、final_answer。只用于保存、哈希与后续跨度溯源，不修改。
2. **结构视图**：用于多小问、图表引用和证明关键词识别，可以遮蔽数学内容；不得用于数学聚簇。
3. **数学 token 视图**：用于同源候选聚簇，保留所有标点、数字、符号、字体宏与 token 边界；不得替换为 MATH 占位符。

`split_clauses_raw` 保留数学内容，但会去除 `$`、合并部分空白并重组子句。因此它是分析视图，不是可用于阶段二的原文偏移索引。阶段二必须重新对原始 question 标注跨度。

### 6.2 结构保护与切分

- CJK 范围固定为 U+3400–U+4DBF、U+4E00–U+9FFF。
- 分析空白字符固定为 ASCII 空格、tab、CR、LF、form feed、vertical tab、U+00A0、U+3000。
- 结构视图先去 `$`，识别句首/行首或 `。；;：:` 后的半角/全角小问编号，再将其替换为保留标记 `__SUBQ__`。源文本若本来含该保留串，输入报错。
- 遮蔽完整、同名匹配、允许嵌套的 LaTeX 环境；以及配对且不含 CJK 的 `()`、`[]`、`{}`、`（）`。保护 `\left.`、`\right.` 的不可见定界点。
- 不匹配或未闭合的外层定界符不保护该外层跨度；已经完整匹配的内层仍可被保护。完整栈处理见参考代码，禁止以两轮正则假设任意嵌套均已处理。
- 原始子句切分不使用 MATH 占位符，而使用字符保护掩码；在受保护跨度内不切。
- 受保护跨度外按 `，。；;！!？?\n` 切分。ASCII 逗号仅当其相邻非空白字符至少一侧为 CJK 时切分。
- ASCII 句点在受保护跨度外通常切分；数字之间的小数点、符合参考代码位置规则的 `.5` 前导小数点不切分。
- 切分后将指定空白串折叠为一个空格，去首尾空白，保留长度≥3 的子句；纯小问编号片段不计入 count。

结构保护是确定性文本启发式，不宣称实现完整 LaTeX 文法。

### 6.3 设问定位

```python
ASK_PAT = r"(?:试确定|试求|(?<!要)求|计算|写出|试问|问)"
```

取得子句列表后：

1. 无子句：stem、target 均为空。
2. 没有引导词：最后一子句作为 target，其余为 stem。
3. 有引导词：取最后一个匹配子句位置 j。
4. 若 j=0：第 0 子句作为 target，其余作为 stem，支持题首设问。
5. 若 j>0，且在最后两个子句内、该子句不以冒号结束：j 到末尾作为 target。
6. 其他情况：仅 j 为 target，其余按原顺序组成 stem，支持题中设问及后置条件。

stem 用中文句号连接；target 用中文逗号连接。该连接仅供分析，不写回原题。

### 6.4 暂定机制

```python
EXTREMUM_PAT = (
    r"(?:最大值|最小值|最[大小]可能|至多|至少|取值范围|"
    r"最[大小]的?(?:正整数|实数|质数|素数|整数|自然数|有理数|值))"
)
```

先在 target 上匹配，命中则为 `TypeB_Extremum`。否则仅做一种窄范围指代解析：若 target 为“求某个单字母变量的值”，前置 stem 子句出现“最大值/最小值 + 记为/记作/为/是 + 同一变量”，则也为 B。其余为 `TypeA_UniqueValue`。

例如 `ob-comp-256` 的“其最小值为 t。求 t 的值”归 B；“最大值为 4。求 a 的值”不会因此自动归 B。暂定机制不使用官方答案、模型作答或测试表现决定。

## 7. Stages 1–3：个体过滤

每条记录按下列顺序检查，只记录首次命中。未达到的后续 Gate 不参与该题淘汰漏斗；独立审计条目仍保留在输入审计文件中。所有正则默认 Python re 默认标志，只有图像/HTML检测明确启用 IGNORECASE。

### 7.1 Stage 1

| 顺序 | exclusion_reason | 判定 |
|---:|---|---|
| 1 | EXCL_S1_EMPTY_ANSWER | final_answer 长度不为 1，或唯一字符串 strip 后为空 |
| 2 | EXCL_S1_EMPTY_SOLUTION | solution 中没有任何 strip 后非空字符串 |
| 3 | EXCL_S1_MULTIPLE_ANSWER_FLAG | 有效多答案标志不是原生 False，或 answer_type 不属于 Numerical、Expression、Interval、Equation、Tuple |
| 4 | EXCL_S1_MULTI_VALUE_STRING | 对 Numerical/Expression/Equation 执行下述保守标量多值检查 |
| 5 | EXCL_S1_IMAGE_OR_TABLE | 题干命中显式图像、表格或指定引用词 |
| 6 | EXCL_S1_PROOF_OR_OPEN | 结构视图命中证明/开放说理正则 |
| 7 | EXCL_S1_MULTI_SUBQUESTION | 至少两个受保护小问标号，且至少两个标号后片段含设问/证明词 |

有效标志由审计修正覆盖，否则取源标志。True 和 None 的排除是保守研究准入，不声称源标注绝无错误。

标量多值检查：去分析副本的外围空白、`$` 和指定句末标点；去除 `\left`/`\right` 排版宏；逐层解开包住整个答案的圆括号。若出现 `或`、`\pm`、`±`，或括号/方括号/花括号之外出现逗号或分号，判多值。`(1,2)` 因整体圆括号被解开而拦截，`f(m,n)=m+n` 的函数参数逗号保留在括号内而放行。Interval、Tuple 不使用此标量检查。此规则不能替代阶段二的答案对象验证。

图表检测精确模式见参考模块 `IMAGE_PAT`、`TABLE_PAT`、`VISUAL_WORDS`。覆盖 `\includegraphics`、HTML `<img>`/`<table>`、`<img_数字>`/`<image_数字>`、Markdown 图片/管道表、LaTeX tabular，以及“如图、右图、左图、下图、下表、上表、表中数据”。这里只硬筛 question；solution 含表格不自动淘汰。

```python
PROOF_PAT = r"(?:求证|证明[:：，]|试证|请举出一例|说明理由|是否存在[^\n]*说明)"
SUBQUESTION_ASK = r"(?:求|试求|计算|确定|写出|证明|求证)"
```

解答出现图像标记或图表引用词时，JSONL 写 `needs_solution_asset_review=true`，交阶段二核查是否自足；false 同样不构成自足证明。

### 7.2 Stage 2

依次检查：

1. `EXCL_S2_NARRATIVE_FLUFF`：question 去 `$` 后按 `[。；！\n]` 切句，存在字符长度>25、既不含 `[0-9a-zA-Z=<>+\-\\]` 又不含以下词表的句子。

```python
MATH_KEYWORDS = (
    r"函数|数列|三角形|椭圆|双曲线|抛物线|集合|概率|随机变量|方差|均值|期望|"
    r"方程|不等式|直线|平面|向量|复数|实数|整数|正数|质数|素数|切线|极坐标|"
    r"参数|最大值|最小值|取值范围|定义域|值域|单调|对称|周期|四面体|球|圆|正方形|矩形"
)
```

2. `EXCL_S2_UNREFERENCED_FUNCTION`：用下式提取显式函数定义符号，按字符排序逐一检查；在 question 中独立标识符出现一次、在所有 solution 以 LF 拼接的文本中零次，则淘汰。

```python
FUNCTION_DEF = r"([fghFGH])\s*(?:\\left)?\(\s*[xXnN]\s*(?:\\right)?\)\s*="
IDENTIFIER = r"(?<![a-zA-Z\\])" + re.escape(fn) + r"(?![a-zA-Z])"
```

3. 静态排除：若 ID 在 exclusions 中，记录其配置原因码。

“未引用函数”只是文本条件；不宣称已证明任意未引用条件都冗余。几何对象 D/l 等的语义必要性由明确审计条目或阶段二处理，不能以单字母字符串出现次数一概判断。

### 7.3 Stage 3

1. `estimated_clause_count < 3`：`EXCL_S3_TOO_FEW_CLAUSES`。
2. `estimated_clause_count > 9`：`EXCL_S3_TOO_MANY_CLAUSES`。

注：v2.4 已移除原 3.3 条（仅针对 CEE 的 `EXCL_S3_CEE_BELOW_DIFFICULTY_GATE` 与 `STEP_PAT` 门槛），COMP 与 CEE 执行完全一致的结构子句范围门槛。

## 8. Stage 4：合格池聚簇和代表选举

仅通过前面三个阶段的题参与聚簇；其他题的两个 cluster 字段为空。

### 8.1 数学 token 及两条合并判据

`math_tokens(text)` 仅将 `$` 替换为空白、移除 `\left`/`\right` 排版定界宏及其不可见点，不删除数学标点，不删除字体宏，不把任何完整方程/参数列表替换为占位符。

按以下优先级扫描 token：LaTeX 命令、反斜线加单个非字母字符、完整十进制数/小数、连续 ASCII 字母串、其余每一个非指定空白字符。结果是 token 元组，不能把 token 重新无分隔拼接。

因此以下必须不同：`1.2`/`12`、`(1,23)`/`(12,3)`、`x:y=2:3`/`xy=23`、`1 2`/`12`、`\mathbf{x}`/`\mathrm{x}`。只去排版差异也不等于证明数学等价。

- `first_sentence_key`：取按第 6 节保护规则，在句号、分号、感叹号、问号或换行处分出的第一个非空句子，转为 token 元组。逗号不作为首句边界。
- `stem_trigrams`：取 stem 的数学 token；少于 20 个 token 返回空集，否则取连续三个 token 的集合。
- 合并条件 A：首句 token 数≥18，且两个首句 token 元组完全相等。
- 合并条件 B：两组三元组均非空，且 `100 * intersection_count >= 65 * union_count`。

B 使用整数交叉乘法，避免浮点阈值边界。A 或 B 为真即连边；按整数主键顺序枚举所有 i<j 的题对，取无向图连通分量。允许传递合并，即 u~v、v~w 时三者同簇；不声称所有成员都两两达到阈值。

### 8.2 代表和簇 ID

簇代表优先键：

```python
(0 if prefer_retained and is_retained_v1 else 1, source_rank, original_id)
```

每簇只允许一个代表进入抽样。其他成员：

```text
status_code = EXCLUDED
exclusion_stage = STAGE_4
exclusion_reason = EXCL_S4_CLUSTER_NON_REP
cluster_representative_id = 代表题 item_id
```

簇 ID 使用**全部成员**按整数主键排序后的 ID 列表：

```python
stem_cluster_id = "CL_" + SHA256(canonical_json(sorted_member_ids))
```

保留完整 64 位摘要。它不依赖当前代表身份，启用旧题优先但成员不变时簇 ID 不变。所有成员写相同的簇 ID；代表也写自己的 representative_id。

阶段二若代表失败，按替补桶队列补位，不自动恢复已排除的同簇其他成员；这是明确的取样规则，避免不同失败路径引入隐式簇内重选。

## 9. Stage 5：八桶、继承与确定性抽样

桶名：`macro_subfield + "__" + provisional_mechanism`。四个宏类各有 A/B 两桶，桶按名称字典序处理。

| 类型 | 每宏类 Test | 每宏类 Dev | 桶最低容量 |
|---|---:|---:|---:|
| TypeA_UniqueValue | 16 | 3 | 19 |
| TypeB_Extremum | 9 | 2 | 11 |

抽取前一次性检查全部八桶。任一容量不足，退出码 3，并报告每个不足桶的 available/required/shortfall；不跨桶借题、不放宽 Gate、不降低目标总量。

对每个桶：

```python
test_key = (
    0 if prefer_retained and item.is_retained_v1 else 1,
    SHA256(f"{test_seed}:{bucket_id}:{item_id}".encode("utf-8")),
    source_rank, original_id,
)
dev_and_reserve_key = (
    SHA256(f"{dev_seed}:{bucket_id}:{item_id}".encode("utf-8")),
    source_rank, original_id,
)
```

摘要取 64 位小写十六进制，升序比较。先按 test_key 取固定 Test 配额；其余按 dev_and_reserve_key 排序，前固定配额为 Dev，其余为 Reserve。

Test/Dev 各桶按抽取顺序赋 `slot_index=1..n`，Reserve 按剩余顺序赋整数 `reserve_rank=1..N`。旧题多于同桶 Test 配额时也按 test_key 决定；未保留旧题可进入 Dev/Reserve，不保证 100 道旧题全部继承。

开发集是给定筛选、簇代表及测试抽取条件下的剩余池划分。它与测试集不重叠，但不能声称与旧题继承策略统计独立。

## 10. 六个核心文件和序列化契约

### 10.1 规范字节

所有文件 UTF-8 无 BOM，使用 LF。每个 JSON 文档/JSONL 记录：

```python
(json.dumps(obj, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
```

不得额外缩进、写时间戳或用平台换行。JSONL 空集合为零字节；非空集合每条恰好一行。JSON 字符串中的源换行由 JSON 转义保存，不能改动解码后的字符串。

CSV：固定列序，逗号分隔，双引号 quotechar，`QUOTE_MINIMAL`，doublequote=true，`lineterminator="\n"`。bool 编码为 `true`/`false`，nullable int 的 null 编码为空串，整数为无前导零十进制。解析时按 schema 恢复类型后比较，不能直接将 JSON bool 与 CSV 文本比较。

### 10.2 全量账本：selection_eligibility_ledger.csv

严格 15 列、1 行表头和 1,648 条数据记录；按整数主键排序。本固定字段内容不含换行，故本版基准也恰为 1,649 物理行；一般 CSV 的记录数校验仍应通过 CSV 解析器完成。

| 顺序 | 字段 | JSON 对应类型/取值 |
|---:|---|---|
| 1 | item_id | str |
| 2 | source_dataset | `OE_TO_maths_zh_COMP` 或 `OE_TO_maths_zh_CEE` |
| 3 | original_id | int |
| 4 | subfield | 源 str |
| 5 | macro_subfield | 四个枚举之一 |
| 6 | provisional_mechanism | TypeA_UniqueValue / TypeB_Extremum |
| 7 | estimated_clause_count | 非负 int |
| 8 | is_retained_v1 | bool |
| 9 | status_code | SELECTED_TEST / SELECTED_DEV / ELIGIBLE_RESERVE / EXCLUDED |
| 10 | exclusion_stage | 空串或 STAGE_1…STAGE_4 |
| 11 | exclusion_reason | 空串或本版原因枚举 |
| 12 | stem_cluster_id | 空串或 `CL_`+64 位摘要 |
| 13 | cluster_representative_id | 空串或有效 item_id |
| 14 | reserve_rank | int 或 null；CSV null 为空串 |
| 15 | question_sha256 | 原始 question 的 UTF-8 字节 SHA-256 |

非 EXCLUDED 的两个 exclusion 字段必须为空；所有 EXCLUDED 必须填写合法且对应阶段的原因。Stage 1–3 淘汰题的两个 cluster 字段为空；Stage 4 淘汰题必须能关联到同簇代表。只有 Reserve 的 reserve_rank 非空。

### 10.3 三个 JSONL 数据文件

文件分别为 `provisional_test_100.jsonl`、`provisional_dev_20.jsonl`、`eligible_reserve_pool.jsonl`。每条包含上述 15 字段及以下 10 字段，恰好 25 字段：

| 字段 | 类型和语义 |
|---|---|
| question | str，源 question 原样 |
| final_answer | str，源 final_answer[0] 原样，不 strip、不删 $ |
| answer_type | str，源值 |
| solution | list[str]，保留全部解法、列表顺序及各字符串 |
| unit | str/null，源值 |
| error | str/null，源值 |
| source_is_multiple_answer | bool/null，源值 |
| effective_is_multiple_answer | bool，所有入选/替补题必须为 false |
| slot_index | Test/Dev 为桶内正整数槽位；Reserve 为 null |
| needs_solution_asset_review | bool，按第 7 节提示解答资源核查 |

Test/Dev 文件分别按 `(bucket_id, slot_index)` 排序；Reserve 按 `(bucket_id, reserve_rank)` 排序。三者的 ID 必须分别与账本对应状态双射；文件名不能与 status_code 不符。源标志修正不改动原始题文或原始标志字段。

### 10.4 selection_manifest.json

精确字段及序列见参考核心 `build()` 返回的 manifest 和 `reference_baseline/selection_manifest.json`，包括：

- schema_version、rule_version、`rules_sha256`；后者绑定随包参考规则文件的原始字节，而不是各开发者重写的入口程序。
- selection_status=`PROVISIONAL_NOT_FROZEN`。
- dataset_repository、固定 dataset_revision、COMP/CEE 固定来源路径/行数/哈希。
- audit schema、原始文件哈希、实际应用的全部 flag correction 及前后值/题干哈希。
- retained_input：未启用为 null；启用时含 raw_sha256、id_set_sha256、count=100。
- parameters：test_seed、dev_seed、prefer_retained、test_total=100、dev_total=20。
- funnel：Stage 1–4 的 entered/excluded/passed；全部原因码计数，包含 0 值项。
- 八桶 total_eligible/selected_test/selected_dev/reserve_count，以及四状态总数。
- outputs：四个数据文件的 SHA-256、字节数和数据记录数。

manifest 不保存输出目录、输入绝对路径、PID、时间戳、主机名、解释器版本或生产脚本实现哈希，避免机器/位置差异改变规范字节。实际环境、绝对路径、入口程序哈希与运行诊断写 stderr 的执行日志，由调用方归档。

本包基准报告记录了实际测试环境。生产实现仍须记录自己的运行环境，不能冒用本包参考环境记录。

### 10.5 selection_manifest.json.sha256

内容恰为 `SHA256(manifest 原始字节) + "\n"`，65 个 ASCII 字节，无文件名。manifest 不包含自身或此 sidecar 的哈希，避免自引用。

## 11. 发布协议：不可变 release 与原子 CURRENT 指针

### 11.1 结果仓库布局

```text
<out_dir>/
  STORE.json
  .publish.lock
  CURRENT
  releases/
    <manifest_sha256>/
      selection_eligibility_ledger.csv
      provisional_test_100.jsonl
      provisional_dev_20.jsonl
      eligible_reserve_pool.jsonl
      selection_manifest.json
      selection_manifest.json.sha256
```

`STORE.json` 为规范 JSON `{"store_schema":"ob_candidate_store_v1"}`。CURRENT 为 manifest 摘要加 LF。`out_dir` 是结果仓库，六个文件实际位于 release 内，不在仓库根直接覆盖。

已有非空目录若缺少正确 STORE.json，拒绝将其当作结果仓库；不得清空任意用户目录。结果仓库及发布目标拒绝符号链接/目录联接等可能使写入逃出预期路径的对象。

### 11.2 写入步骤

1. 先完成输入验证和纯函数构建。创建/验证结果仓库结构，对 `.publish.lock` 获取进程级排他锁；Unix 可用 flock，Windows 可用 msvcrt 锁定固定字节。锁文件可以长期存在，锁的持有由操作系统判定，不凭文件存在判断“有进程”。不能取得锁则退出 11。
2. 持锁后重新读取 CURRENT。若已有有效 CURRENT 且未传 --overwrite，退出 1。若 CURRENT 格式错误或指向损坏 release，退出 4，不默默覆盖破损状态。
3. 在同一文件系统的 `releases/` 内创建本次独占 staging 目录。写入六个文件，flush、关闭，并在适用平台对文件 fsync。执行第 13 节所有运行时校验。
4. 以 manifest 实际摘要 H 命名 release。若 `releases/H` 不存在，原子重命名 staging 为该目录。若已存在，必须验证六文件字节完全相同；相同则复用，不同则报错，不覆盖不可变 release。
5. 在 out_dir 内写临时指针文件，内容 `H + LF`，flush/关闭并同步。执行一次 `os.replace(temp_pointer, CURRENT)`。只有此单文件替换是对读者生效的提交点。
6. CURRENT 提交成功后才打印成功摘要并返回 0。保持旧 release，不在普通 --overwrite 中删除历史结果。释放锁。

生产脚本不能通过递归删除 out_dir 实现 overwrite。失败清理只允许删除本进程刚创建且路径已验证位于仓库内的 staging/指针临时文件。

### 11.3 读者和崩溃恢复

- 读者先读取 CURRENT 一次并校验格式，再到对应不可变 release 校验 manifest/数据哈希；不能在读取四个文件之间反复读取 CURRENT。
- 提交前进程退出：旧 CURRENT 继续有效；首次发布则尚无可用 CURRENT。孤立 staging/release 不自动变成正式结果。
- 提交后进程退出：CURRENT 指向已完整校验的 release。新启动进程可以忽略孤立临时目录；不自动提升孤立 release，也不因其存在删除有效结果。
- 同时写同一 out_dir 必须由排他锁串行化；其他结果仓库不受影响。
- 原子可见性承诺限定于遵循 CURRENT 协议、使用支持原子同文件系统 rename 的本地文件系统。网络同步盘及突然断电后的持久性需平台专项验证，不扩张为未经测试的保证。

此方案不再使用“旧目录移走，再将新目录移入”的双重命名来声称事务级原子替换。

## 12. 阶段二接口及冻结规则

本节规定阶段一交付物如何被消费和验证结果如何影响补位；不要求开发阶段一时实现通用数学自动证明器。

### 12.1 输入与完整题目契约

阶段二解析阶段一 CURRENT，锁定一个 release H，验证 manifest、四个数据文件及本版 schema。原始 Test/Dev 和所有 Reserve 的完整正文、答案、解答已在 JSONL 中，无需重新读取上游 Parquet。不得修改阶段一 release。

阶段二的规则调试只使用初始开发候选，冻结验证规则及其版本后，才执行最终确定性补位。规则未冻结前不得消费共享 Reserve。调试后若改规则，必须提升版本并从阶段一的原始槽位和 rank=1 重新执行最终状态机。

不按受测模型的 FULL/CONCAT 正确与否或拒答结果淘汰测试题。模型表现属于实验结果；原题、分片及 CONCAT 的信息对应由静态与数学验证完成。

### 12.2 分片数定义与数学证据范围

本版使用 `canonical_k`，不再使用暗示全局最小性的 k_true 命名。它是**版本化原文条件单元划分下的片数**，S1 计为一片。留一法只证明所验证条件单元的不可删性，不证明所有等价表述/划分中的最少片数。

划分原则：

- S1 包含对象名称、通用记号约定与求解目标；目标句内夹带的定量限制必须另归条件单元，不能借“整个目标句”隐藏条件。
- 独立写出的方程、初值、域限制、几何关系各为条件单元。并列写出的 `x>0，y>0` 为两单元；链式 `0<x<1` 和共享谓词缩写 `x,y>0` 保留为一原文单元，内部逻辑原子分别登记，不能声称这一写法与改写后片数相同。
- 坐标定义 A(1,2)、完整函数定义、分段函数各作为一个定义单元；内部逻辑成分必须列明。显示方程组的外层排版环境本身不构成数学合并理由，能按原文跨度分别标注的独立方程分别归单元。
- 跨度使用原始 question 的 Unicode 码点半开区间 `[start,end)`，可为多个不相交跨度。语义文本不可凭空增加或删除；为显示补充的 LaTeX 排版定界符须另标记为非语义 scaffolding，不得加入假设。
- 数学原子到源跨度、到分片均有显式对应。共享谓词/复合定义内部还要检查是否藏有与目标无关的独立附加条件；若有，应拒绝而非用分组掩盖。
- 无法在此规则下明确划分、证明信息对应或形式化的题，返回有证据的失败；不能凭模型“看起来可解”通过。

阶段二必须验证：原题形式化的可满足性、源参考答案与完整目标的一致性、源条件覆盖、完整题与 CONCAT 的语义对应，以及每个 j≥2 条件片的留一必要性。几何存在性、函数/变量定义域与模型类型必须显式登记，不能通过未声明默认值制造证书。

TypeA 的 WitnessPair 要给出两个满足“移除该片后的所有可见约束”的合法实例，且求解目标值不同。还需单独证明完整条件下目标正确且唯一。单个反例对不能代替完整答案验证。

TypeB 的 RelaxedCertificate：若目标为最大/最小值，给出满足放宽后约束、且目标严格优于完整题所声明最优值的可行实例，并验证完整题最优值及其界；若目标为取值范围，给出放宽后可行、但投影值落在完整答案范围外的实例，并验证完整范围。正负、严格/非严格、不存在最值与上确界的区别须由凭证格式明确。不得把关键词本身当作数学机制证明。

PASS 要求 3≤canonical_k≤6、完整答案验证通过、全部片留一验证通过、未发现隐藏冗余、verified_mechanism 等于 provisional_mechanism。未支持的凭证不能通过；不允许仅数值近似采样就声称形式化证明。

### 12.3 单题返回及缓存

```text
item_id: str
verify_status: VERIFY_PASS | VERIFY_FAIL
canonical_k: int | null
verified_mechanism: TypeA_UniqueValue | TypeB_Extremum | null
reject_reason: str，PASS 为空
certificate_sha256: str | null
verification_report_sha256: str，PASS/FAIL 均必须有完整报告
binding_digest: str
```

PASS 的 certificate_sha256、canonical_k、verified_mechanism 必须非空。FAIL 原因枚举：

```text
FAIL_K_OUT_OF_RANGE
FAIL_REDUNDANT_PREMISE
FAIL_MECHANISM_MISMATCH
FAIL_AMBIGUOUS_STEM
FAIL_SOURCE_ANSWER_MISMATCH
FAIL_MISSING_REFERENCE_ASSET
FAIL_CERTIFICATE_UNAVAILABLE
FAIL_SHARD_ALIGNMENT
```

“未构造出凭证”使用 FAIL_CERTIFICATE_UNAVAILABLE，不冒充已经证明题干歧义或数学错误。解答图片缺失但可用其他已验证凭证完整验证时，不必因此淘汰；缺失资源使验证无法成立时才失败。

程序崩溃、文件损坏、依赖缺失、机器运行超时不是题目失败，整个运行退出 12/13并保持当前槽位未提交。若未来使用确定性符号搜索预算作为准入界限，预算、计步算法必须进入规则版本；不能按机器快慢决定补位。

`binding_digest` 必须绑定：阶段一 manifest H、完整候选记录的规范 JSON 哈希、宏类与暂定机制、阶段二规则版本和实现/验证器配置摘要、形式化定义域及分片方案摘要。报告还绑定证书原字节哈希与验证器版本。恢复时重新校验这些输入及报告/凭证文件的实际字节；不能只比较 question_sha256 后复用 PASS。

尚未形成形式化或分片方案的失败记录，对相应摘要显式写 null，不能填占位哈希。报告文件不得包含自己的 verification_report_sha256；该摘要只写在返回对象、事件与索引中。证书不得反向依赖报告哈希，binding_digest 不包含报告结果哈希，形成“输入 → 证书 → 报告 → 事件”的无环哈希依赖。缓存当前输入哈希必须由当前真实输入重新计算，不能直接信任旧报告自报的值。

### 12.4 确定性替补状态机

固定顺序：先全部 Test，再全部 Dev；每一集合内按桶名，再按初始 slot_index 递增。每个桶一个全局共享替补游标，初始 1。Test/Dev 初始候选已预留，不能被拿去补另一集合。

```text
for split in [TEST, DEV]:
    for bucket in sorted(all_8_buckets):
        for slot in initial_slots[split, bucket]:
            验证原始候选
            若 PASS：填槽；继续下一槽
            若 FAIL：记录 INITIAL_REJECTED_IN_STAGE2
            while 槽未填满:
                若无剩余 Reserve：退出 5，不发布冻结集
                获取本桶 next_reserve_rank 的题
                按当前绑定输入验证
                提交此顺位的验证事件并消费该 rank
                若 PASS：PROMOTED_TO_TEST/DEV，填槽
                若 FAIL：RESERVE_REJECTED_IN_STAGE2，取下一顺位
```

机制不符的题直接失败，不改桶、不从其真实机制桶反向补入。已经消费的替补不能再次分配。

验证报告采用不可变文件，消费事件顺序写入 `stage2_events.jsonl` 并同步；事件含序号、集合、桶、槽位、item_id、rank或null、binding_digest、报告摘要、结果。只有完整提交的事件才推进游标。崩溃留下的末尾不完整行不能视为已消费；恢复时从完整事件前缀重建槽位和各桶指针，对未提交尝试可复用合法报告后提交一次事件。

汇总账本 `stage2_verification_ledger.csv` 可以由事件日志重建。恢复必须检查事件顺序与固定状态机相容，禁止跳过未消费 rank。

### 12.5 正式冻结出口

最终必须恰好 Test=100、Dev=20、每桶配额仍精确、120 个题 ID 和 cluster ID 均唯一、所有题均有同版本验证报告。由独立 `verify_certificates.py` 回读证书、源映射及报告复验；通过后以与第 11 节相同的不可变 release/CURRENT 协议发布正式结果。

冻结文件至少包含 `frozen_test_100.jsonl`、`frozen_dev_20.jsonl`、分片/凭证文件、状态机事件、最终 manifest。最终 manifest 绑定阶段一 H、阶段二规则摘要、各槽最终 ID 和替补轨迹。正式运行读取最终冻结 release，不能误读阶段一 provisional 文件。

阶段二耗尽或失败时可以保留诊断、事件和凭证缓存，但不得输出带“已冻结”标记的 CURRENT。

## 13. 验收：运行时检查与外部集成测试分开

### 13.1 生产脚本每次必须执行的回读校验

在 staging 中读取真实落盘文件，用显式异常检查：

1. 所有输入哈希、类型、行数、ID 唯一性和审计绑定正确。
2. 账本恰好 1,648 记录，字段顺序正确，与源主键双射且整数主键严格递增。
3. 四状态集合互斥、并集为全源；Test=100、Dev=20，总数守恒。
4. 三个 JSONL 与对应账本状态的 ID 集合及记录数双射；无重复 ID。
5. 三个 JSONL 的 25 字段类型准确；其 15 个共享字段按显式 CSV 类型转换后与账本一致。
6. question、final_answer、solution、answer_type、unit、error 与源数据逐字段相等；重新计算 question_sha256。
7. 八桶 Test/Dev 精确配额；slot_index 连续且文件排序正确。
8. Test+Dev+Reserve 均为簇代表，ID和cluster ID互不重复；Stage 4 排除记录确实指向该连通分量代表。
9. 所有提前淘汰空值、预计算非空字段、合法原因码/阶段关系闭合。
10. Reserve 排名和排序正确，不能只检查排名集合而忽略实际顺序；非 Reserve 的 rank 为空。
11. 审计排除全部生效、标志修正与源 flag 及 manifest 应用记录相符；不能强行要求提前被另一原因淘汰的题记录 audit 原因。
12. 漏斗每级 entered=上一层passed、entered=excluded+passed，by_reason 和状态总数与账本一致；桶汇总与实际集合一致。
13. 核心文件 UTF-8/LF/BOM/JSON/CSV 序列化正确，数据哈希、字节数、记录数和 manifest sidecar 全部闭合。
14. 当前版本的纯函数回归样例成立，至少包含本节后续列出的数学标点与设问用例；不启动子进程。

运行时失败退出 4，并保留旧有效 CURRENT。生产脚本不能调用下面的跨进程验收器。

### 13.2 独立集成验收

开发者交付 `test_select_candidates.py` 或等价测试驱动，独立执行：

- 固定输入在 PYTHONHASHSEED=0 和 42 的独立进程运行，使用不同输入副本路径/输出目录；两个 release 中六文件全部逐字节相同。
- 再使用 `python -O` 运行，结果仍相同，坏输入仍被拒绝。
- 与随包 `reference_baseline/` 的全部六文件比较。规则文件、审计、种子和继承模式必须相同。
- 测试缺失输入、坏哈希、未知/重复审计键、审计题干绑定不符、非法旧题文件、重复 ID 等失败行为。
- 用显式测试夹具验证某桶不足时退出 3；测试不修改正式规则以“造出成功”。
- 验证旧题优先、旧题超额、旧题不合格、同簇旧题代表优先等情形。
- 首次发布、无 overwrite 重跑、相同内容 overwrite、不同种子 overwrite、同目录并发写锁均按协议执行。
- 在 staging 写入后、release 提升后、CURRENT 切换前后分别注入进程退出；读者只能看到旧有效 release 或新有效 release，不能看到混合文件。首次发布前中断允许无 CURRENT。
- 测试已有非结果目录、符号链接/目录联接、不可写目录，不能误删源数据或用户文件。
- 阶段二另以夹具验证测试/开发共用游标、连续多个失败、耗尽、报告缓存绑定变化、未完成事件恢复和终止后不冻结。

跨进程测试由外部驱动调生产脚本；生产脚本仅做自身运行时检查，避免无限递归启动。

### 13.3 最小回归样例

| 用例 | 必须结果 |
|---|---|
| `(1)求a的值；(2)求b的值` | 多小问 True |
| `（1）求a的值；（2）求b的值` | 多小问 True |
| ob-comp-289 | f(1) 不触发多小问；该题仍可因有效多答案标志被 Stage 1 排除 |
| ob-comp-700 / ob-comp-530 | 题首求最小值目标被识别；暂定 B；条件编号不自动视作小问 |
| ob-comp-256 | “最小值为 t，求 t”暂定 B |
| “最大值为4，求a的值” | 不能只因前置极值词归 B |
| `x=1.2` 与 `x=12` | 数学 token 不同 |
| `A(1,23)` 与 `A(12,3)` | 数学 token 不同 |
| `x:y=2:3` 与 `xy=23` | 数学 token 不同 |
| `1 2` 与 `12` | 数学 token 不同 |
| `\mathbf{x}` 与 `\mathrm{x}` | 字体宏保留，token 不同 |
| `$(1,2)$`，标量类型 | 多值检查 True |
| `$f(m,n)=m+n$` | 多值检查 False |
| `$\frac{1}{50}；$` | 句末分号不触发多值检查 |
| ob-cee-3106 | raw flag=True，effective=False，修正证据可追溯 |
| ob-comp-230 | 不在静态排除表，不承诺绕过其他 Gate |

## 14. v2.4 实测基准

条件：固定两份 Parquet、本包审计表、test_seed=2026、dev_seed=2027、未启用继承、本包参考规则（移除原 3.3 CEE 解答长度门槛）。不能把此容量用于其他规则版本或继承模式。

| 阶段 | 进入 | 淘汰 | 通过 |
|---|---:|---:|---:|
| STAGE_1 | 1648 | 499 | 1149 |
| STAGE_2 | 1149 | 30 | 1119 |
| STAGE_3 | 1119 | 229 | 890 |
| STAGE_4 | 890 | 249 | 641 |

最终：EXCLUDED=1007，SELECTED_TEST=100（58 CEE + 42 COMP），SELECTED_DEV=20（11 CEE + 9 COMP），ELIGIBLE_RESERVE=521。

| 宏类 | A 容量 | B 容量 | A 替补 | B 替补 |
|---|---:|---:|---:|---:|
| G1_Func_Calc_Trig | 113 | 114 | 94 | 103 |
| G2_Alg_Ineq | 45 | 45 | 26 | 34 |
| G3_Seq_Comb_NT | 99 | 28 | 80 | 17 |
| G4_Geom_Analytic | 161 | 36 | 142 | 25 |

详细原因计数见 `reference_baseline/baseline_report.json`；包含零值的完整漏斗也在基准 manifest 中。基准 manifest SHA-256：

```text
51bcb9039d0c269fa073744ea76dcaaf73be7297c08366063f12ae332a66427f
```

已执行参考验收：PYTHONHASHSEED=0、42，以及 42+`-O`，六个核心文件字节一致；输入枚举反转不影响结果；源文回读、审计、配额、分片标签回归和继承夹具检查通过（共 8,330 项检查）。实际平台为 Windows/CPython 3.11.9/pyarrow 19.0.1。证据见 `reference_verification.json`。

此结果证明阶段一基准可执行、初选容量充足；不证明 120 道题全部能通过阶段二，也不保证替补池一定足够。阶段二耗尽必须按规定失败，不能倒改阶段一配额掩盖。

## 15. 开发交付清单与变更控制

开发者最终交付至少包括：

1. 生产入口 `01_select_candidates.py`，完整 CLI、锁、CURRENT 发布和错误码实现。
2. 本规范要求的规则实现及版本化审计附件。
3. 独立集成测试驱动和实际运行报告，明确已测试环境与未测范围。
4. 默认模式生成的六文件与本包基准逐字节对比记录。
5. 继承、容量不足、坏输入、发布中断和并发故障用例结果。
6. 阶段二消费接口样例，避免把暂定输出误认为冻结实验数据。

任何修改文本切分、聚簇 token/阈值、原因优先级、审计条目、学科映射、机制分类、配额、排序键、字段或序列化方式的变更，都要更新规则版本并重生成基准。只改变部署路径或非语义日志不应改变六文件。

本版的 641/521 及上述哈希取代 v2.3 的 404/284（移除原 Stage 3 第 3.3 条仅针对 CEE 的解答步数/长度门槛）。对 ob-comp-230 的修订及全部审计证据见 `AUDIT_DECISIONS.md`。

