# v2.4 开发交付包（已移除条件 3.3 高考解答长度门槛）

先读 **SPEC_v2.3.md**（已更新至 v2.4 规则与实测基准）。它是完整脚本需求文档；其余文件为精确规则附件、审计依据和可复算验收基准。

| 文件 | 用途 |
|---|---|
| SPEC_v2.3.md | 完整 CLI、字段、算法、错误码、发布、阶段二接口和验收规范（v2.4 更新） |
| reference_rules_v2_3.py | 规范性纯函数核心，固定所有选题与序列化细节（`ob-candidate-rules-2.4.0`） |
| audit_overrides.v2.3.json | 实际使用的完整审计配置，5 条排除、1 条标志修正 |
| audit_overrides.v2.3.json.sha256 | 审计文件原始字节哈希 |
| AUDIT_DECISIONS.md | 可读证据，包括撤回 ob-comp-230 的错误过定指控 |
| reference_baseline/ | 六个暂定基准文件及环境/漏斗报告 |
| recompute_baseline.py | 用本地官方对应 Parquet 重算基准；输出目录必须未存在 |
| verify_reference.py | 外部验收驱动；包含跨进程与 -O 检查，不会递归调用自身 |
| reference_verification.json | 本轮实际执行的参考核心验收记录 |
| requirements-reference.txt | 参考运行依赖 |
| PACKAGE_MANIFEST.json | 本交付包各文件的相对路径、字节数和 SHA-256 |

本包没有把生产入口命名为 01_select_candidates.py：发布锁、CURRENT 提交、崩溃恢复等仍需开发者按规范实现。参考核心和基准重算工具不承担生产发布。阶段二数学凭证也未在此包执行。

## 已核验的基准

- 固定官方文件、规则版本 ob-candidate-rules-2.4.0（已移除原 3.3 仅针对 CEE 的解答长度/步数门槛）、test_seed=2026、dev_seed=2027、无继承。
- 合格簇代表 641：暂定 Test 100（58 CEE + 42 COMP）、暂定 Dev 20（11 CEE + 9 COMP）、Reserve 521；排除 1007。
- 八桶全部满足配额。
- Windows / CPython 3.11.9 / pyarrow 19.0.1 下，PYTHONHASHSEED=0、42、42 加 -O 的六文件逐字节一致。
- 这些文件均为 PROVISIONAL_NOT_FROZEN，不能直接用作已经完成数学验证的正式实验基准。

## 本地重算

在本包目录下执行以下单行命令，将两份源文件路径替换为实际路径；new_baseline 必须尚不存在：

```text
python -B recompute_baseline.py --comp-parquet <COMP.parquet路径> --cee-parquet <CEE.parquet路径> --audit-json audit_overrides.v2.3.json --output new_baseline
```

运行外部参考验收：

```text
python -B verify_reference.py --comp-parquet <COMP.parquet路径> --cee-parquet <CEE.parquet路径> --audit-json audit_overrides.v2.3.json --report my_reference_verification.json
```

这些命令只读源 Parquet。依赖可在联网准备环境安装，实际运行不访问网络。交付包不重复打包官方 Parquet；固定版本下载位置和 SHA-256 已在规范第 2 节给出。

## 哈希与版本

所有随包规范文件使用 UTF-8 无 BOM 和 LF。不要让编辑器或 Git 自动改成 CRLF，否则原始字节哈希改变。若修改规范性规则或审计内容，应更新版本并重算基准；不得仅把旧哈希替换成新值却继续冒称原发布版本。

PACKAGE_MANIFEST.json 不包含自己、其 sidecar 或外层 ZIP，避免哈希自引用。外层 ZIP 的哈希由同目录 sidecar 提供。

