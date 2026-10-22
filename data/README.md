# 示例数据与业务材料

| 路径 | 内容 | 用途 |
| --- | --- | --- |
| `churn.csv` | 5000 行模拟电信客户数据，流失率约 26.6% | 主场景：表格二分类 |
| `tickets.csv` | 2000 条模拟客服工单，共 4 类，类别不均衡，含跨类表述与约 6% 标签噪声 | 迁移场景：中文文本分类 |
| `scripts/make_churn.py`、`scripts/make_tickets.py` | 数据生成脚本，随机种子固定，可复现 | `python data/scripts/make_churn.py` |
| `materials/manifest.yaml` | 业务材料清单：所属场景、代码仓库、文档 | 能力抽取的入口 |
| `materials/legacy_repo/` | 模拟的历史代码仓库 `churn_v1`，含 9 个公开函数 | 抽取算子、依赖和参考源码 |
| `materials/docs/` | 需求说明、实验记录、失败复盘、调参经验 | 抽取指标、约束、历史实验和经验 |

`churn.csv` 中故意保留了几处真实数据里常见的"脏点"，供后续的验证、修复和经验复用环节使用：
- `total_charges` 是字符串列，新入网客户（tenure=0，共 69 人）的值是空白字符串 `" "`；
- `monthly_charges` 约 3% 缺失，`payment_method` 约 2% 缺失；
- `customer_id` 是高基数标识列，不应参与建模。
