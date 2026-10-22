# 失败案例复盘

## 1. total_charges 转换报错
- 现象：`ValueError: could not convert string to float: ' '`
- 原因：新入网客户的累计消费为空白字符串，pandas 读入后整列是 object 类型。
- 修复：先 `str.strip()`，再 `pd.to_numeric(errors="coerce")`，缺失填 0。
- 教训：数值列读入后要先检查 dtype；object 类型的"数值列"会被 one-hot 当成类别，维度暴涨。

## 2. 客户 ID 被 one-hot 编码
- 现象：训练内存溢出，特征维度从 40 涨到 5000+。
- 原因：`customer_id` 未删除，被当作类别特征。
- 修复：建模前删除标识列；编码器设置最大类别数。

## 3. 召回率不达标
- 现象：默认 0.5 阈值下 recall 只有 0.55 左右。
- 原因：正负样本 1:3 不平衡，模型偏向多数类。
- 修复：`class_weight="balanced"`，或在验证集上调阈值以满足最低召回。

## 4. 预测阶段出现未见类别
- 现象：线上出现新的支付方式"数字人民币"，`OneHotEncoder` 抛出 `ValueError: Found unknown categories`。
- 修复：编码器使用 `handle_unknown="ignore"` 或 `infrequent_if_exist`。

## 5. 特征工程函数修改了原始 DataFrame
- 现象：交叉验证第二折开始特征列重复，指标异常。
- 原因：函数内直接 `df[col] = ...` 原地修改了输入。
- 修复：函数开头先 `df = df.copy()`。
