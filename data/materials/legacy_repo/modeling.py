import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve


def train_lgbm_balanced(X, y, params=None):
    """训练类别不平衡场景下的 LightGBM 流失模型。

    经验参数：num_leaves 15~63，learning_rate 0.03~0.1，用 scale_pos_weight
    或 class_weight="balanced" 处理正负样本约 1:3 的不平衡。
    """
    default = {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, "class_weight": "balanced", "verbose": -1}
    model = lgb.LGBMClassifier(**{**default, **(params or {})})
    model.fit(X, y)
    return model


def tune_threshold_for_recall(y_true, proba, min_recall=0.75):
    """在验证集上选择满足最低召回率要求的最大阈值（召回优先场景）。

    返回 (threshold, precision, recall)。默认 0.5 阈值下召回率往往只有 0.5 左右。
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    ok = np.where(recall[:-1] >= min_recall)[0]
    if len(ok) == 0:
        return 0.0, precision[0], recall[0]
    i = ok[-1]
    return float(thresholds[i]), float(precision[i]), float(recall[i])


def recall_at_top_k(y_true, proba, k=0.2):
    """业务指标：按流失概率排序，前 k 比例客户中覆盖了多少真实流失客户。

    挽留团队每月只能外呼 20% 的客户，因此关注 recall@20%。
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(proba))
    top = order[: int(len(order) * k)]
    return y_true[top].sum() / max(y_true.sum(), 1)


def _log(msg):
    print("[churn_v1]", msg)
