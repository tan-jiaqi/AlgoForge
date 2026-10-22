"""生成模拟的电信客服工单文本分类数据集（跨场景迁移用，固定随机种子可复现）。

每条工单由"类别模板 + 随机槽位 + 口语噪声"组成，类别分布不均衡。
为贴近真实工单，约 25% 的工单混入其他类别的表述（例如"网速慢，想换套餐"），
另有约 6% 的标签噪声（坐席分错类），使任务不至于被模板轻易记住。

用法：python data/scripts/make_tickets.py [--rows 2000] [--out data/tickets.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

TEMPLATES: dict[str, list[str]] = {
    "网络故障": [
        "{place}的宽带{time}断网了，路由器灯一直闪",
        "手机{time}没有信号，打不出电话也上不了网",
        "光猫显示LOS红灯，{place}整栋楼都上不了网",
        "网速特别慢，测速只有{num}兆，看视频一直卡",
        "WiFi老是掉线，{time}已经掉了{num}次",
    ],
    "账单争议": [
        "这个月话费多扣了{num}元，我没有订过这个业务",
        "账单里有一笔{num}元的流量费，我根本没用那么多",
        "已经销户了为什么还在扣月租",
        "充值{num}元没有到账，请核实",
        "发票金额和实际扣费对不上，差了{num}元",
    ],
    "套餐变更": [
        "想把现在的套餐升级成{num}元的5G套餐",
        "流量不够用，有没有更划算的套餐推荐",
        "我要降档，现在{num}元的套餐太贵了",
        "家里想办融合套餐，宽带和手机一起",
        "合约{time}到期，想了解续约优惠",
    ],
    "销户投诉": [
        "服务太差了，我要销户，{time}就办",
        "多次报修没人处理，再不解决我就携号转网",
        "营业厅态度恶劣，我要投诉{place}的工作人员",
        "宽带合约到期不续了，帮我办理拆机",
        "对你们的处理结果不满意，要求升级投诉",
    ],
}
WEIGHTS = {"网络故障": 0.40, "账单争议": 0.25, "套餐变更": 0.22, "销户投诉": 0.13}
SLOTS = {
    "place": ["小区", "公司", "家里", "宿舍", "店里", "老家"],
    "time": ["今天", "昨晚", "这两天", "一直", "上周", "刚才"],
}
NOISE = ["", "", "", "！", "，麻烦尽快处理", "，急！", "。谢谢", "，都打了好几次客服了"]


def make_tickets(rows: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(WEIGHTS)
    labels = rng.choice(names, rows, p=list(WEIGHTS.values()))

    def fill(label: str) -> str:
        return rng.choice(TEMPLATES[label]).format(
            place=rng.choice(SLOTS["place"]), time=rng.choice(SLOTS["time"]), num=int(rng.integers(2, 300))
        )

    texts = []
    for label in labels:
        text = fill(label)
        if rng.random() < 0.25:  # 混入其他类别的表述，主诉仍是本类
            other = rng.choice([n for n in names if n != label])
            parts = [text, fill(other)]
            text = "，".join(parts if rng.random() < 0.6 else parts[::-1])
        texts.append(text + rng.choice(NOISE))
    flip = rng.random(rows) < 0.06  # 标签噪声
    labels = np.where(flip, rng.choice(names, rows), labels)
    return pd.DataFrame({"ticket_id": [f"T{200000 + i}" for i in range(rows)], "text": texts, "label": labels})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "tickets.csv")
    args = parser.parse_args()
    df = make_tickets(args.rows, args.seed)
    df.to_csv(args.out, index=False)
    print(f"写入 {args.out}：{len(df)} 行，类别分布 {df['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
