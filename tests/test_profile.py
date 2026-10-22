from factory.profile import profile


def test_churn_profile_tags(churn_csv):
    p = profile(churn_csv)
    roles = {c.name: c.role for c in p.columns}
    assert p.target == "churn"
    assert roles["total_charges"] == "string_numeric"
    assert roles["customer_id"] == "id"
    assert roles["contract"] == "categorical"
    assert {"has_missing_values", "string_typed_numeric", "id_columns", "has_categorical_features", "time_tenure"} <= set(p.tags)
    assert p.modality == "tabular"


def test_text_profile(tmp_path, text_df):
    path = tmp_path / "t.csv"
    text_df.assign(text=[f"{t}，工单号{i}，请尽快处理" for i, t in enumerate(text_df["text"])]).to_csv(path, index=False)
    p = profile(path, "label")
    assert p.modality == "text" and p.text_column == "text"
    assert len(p.class_counts) == 3
