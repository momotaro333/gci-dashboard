import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("/Users/tanaamidaisuke/Documents/tokyo_univ/バイト/松尾研/GCIグローバル/効果測定/データ")
OUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------
# 1. 申込一覧の読み込みと学校名の正規化
# ---------------------------------------------------------------
mo = pd.read_csv(
    DATA_DIR / "申込一覧.csv",
    usecols=["account_name", "country", "学校名", "申込ステータス", "applicant_type_ja"],
)
sch = pd.read_csv(DATA_DIR / "大学名テーブルのコピー.csv")

# 学校名テーブルから 小文字キー -> 正式名 の辞書を作成
school_dict = {}
for _, row in sch.iterrows():
    if pd.notna(row["school_name"]):
        key = str(row["school_name"]).lower().strip()
        if key not in school_dict:
            school_dict[key] = row["school_name_canonical"]

# school_name が NaN の行を残すと、pandas の merge は NaN 同士を「一致」とみなし
# 学校名未入力の申込者が誤って学校名テーブルの空欄行（canonical="Others" 等）に
# 紐付いてしまう。ここでは学校名が入力されている行だけを対象にする。
exact_map = sch[sch["school_name"].notna()][["school_name", "school_name_canonical"]].drop_duplicates(
    subset=["school_name"], keep="first"
)
mo = mo.merge(exact_map, left_on="学校名", right_on="school_name", how="left", validate="many_to_one")


def repair(row):
    if pd.notna(row["school_name_canonical"]):
        return row["school_name_canonical"]
    if pd.notna(row["学校名"]):
        return school_dict.get(str(row["学校名"]).lower().strip())
    return None


mo["school_name_canonical"] = mo.apply(repair, axis=1)
mo["school_final"] = mo["school_name_canonical"].fillna(mo["学校名"]).fillna("(未回答)")
mo["country_final"] = mo["country"].fillna("(未回答)")

print("学校名一致率:", mo["school_name_canonical"].notna().mean())

# 申込み全体（ステータス別）集計用
apps_all = mo.copy()

# 出席率・提出率の母集団は「招待済」アカウント
accepted = mo[mo["申込ステータス"] == "招待済"].copy()
# account_name 重複は最初の1件を採用
# account_name が NaN の行は「同一アカウント」として束ねられてしまう
# （pandas の drop_duplicates は NaN 同士を重複とみなす）ため、集計対象から除外する。
accounts = accepted[accepted["account_name"].notna()].drop_duplicates(subset=["account_name"], keep="first")[
    ["account_name", "country_final", "school_final"]
].reset_index(drop=True)
print("accepted accounts:", len(accounts))

# ---------------------------------------------------------------
# 2. アンケート回答（出席率）
# ---------------------------------------------------------------
q = pd.read_csv(
    DATA_DIR / "アンケート回答.csv",
    usecols=["account_name", "questionnaire_title"],
)
q["session_num"] = q["questionnaire_title"].str.extract(r"Session(\d+)").astype(float)
q = q.dropna(subset=["session_num"])
q["session_num"] = q["session_num"].astype(int)
q = q.drop_duplicates(subset=["account_name", "session_num"])
q = q.merge(accounts, on="account_name", how="inner")  # 招待済アカウントのみ対象

SURVEY_SESSIONS = sorted(q["session_num"].unique().tolist())
print("survey sessions:", SURVEY_SESSIONS)

# ---------------------------------------------------------------
# 3. 課題提出（提出率） 本課題 + 遅延提出 を統合
# ---------------------------------------------------------------
kadai = pd.read_csv(
    DATA_DIR / "課題一覧取得_2026_09_17.csv",
    usecols=["account_name", "assignment_name", "is_competition", "submitted", "score"],
)
kadai = kadai[kadai["is_competition"] == 0].copy()
kadai["session_num"] = kadai["assignment_name"].str.extract(r"for Session(\d+)").astype(float)
kadai = kadai.dropna(subset=["session_num"])
kadai["session_num"] = kadai["session_num"].astype(int)
kadai["submitted"] = pd.to_numeric(kadai["submitted"], errors="coerce").fillna(0).astype(int)

# 本提出・遅延提出のどちらかで submitted=1 なら提出済みとして統合
kadai_sub = (
    kadai[kadai["submitted"] == 1]
    .groupby(["account_name", "session_num"], as_index=False)
    .agg(score=("score", "max"))
)
kadai_sub = kadai_sub.merge(accounts, on="account_name", how="inner")

HW_SESSIONS = sorted(kadai_sub["session_num"].unique().tolist())
print("hw sessions:", HW_SESSIONS)


# ---------------------------------------------------------------
# 4. 集計関数
# ---------------------------------------------------------------
def session_breakdown(target_accounts_df, submitted_df, sessions):
    """target_accounts_df: account_name を含む対象アカウントの df
    submitted_df: account_name, session_num を含む提出済みレコードの df (対象範囲で事前フィルタ済み)
    """
    target_n = target_accounts_df["account_name"].nunique()
    result = []
    for s in sessions:
        submitted_n = submitted_df.loc[submitted_df["session_num"] == s, "account_name"].nunique()
        rate = submitted_n / target_n if target_n else 0.0
        result.append({"session": s, "target": int(target_n), "submitted": int(submitted_n), "rate": round(rate, 4)})
    return result, target_n


def avg_rate(breakdown):
    if not breakdown:
        return 0.0
    return round(float(np.mean([b["rate"] for b in breakdown])), 4)


def build_scope(accounts_scope, survey_scope, hw_scope):
    survey_bd, target_n = session_breakdown(accounts_scope, survey_scope, SURVEY_SESSIONS)
    hw_bd, _ = session_breakdown(accounts_scope, hw_scope, HW_SESSIONS)
    return {
        "applicants": int(target_n),
        "avg_attendance_rate": avg_rate(survey_bd),
        "avg_submission_rate": avg_rate(hw_bd),
        "survey": survey_bd,
        "hw": hw_bd,
    }


# ---------------------------------------------------------------
# 5. 申込みステータス集計（国別・大学別）
# ---------------------------------------------------------------
def status_counts(df):
    total = len(df)
    accepted_n = int((df["申込ステータス"] == "招待済").sum())
    rejected_n = int((df["申込ステータス"] == "却下").sum())
    pending_n = int((df["申込ステータス"] == "審査待").sum())
    return {
        "total_applications": int(total),
        "accepted": accepted_n,
        "rejected": rejected_n,
        "pending": pending_n,
    }


# ---------------------------------------------------------------
# 6. 全体・国別・大学別に集計してJSON化
# ---------------------------------------------------------------
overall = build_scope(accounts, q, kadai_sub)
overall.update(status_counts(apps_all))
overall["countries"] = int(accounts["country_final"].nunique())
overall["schools"] = int(accounts["school_final"].nunique())

countries_out = []
for country, acc_c in accounts.groupby("country_final"):
    q_c = q[q["country_final"] == country]
    kadai_c = kadai_sub[kadai_sub["country_final"] == country]
    apps_c = apps_all[apps_all["country_final"] == country]

    country_node = build_scope(acc_c, q_c, kadai_c)
    country_node.update(status_counts(apps_c))
    country_node["country"] = country
    country_node["schools"] = int(acc_c["school_final"].nunique())

    universities = []
    for school, acc_s in acc_c.groupby("school_final"):
        q_s = q_c[q_c["school_final"] == school]
        kadai_s = kadai_c[kadai_c["school_final"] == school]
        apps_s = apps_c[apps_c["学校名"].notna() & (apps_c["school_final"] == school)]

        school_node = build_scope(acc_s, q_s, kadai_s)
        school_node.update(status_counts(apps_s))
        school_node["school"] = school
        universities.append(school_node)

    universities.sort(key=lambda x: (-x["applicants"], x["school"]))
    country_node["universities"] = universities
    countries_out.append(country_node)

countries_out.sort(key=lambda x: (-x["applicants"], x["country"]))

data = {
    "meta": {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "survey_sessions": SURVEY_SESSIONS,
        "hw_sessions": HW_SESSIONS,
        "school_match_rate": round(float(mo["school_name_canonical"].notna().mean()), 4),
    },
    "overall": overall,
    "countries": countries_out,
}

out_path = OUT_DIR / "dashboard_data.json"
out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
print("wrote", out_path, out_path.stat().st_size / 1024 / 1024, "MB")
