"""
Banking Operations & Transformation Intelligence
CFPB Consumer Complaint Database | 2021-2025

I use this script to turn public financial-services complaint data into
operational insights and transformation priorities.

The analysis is deliberately business-led: I first understand where
customer pain is concentrated, then I quantify operational signals, and
finally I translate the findings into areas that deserve further investigation.

I do not treat complaint volume as a direct measure of poor performance.
I use it as a signal that should be interpreted in context, especially when
comparing financial institutions of different sizes.
"""

from __future__ import annotations

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# %%
# 01. Project setup
#
# I keep the raw CFPB file untouched and write all derived outputs to a
# separate folder. This makes my analysis reproducible and easy to review.

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_FILE = PROJECT_ROOT / "complaints.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

START_DATE = pd.Timestamp("2021-01-01")
END_DATE = pd.Timestamp("2025-12-31")
CHUNK_SIZE = 250_000

USE_COLS = [
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Company public response",
    "Company",
    "State",
    "Submitted via",
    "Date sent to company",
    "Company response to consumer",
    "Timely response?",
    "Complaint ID",
]

# %%
# 02. Business context
#
# I approach the dataset from the perspective of a financial-services
# transformation consultant. My objective is not to rank companies, but to
# identify recurring customer pain points and operational signals that could
# justify further process investigation or transformation.

print("\nBanking Operations & Transformation Intelligence")
print("CFPB Consumer Complaint Database | 2021-2025")
print("=" * 60)

if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"I could not find the raw dataset at: {RAW_FILE}\n"
        "Place complaints.csv next to this script before running it."
    )

# %%
# 03. Data preparation helpers
#
# I standardise labels before aggregating the data. This is particularly
# important here because CFPB product taxonomy changed over time. Without
# harmonisation, I could mistake a taxonomy change for a business trend.

PRODUCT_MAP = {
    "Credit reporting": "Credit Reporting",
    "Credit reporting or other personal consumer reports": "Credit Reporting",
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit Reporting",
    "Debt collection": "Debt Collection",
    "Checking or savings account": "Checking & Savings",
    "Credit card": "Credit Card",
    "Credit card or prepaid card": "Credit Card & Prepaid",
    "Prepaid card": "Credit Card & Prepaid",
    "Mortgage": "Mortgage",
    "Vehicle loan or lease": "Vehicle Loan / Lease",
    "Student loan": "Student Loan",
    "Payday loan, title loan, or personal loan": "Payday / Personal Loan",
    "Payday loan, title loan, personal loan, or advance loan": "Payday / Personal Loan",
    "Money transfer, virtual currency, or money service": "Money Transfer / Payments",
    "Debt or credit management": "Debt / Credit Management",
}

RENAME_MAP = {
    "Date received": "date_received",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Company public response": "company_public_response",
    "Company": "company",
    "State": "state",
    "Submitted via": "submitted_via",
    "Date sent to company": "date_sent_to_company",
    "Company response to consumer": "company_response",
    "Timely response?": "timely_response",
    "Complaint ID": "complaint_id",
}


def clean_text(value: pd.Series) -> pd.Series:
    """I normalise text fields while keeping missing values explicit."""
    return value.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA})


def normalise_product(value: pd.Series) -> pd.Series:
    mapped = value.map(PRODUCT_MAP)
    return mapped.fillna(value)


def parse_dates(frame: pd.DataFrame) -> pd.DataFrame:
    frame["date_received"] = pd.to_datetime(frame["date_received"], errors="coerce")
    frame["date_sent_to_company"] = pd.to_datetime(
        frame["date_sent_to_company"], errors="coerce"
    )
    return frame


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame["complaint_year"] = frame["date_received"].dt.year.astype("Int64")
    frame["complaint_month"] = frame["date_received"].dt.to_period("M").astype("string")
    frame["response_delay_days"] = (
        frame["date_sent_to_company"] - frame["date_received"]
    ).dt.days
    frame["response_delay_days"] = frame["response_delay_days"].clip(lower=0)
    frame["is_timely"] = frame["timely_response"].eq("Yes")
    frame["is_resolved"] = frame["company_response"].fillna("").str.startswith("Closed")
    frame["has_sub_issue"] = frame["sub_issue"].notna()
    return frame

# %%
# 04. Chunked data loading
#
# I process the file in chunks because the raw CSV can be several gigabytes.
# This keeps the workflow reproducible on a normal laptop instead of requiring
# a large-memory environment.

required_columns = None
monthly_parts: list[pd.DataFrame] = []
product_parts: list[pd.DataFrame] = []
issue_parts: list[pd.DataFrame] = []
channel_parts: list[pd.DataFrame] = []
company_parts: list[pd.DataFrame] = []
quality_parts: list[dict] = []

processed_rows = 0
filtered_rows = 0
unique_ids: set[str] = set()

reader = pd.read_csv(
    RAW_FILE,
    usecols=USE_COLS,
    chunksize=CHUNK_SIZE,
    low_memory=False,
)

for chunk_number, chunk in enumerate(reader, start=1):
    processed_rows += len(chunk)
    chunk = chunk.rename(columns=RENAME_MAP)

    for column in [
        "product",
        "sub_product",
        "issue",
        "sub_issue",
        "company",
        "state",
        "submitted_via",
        "company_response",
        "timely_response",
    ]:
        chunk[column] = clean_text(chunk[column])

    chunk = parse_dates(chunk)
    chunk = chunk.loc[
        chunk["date_received"].between(START_DATE, END_DATE, inclusive="both")
    ].copy()

    if chunk.empty:
        continue

    filtered_rows += len(chunk)
    chunk["product"] = normalise_product(chunk["product"])
    chunk = add_features(chunk)

    ids = chunk["complaint_id"].astype("string")
    unique_ids.update(ids.dropna().tolist())

    quality_parts.append(
        {
            "rows": len(chunk),
            "missing_product": int(chunk["product"].isna().sum()),
            "missing_issue": int(chunk["issue"].isna().sum()),
            "missing_sub_issue": int(chunk["sub_issue"].isna().sum()),
            "missing_company": int(chunk["company"].isna().sum()),
            "missing_timely_response": int(chunk["timely_response"].isna().sum()),
            "negative_response_delay": int((chunk["response_delay_days"] < 0).sum()),
        }
    )

    monthly_parts.append(
        chunk.groupby("complaint_month", dropna=False).agg(
            complaints=("complaint_id", "size"),
            timely_response_rate=("is_timely", "mean"),
            average_response_delay_days=("response_delay_days", "mean"),
            resolved_rate=("is_resolved", "mean"),
        ).reset_index()
    )

    product_parts.append(
        chunk.groupby("product", dropna=False).agg(
            complaints=("complaint_id", "size"),
            timely_response_rate=("is_timely", "mean"),
            average_response_delay_days=("response_delay_days", "mean"),
            resolved_rate=("is_resolved", "mean"),
        ).reset_index()
    )

    issue_parts.append(
        chunk.groupby(["product", "issue"], dropna=False).agg(
            complaints=("complaint_id", "size"),
            timely_response_rate=("is_timely", "mean"),
            average_response_delay_days=("response_delay_days", "mean"),
            resolved_rate=("is_resolved", "mean"),
        ).reset_index()
    )

    channel_parts.append(
        chunk.groupby("submitted_via", dropna=False).agg(
            complaints=("complaint_id", "size"),
            timely_response_rate=("is_timely", "mean"),
            average_response_delay_days=("response_delay_days", "mean"),
        ).reset_index()
    )

    company_parts.append(
        chunk.groupby("company", dropna=False).agg(
            complaints=("complaint_id", "size"),
            timely_response_rate=("is_timely", "mean"),
            average_response_delay_days=("response_delay_days", "mean"),
        ).reset_index()
    )

    if chunk_number % 10 == 0:
        print(f"Processed {processed_rows:,} raw rows...", flush=True)

# %%
# 05. Reconcile chunk-level aggregates
#
# I now combine the chunk-level summaries. I aggregate again because the same
# product or issue can appear in multiple chunks.

def reconcile(parts: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    combined = pd.concat(parts, ignore_index=True)
    if "complaints" not in combined.columns:
        return combined

    weighted = combined.copy()
    weighted["timely_numerator"] = weighted["complaints"] * weighted["timely_response_rate"]
    weighted["delay_numerator"] = weighted["complaints"] * weighted["average_response_delay_days"]
    weighted["resolved_numerator"] = weighted["complaints"] * weighted.get("resolved_rate", 0)

    aggregations = {
        "complaints": ("complaints", "sum"),
        "timely_numerator": ("timely_numerator", "sum"),
        "delay_numerator": ("delay_numerator", "sum"),
    }
    if "resolved_rate" in weighted.columns:
        aggregations["resolved_numerator"] = ("resolved_numerator", "sum")
    result = weighted.groupby(keys, dropna=False).agg(**aggregations).reset_index()
    result["timely_response_rate"] = result["timely_numerator"] / result["complaints"]
    result["average_response_delay_days"] = result["delay_numerator"] / result["complaints"]
    if "resolved_numerator" in result.columns:
        result["resolved_rate"] = result["resolved_numerator"] / result["complaints"]

    return result.drop(columns=["timely_numerator", "delay_numerator", "resolved_numerator"], errors="ignore")

monthly = reconcile(monthly_parts, ["complaint_month"])
product_summary = reconcile(product_parts, ["product"])
issue_summary = reconcile(issue_parts, ["product", "issue"])
channel_summary = reconcile(channel_parts, ["submitted_via"])
company_summary = reconcile(company_parts, ["company"])

# %%
# 06. Transformation opportunity scoring
#
# I use the score as a practical prioritisation framework rather than as a
# statistical measure of customer or financial impact.
#
# I deliberately avoid giving complaint volume and global complaint share
# separate weights because they are strongly related. Instead, I combine:
#   1. complaint volume
#   2. untimely-response rate
#   3. average response delay
#   4. issue concentration within the product
#
# This gives me a more balanced view of scale, operational friction and
# how concentrated a specific issue is within its product.

issue_summary["untimely_response_rate"] = 1 - issue_summary["timely_response_rate"]

# I calculate issue concentration within each product. This tells me whether
# a particular issue represents a meaningful share of complaints for that
# product, rather than simply rewarding issues with large absolute volumes.
product_totals = issue_summary.groupby("product", dropna=False)["complaints"].transform("sum")
issue_summary["issue_share_within_product"] = (
    issue_summary["complaints"] / product_totals.replace(0, np.nan)
)

# I focus the priority matrix on meaningful issue volumes to avoid allowing
# very small categories to dominate because of noisy rates.
MIN_ISSUE_VOLUME = 500
priority = issue_summary.loc[issue_summary["complaints"] >= MIN_ISSUE_VOLUME].copy()


def percentile_score(series: pd.Series) -> pd.Series:
    if series.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=series.index)
    return series.rank(pct=True, method="average")


priority["volume_score"] = percentile_score(priority["complaints"])
priority["untimely_score"] = percentile_score(priority["untimely_response_rate"])
priority["delay_score"] = percentile_score(priority["average_response_delay_days"].fillna(0))
priority["concentration_score"] = percentile_score(
    priority["issue_share_within_product"].fillna(0)
)

# I give equal weight to scale and operational friction, while keeping
# concentration as a supporting signal.
priority["transformation_priority_score"] = (
    0.35 * priority["volume_score"]
    + 0.35 * priority["untimely_score"]
    + 0.20 * priority["delay_score"]
    + 0.10 * priority["concentration_score"]
)

priority["priority_rank"] = priority["transformation_priority_score"].rank(
    ascending=False, method="first"
).astype(int)

priority["priority_band"] = pd.cut(
    priority["transformation_priority_score"],
    bins=[-np.inf, 0.40, 0.70, np.inf],
    labels=["Monitor", "Investigate", "Prioritise"],
)

# I also keep global complaint share as a descriptive KPI for the visual analysis.
# I do not use it as a second volume weight in the priority score.
priority["complaint_share"] = priority["complaints"] / issue_summary["complaints"].sum()

priority = priority.sort_values("transformation_priority_score", ascending=False)

# %%
# 07. Save analytical outputs
#
# I save compact analytical tables rather than exporting another copy of the
# 5+ GB raw dataset. This keeps the repository light and the workflow easy to
# reproduce from the original CFPB source.

monthly.to_csv(OUTPUT_DIR / "monthly_operational_kpis.csv", index=False)
product_summary.to_csv(OUTPUT_DIR / "product_summary.csv", index=False)
issue_summary.to_csv(OUTPUT_DIR / "issue_summary.csv", index=False)
channel_summary.to_csv(OUTPUT_DIR / "channel_summary.csv", index=False)
priority.to_csv(OUTPUT_DIR / "transformation_priority.csv", index=False)

quality = pd.DataFrame(quality_parts)
quality_summary = pd.DataFrame(
    {
        "metric": [
            "Raw rows processed",
            "Rows in selected period",
            "Unique complaint IDs",
            "Duplicate complaint IDs",
            "Missing product values",
            "Missing issue values",
            "Missing sub-issue values",
            "Missing company values",
            "Missing timely-response values",
            "Negative response delays",
        ],
        "value": [
            processed_rows,
            filtered_rows,
            len(unique_ids),
            filtered_rows - len(unique_ids),
            int(quality["missing_product"].sum()),
            int(quality["missing_issue"].sum()),
            int(quality["missing_sub_issue"].sum()),
            int(quality["missing_company"].sum()),
            int(quality["missing_timely_response"].sum()),
            int(quality["negative_response_delay"].sum()),
        ],
    }
)
quality_summary.to_csv(OUTPUT_DIR / "data_quality_summary.csv", index=False)

# %%
# 08. Visual outputs
#
# I keep the visual layer focused on a small number of decision-useful charts.
# The goal is to support the consulting story, not to create a catalogue of
# every possible chart.

plt.rcParams.update({"figure.figsize": (10, 6), "axes.titlesize": 14})

# Top products
plot_data = product_summary.nlargest(10, "complaints").sort_values("complaints")
fig, ax = plt.subplots()
ax.barh(plot_data["product"], plot_data["complaints"])
ax.set_title("Top Financial Product Categories by Complaint Volume")
ax.set_xlabel("Number of complaints")
ax.set_ylabel("")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "top_products.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# Monthly trend
plot_data = monthly.sort_values("complaint_month")
fig, ax = plt.subplots()
ax.plot(plot_data["complaint_month"], plot_data["complaints"], linewidth=1.8)
ax.set_title("Monthly Complaint Volume")
ax.set_xlabel("Month")
ax.set_ylabel("Complaints")
ax.tick_params(axis="x", rotation=60)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "monthly_complaint_trend.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# Priority matrix
plot_data = priority.head(30).copy()
fig, ax = plt.subplots()
ax.scatter(
    plot_data["complaint_share"],
    plot_data["untimely_response_rate"],
    s=np.sqrt(plot_data["complaints"]) * 8,
    alpha=0.65,
)
for _, row in plot_data.head(10).iterrows():
    label = f"{row['product']} | {row['issue']}"
    label = re.sub(r"\s+", " ", label)
    if len(label) > 65:
        label = label[:62] + "..."
    ax.annotate(label, (row["complaint_share"], row["untimely_response_rate"]), fontsize=7)
ax.set_title("Transformation Opportunity Map")
ax.set_xlabel("Share of complaints")
ax.set_ylabel("Untimely-response rate")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "transformation_priority_matrix.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# %%
# 09. Executive summary
#
# I finish the run with a concise summary that I can use to guide the next
# stage of my notebook analysis. I intentionally keep the wording factual and
# avoid presenting correlation as causation.

timely_rate = 1 - (issue_summary["complaints"] * (1 - issue_summary["timely_response_rate"])).sum() / issue_summary["complaints"].sum()

executive_summary = pd.DataFrame(
    {
        "metric": [
            "Analysis period",
            "Complaints analysed",
            "Unique complaint IDs",
            "Timely response rate",
            "Product categories after harmonisation",
            "Issue categories",
            "Minimum issue volume used for priority scoring",
        ],
        "value": [
            f"{START_DATE.date()} to {END_DATE.date()}",
            filtered_rows,
            len(unique_ids),
            timely_rate,
            product_summary["product"].nunique(dropna=True),
            issue_summary["issue"].nunique(dropna=True),
            MIN_ISSUE_VOLUME,
        ],
    }
)
executive_summary.to_csv(OUTPUT_DIR / "executive_summary.csv", index=False)

print("\nAnalysis complete.")
print(f"Rows analysed: {filtered_rows:,}")
print(f"Unique complaint IDs: {len(unique_ids):,}")
print(f"Timely response rate: {timely_rate:.2%}")
print(f"Outputs written to: {OUTPUT_DIR}")
print("\nTop transformation priorities:")
print(
    priority[["priority_rank", "product", "issue", "complaints", "transformation_priority_score", "priority_band"]]
    .head(10)
    .to_string(index=False)
)
