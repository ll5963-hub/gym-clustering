"""Gym member segments: who should get the monthly offer?

k-means on a 70/30 split of gym_members.csv (scaling fit on training only),
training vs test elbow curves, segment profiles, a credit rule, and the
gym's monthly net from the $10 credit under the baseline vs v1.

Run:  pip install pandas scikit-learn matplotlib
      python fit.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

SEED = 42
K = 4
FEATURES = ["age", "visits_per_month", "avg_visit_minutes", "classes_per_month",
            "months_member", "monthly_extras_spend"]           # member_id left out

# ---------------------------------------------------------------- data
df = pd.read_csv("gym_members.csv")
print(f"gym_members.csv: {df.shape[0]} rows, {df.shape[1]} columns, "
      f"average extras spend ${df['monthly_extras_spend'].mean():.2f}")
train, test = train_test_split(df, test_size=0.3, random_state=SEED)
print(f"Train: {len(train)} members   Test: {len(test)} members")

scaler = StandardScaler().fit(train[FEATURES])
X_train = scaler.transform(train[FEATURES])
X_test = scaler.transform(test[FEATURES])

# ---------------------------------------------------------------- elbow: train vs test
models, train_spread, test_spread = {}, [], []
print("\nk   train spread   test spread")
for k in range(1, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X_train)
    centers = km.cluster_centers_[km.predict(X_test)]       # no refit on test
    t = float(((X_test - centers) ** 2).sum())
    models[k] = km
    train_spread.append(km.inertia_)
    test_spread.append(t)
    print(f"{k}   {km.inertia_:12.1f}   {t:11.1f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, vals, title in [(axes[0], train_spread, "Training members (231)"),
                        (axes[1], test_spread, "Test members (99)")]:
    ax.plot(range(1, 9), vals, marker="o", color="black")
    ax.axvline(K, color="gray", ls="--", lw=1)
    ax.set_title(title)
    ax.set_xlabel("Number of groups (k)")
    ax.set_ylabel("Within-group spread")
    ax.set_xticks(range(1, 9))
fig.tight_layout()
fig.savefig("elbow.png", dpi=200)
plt.close(fig)

print(f"\nTest spread: k=1 (baseline) {test_spread[0]:.1f}  vs  k={K} {test_spread[K-1]:.1f} "
      f"({1 - test_spread[K-1] / test_spread[0]:.0%} lower)")

# ---------------------------------------------------------------- v1 segments
v1 = models[K]
centers = pd.DataFrame(scaler.inverse_transform(v1.cluster_centers_), columns=FEATURES).round(1)
centers["train_members"] = np.bincount(v1.labels_, minlength=K)

# Name each cluster from its centroid (cluster numbers are arbitrary, so match by traits)
names = {}
for c, row in centers.iterrows():
    if row["visits_per_month"] < 6:
        names[c] = "Drifting newcomers"
    elif row["age"] >= 55:
        names[c] = "Loyal seniors"
    elif row["classes_per_month"] >= 4:
        names[c] = "Class-goers"
    else:
        names[c] = "Gym regulars"
centers["segment"] = centers.index.map(names)

# Credit goes only to Drifting newcomers: they average ~$9 on extras, so the credit
# pays off. Loyal seniors average ~$17 but many already spend $20+, so it roughly breaks
# even or loses money there; regulars and class-goers already spend well over $20.
CREDIT = {names[c]: names[c] == "Drifting newcomers" for c in centers.index}
centers["gets_credit"] = centers["segment"].map(CREDIT)
print("\n=== v1 centroids (training members, original units) ===")
print(centers.to_string())


def assign(member: dict):
    """Return (segment name, gets the $10 credit?) for one member."""
    x = scaler.transform(pd.DataFrame([member])[FEATURES])
    seg = names[int(v1.predict(x)[0])]
    return seg, CREDIT[seg]


new_member = dict(age=30, visits_per_month=4, avg_visit_minutes=42, classes_per_month=1,
                  months_member=5, monthly_extras_spend=10)
seg, credit = assign(new_member)
print(f"\nNew member -> {seg}, credit: {'yes' if credit else 'no'}")

# ---------------------------------------------------------------- value, member by member
df["segment"] = [names[c] for c in v1.predict(scaler.transform(df[FEATURES]))]
df["credit_value"] = np.where(df["monthly_extras_spend"] < 20, 5, -10)  # +$5 or -$10 per member

print("\n=== All 330 members by segment ===")
summary = df.groupby("segment").agg(
    members=("member_id", "count"),
    avg_extras_spend=("monthly_extras_spend", "mean"),
    net_if_credited=("credit_value", "sum"))
summary["gets_credit"] = summary.index.map(CREDIT)
print(summary.round(1).to_string())

baseline_net = df["credit_value"].sum()
v1_net = df.loc[df["segment"].map(CREDIT), "credit_value"].sum()
n_credited = int(df["segment"].map(CREDIT).sum())
print(f"\nBaseline (credit to all {len(df)} members): ${baseline_net:,} per month")
print(f"v1 (credit to {n_credited} members in chosen segments): ${v1_net:,} per month")
print(f"Difference: ${v1_net - baseline_net:,} per month, ${12 * (v1_net - baseline_net):,} per year")
