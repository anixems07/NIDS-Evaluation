# align_final.py — FINAL: chunked + memory-safe + all label fixes included.
# Creates the aligned 2018 test set with BENIGN present.
# Run from project root:  nohup python -u src/align_final.py > align_final.log 2>&1 &
import glob
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

RAW_2018, OUT = "data/raw_2018", "data/processed"

RENAME = {
    "Tot Fwd Pkts": "Total Fwd Packets",            "Tot Bwd Pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets", "TotLen Bwd Pkts": "Total Length of Bwd Packets",
    "Fwd Pkt Len Max": "Fwd Packet Length Max",     "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",   "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Bwd Pkt Len Max": "Bwd Packet Length Max",     "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",   "Bwd Pkt Len Std": "Bwd Packet Length Std",
    "Flow Byts/s": "Flow Bytes/s",                  "Flow Pkts/s": "Flow Packets/s",
    "Fwd IAT Tot": "Fwd IAT Total",                 "Bwd IAT Tot": "Bwd IAT Total",
    "Fwd Header Len": "Fwd Header Length",          "Bwd Header Len": "Bwd Header Length",
    "Fwd Pkts/s": "Fwd Packets/s",                  "Bwd Pkts/s": "Bwd Packets/s",
    "Pkt Len Min": "Min Packet Length",             "Pkt Len Max": "Max Packet Length",
    "Pkt Len Mean": "Packet Length Mean",           "Pkt Len Std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance",
    "FIN Flag Cnt": "FIN Flag Count", "SYN Flag Cnt": "SYN Flag Count",
    "RST Flag Cnt": "RST Flag Count", "PSH Flag Cnt": "PSH Flag Count",
    "ACK Flag Cnt": "ACK Flag Count", "URG Flag Cnt": "URG Flag Count",
    "CWE Flag Cnt": "CWE Flag Count", "ECE Flag Cnt": "ECE Flag Count",
    "Pkt Size Avg": "Average Packet Size",
    "Fwd Seg Size Avg": "Avg Fwd Segment Size",     "Bwd Seg Size Avg": "Avg Bwd Segment Size",
    "Fwd Byts/b Avg": "Fwd Avg Bytes/Bulk",         "Fwd Pkts/b Avg": "Fwd Avg Packets/Bulk",
    "Fwd Blk Rate Avg": "Fwd Avg Bulk Rate",        "Bwd Byts/b Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Pkts/b Avg": "Bwd Avg Packets/Bulk",       "Bwd Blk Rate Avg": "Bwd Avg Bulk Rate",
    "Subflow Fwd Byts": "Subflow Fwd Bytes",        "Subflow Fwd Pkts": "Subflow Fwd Packets",
    "Subflow Bwd Byts": "Subflow Bwd Bytes",        "Subflow Bwd Pkts": "Subflow Bwd Packets",
    "Init Fwd Win Byts": "Init_Win_bytes_forward",  "Init Bwd Win Byts": "Init_Win_bytes_backward",
    "seg_sz_min": "min_seg_size_forward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",        # your files' naming
    "Fwd Seg Size Min":  "min_seg_size_forward",    # your files' naming
}

LABEL_MAP = {
    "BENIGN": "BENIGN", "Benign": "BENIGN",        # ← THE FIX: this packaging's casing
    "Bot": "Bot",
    "DDOS attack-HOIC": "DDoS", "DDOS attack-LOIC-HTTP": "DDoS", "DDOS attack-LOIC-UDP": "DDoS",
    "DDoS attacks-HOIC": "DDoS", "DDoS attacks-LOIC-HTTP": "DDoS", "DDoS attacks-LOIC-UDP": "DDoS",
    "DoS attacks-Hulk": "DoS Hulk", "DoS attacks-GoldenEye": "DoS GoldenEye",
    "DoS attacks-Slowloris": "DoS slowloris", "DoS attacks-SlowHTTPTest": "DoS Slowhttptest",
    "FTP-BruteForce": "FTP-Patator", "SSH-Bruteforce": "SSH-Patator",
    "FTP-Cracking": "FTP-Patator", "SSH-Cracking": "SSH-Patator",
    "Brute Force -Web": "Web Attack", "Brute Force -XSS": "Web Attack", "SQL Injection": "Web Attack",
    "Infilteration": "Infiltration",
}
EXCLUDE = ["Infiltration"]

# case-insensitive matching layer: any casing variant maps automatically
LABEL_LOOKUP = {k.lower(): v for k, v in LABEL_MAP.items()}

features = pd.read_csv("results/tables/final_feature_list.csv")["feature"].tolist()
mapping  = pd.read_csv("results/tables/label_mapping.csv")
name_to_code = dict(zip(mapping["label_name"], mapping["code"]))

needed_raw = set(features) | set(RENAME.keys()) | {"Label"}
CHUNK, BENIGN_PER_CHUNK = 300_000, 6_000

frames = []
for f in sorted(glob.glob(f"{RAW_2018}/*.csv")):
    name, parts, unmapped, excluded_rows = Path(f).name, [], Counter(), 0
    for chunk in pd.read_csv(f, encoding="latin1", low_memory=False,
                             usecols=lambda c: c.strip() in needed_raw,
                             chunksize=CHUNK):
        chunk.columns = [c.strip() for c in chunk.columns]
        chunk = chunk.rename(columns=RENAME)

        lab_raw = chunk["Label"].astype(str).str.strip()
        mapped  = lab_raw.str.lower().map(LABEL_LOOKUP)          # ← THE FIX (case-insensitive)
        unmapped.update(lab_raw[mapped.isna()].value_counts().to_dict())
        excl = mapped.isin(EXCLUDE)
        excluded_rows += int(excl.sum())
        keep = mapped.notna() & ~excl
        chunk, mapped = chunk[keep].copy(), mapped[keep]

        for c in features:
            s = pd.to_numeric(chunk[c], errors="coerce")
            chunk[c] = s.replace([np.inf, -np.inf], np.nan)
        chunk["Label"] = mapped
        chunk = chunk.dropna(subset=features)

        is_b = chunk["Label"] == "BENIGN"
        attacks  = chunk[~is_b]
        benign_s = chunk[is_b].sample(n=min(BENIGN_PER_CHUNK, int(is_b.sum())), random_state=42)
        parts.append(pd.concat([attacks, benign_s], ignore_index=True))

    df = pd.concat(parts, ignore_index=True)
    frames.append(df)
    print(f"{name}: kept {len(df)} rows | Infiltration excluded: {excluded_rows}", flush=True)
    if unmapped:
        print(f"  !! UNMAPPED labels dropped: {dict(unmapped)}", flush=True)

test = pd.concat(frames, ignore_index=True)
attack = test[test["Label"] != "BENIGN"]
benign = test[test["Label"] == "BENIGN"].sample(n=min(400_000, (test["Label"] == "BENIGN").sum()),
                                                random_state=42)
test = pd.concat([attack, benign], ignore_index=True)

X18 = test[features]
y18 = test["Label"].map(name_to_code)
X18.to_csv(f"{OUT}/X_test_2018_aligned.csv", index=False)
y18.rename("label").to_csv(f"{OUT}/y_test_2018_aligned.csv", index=False)
test["Label"].value_counts().to_csv("results/tables/class_distribution_2018_aligned.csv")
print("Aligned 2018 test set:", X18.shape, flush=True)
print(test["Label"].value_counts().to_string(), flush=True)