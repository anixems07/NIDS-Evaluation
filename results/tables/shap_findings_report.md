# SHAP Diagnosis — Plain-Language Findings (auto-generated)
Every number is traceable to a CSV in results/tables.

## How to read the metrics
- mean |SHAP| = average influence of a feature on the model's verdict for a class.
  The ranking IS the model's 'evidence list' for that class.
- Spearman rho = agreement between two evidence lists (1 = identical reasoning,
  0 = completely different). Bands: >=0.9 very stable; 0.7-0.9 mostly stable;
  0.5-0.7 moderate; <0.5 major shift.
- Top-10 Jaccard = fraction of the most important features the two lists share.


## DDoS
- Performance: rf_imbalanced — within-2017 F1 1.000 -> cross-2018 F1 0.000 (drop +1.000)
- Performance: rf_smote — within-2017 F1 0.999 -> cross-2018 F1 0.000 (drop +0.999)
- transfer_shift_SMOTE: rho = 0.86 (mostly stable), top-10 overlap 90%
  evidence gained cross/context: Flow IAT Min, Flow Bytes/s, Fwd IAT Min, Bwd IAT Min, Flow IAT Max
  evidence lost: Bwd IAT Std, Flow IAT Mean, Fwd IAT Mean, Bwd Packet Length Min, Fwd Packet Length Min
- transfer_shift_IMB: rho = 0.93 (very stable), top-10 overlap 80%
  evidence gained cross/context: Flow IAT Min, Fwd IAT Min, PSH Flag Count, act_data_pkt_fwd, Flow IAT Mean
  evidence lost: Bwd Packet Length Min, Bwd IAT Max, Fwd Packet Length Min, Fwd Header Length, Init_Win_bytes_backward
- smote_effect_WITHIN: rho = 0.82 (mostly stable), top-10 overlap 80%
  evidence gained cross/context: Total Fwd Packets, Flow Bytes/s, Down/Up Ratio, Fwd IAT Min, Fwd Header Length
  evidence lost: ACK Flag Count, Flow IAT Mean, Bwd IAT Min, PSH Flag Count, min_seg_size_forward
- smote_effect_CROSS: rho = 0.84 (mostly stable), top-10 overlap 90%
  evidence gained cross/context: Fwd IAT Mean, Active Mean, Down/Up Ratio, Flow Packets/s, Active Max
  evidence lost: Init_Win_bytes_backward, PSH Flag Count, ACK Flag Count, Bwd IAT Min, min_seg_size_forward
- Attribution-shift vs distribution-drift correlation: rho = 0.34 (do the features whose importance moved most also drift most between networks?)

## FTP-Patator
- Performance: rf_imbalanced — within-2017 F1 0.999 -> cross-2018 F1 0.000 (drop +0.999)
- Performance: rf_smote — within-2017 F1 0.996 -> cross-2018 F1 0.000 (drop +0.996)
- transfer_shift_SMOTE: rho = 0.75 (mostly stable), top-10 overlap 60%
  evidence gained cross/context: Total Fwd Packets, Bwd IAT Total, ACK Flag Count, Fwd PSH Flags, Fwd Packet Length Mean
  evidence lost: Flow IAT Mean, Packet Length Variance, Flow Packets/s, Flow IAT Std, Max Packet Length
- transfer_shift_IMB: rho = 0.76 (mostly stable), top-10 overlap 50%
  evidence gained cross/context: Flow Packets/s, URG Flag Count, Fwd Packet Length Min, Flow IAT Max, Flow IAT Min
  evidence lost: Bwd Packet Length Max, act_data_pkt_fwd, Packet Length Mean, Packet Length Variance, Flow IAT Mean
- smote_effect_WITHIN: rho = 0.78 (mostly stable), top-10 overlap 50%
  evidence gained cross/context: Total Fwd Packets, Bwd IAT Std, Fwd Packet Length Mean, Total Length of Fwd Packets, Fwd PSH Flags
  evidence lost: PSH Flag Count, Flow Duration, Flow Packets/s, Flow IAT Mean, min_seg_size_forward
- smote_effect_CROSS: rho = 0.58 (moderate shift), top-10 overlap 60%
  evidence gained cross/context: Max Packet Length, Flow IAT Max, Flow Packets/s, Fwd Packet Length Min, URG Flag Count
  evidence lost: Packet Length Mean, Flow IAT Mean, Flow Duration, PSH Flag Count, min_seg_size_forward
- Attribution-shift vs distribution-drift correlation: rho = 0.05 (do the features whose importance moved most also drift most between networks?)

## DoS Hulk
- Performance: rf_imbalanced — within-2017 F1 0.998 -> cross-2018 F1 0.000 (drop +0.998)
- Performance: rf_smote — within-2017 F1 0.997 -> cross-2018 F1 0.000 (drop +0.997)
- transfer_shift_SMOTE: rho = 0.60 (moderate shift), top-10 overlap 40%
  evidence gained cross/context: URG Flag Count, Fwd Header Length, Total Fwd Packets, Flow IAT Min, PSH Flag Count
  evidence lost: Max Packet Length, Packet Length Mean, Fwd Packet Length Max, Fwd Packet Length Mean, Bwd Header Length
- transfer_shift_IMB: rho = 0.67 (moderate shift), top-10 overlap 30%
  evidence gained cross/context: URG Flag Count, Flow IAT Min, min_seg_size_forward, Fwd IAT Min, Flow Duration
  evidence lost: Flow IAT Std, Packet Length Mean, Bwd Packet Length Max, Total Length of Fwd Packets, Max Packet Length
- smote_effect_WITHIN: rho = 0.88 (mostly stable), top-10 overlap 80%
  evidence gained cross/context: FIN Flag Count, Active Max, Total Fwd Packets, Fwd Header Length, Fwd IAT Mean
  evidence lost: Flow IAT Min, Bwd IAT Mean, Bwd Header Length, min_seg_size_forward, PSH Flag Count
- smote_effect_CROSS: rho = 0.81 (mostly stable), top-10 overlap 70%
  evidence gained cross/context: Init_Win_bytes_backward, Fwd Packet Length Mean, FIN Flag Count, Down/Up Ratio, Flow Bytes/s
  evidence lost: ACK Flag Count, Bwd IAT Mean, Bwd Packet Length Max, Bwd Packet Length Min, PSH Flag Count
- Attribution-shift vs distribution-drift correlation: rho = -0.06 (do the features whose importance moved most also drift most between networks?)

## DoS slowloris
- Performance: rf_imbalanced — within-2017 F1 0.998 -> cross-2018 F1 0.865 (drop +0.133)
- Performance: rf_smote — within-2017 F1 0.995 -> cross-2018 F1 0.847 (drop +0.148)
- transfer_shift_SMOTE: rho = 0.93 (very stable), top-10 overlap 90%
  evidence gained cross/context: Active Min, Flow Bytes/s, Fwd Packet Length Mean, Total Length of Fwd Packets, Bwd Packet Length Max
  evidence lost: Flow IAT Mean, Flow IAT Max, Init_Win_bytes_backward, Flow IAT Min, Flow Duration
- transfer_shift_IMB: rho = 0.81 (mostly stable), top-10 overlap 60%
  evidence gained cross/context: Active Max, Packet Length Variance, Max Packet Length, Bwd IAT Mean, Fwd PSH Flags
  evidence lost: Fwd IAT Std, Init_Win_bytes_backward, Init_Win_bytes_forward, min_seg_size_forward, Active Std
- smote_effect_WITHIN: rho = 0.76 (mostly stable), top-10 overlap 50%
  evidence gained cross/context: Active Std, Bwd IAT Min, Total Length of Fwd Packets, Bwd IAT Mean, Fwd Header Length
  evidence lost: Flow Packets/s, Max Packet Length, Bwd Packet Length Min, Active Min, Flow IAT Mean
- smote_effect_CROSS: rho = 0.72 (mostly stable), top-10 overlap 50%
  evidence gained cross/context: Bwd IAT Min, Bwd IAT Max, Bwd IAT Mean, Active Max, Flow Duration
  evidence lost: Flow IAT Std, Init_Win_bytes_backward, Flow IAT Mean, min_seg_size_forward, Active Min
- Attribution-shift vs distribution-drift correlation: rho = 0.39 (do the features whose importance moved most also drift most between networks?)