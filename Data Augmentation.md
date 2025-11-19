Dataset Augmentation Summary (Team Zenloop)


✔️ Constructed RAW_TEXT and CLEAN_TEXT

RAW_TEXT built from merchant + amount.

CLEAN_TEXT normalized (lowercased, punctuation/extra tokens removed) for model training.

✔️ Created Three Synthetic Variants (aug_v1, aug_v2, aug_v3)

aug_v1 — injects payment-channel noise (UPI, NEFT, IMPS, ATM).

aug_v2 — adds marketplace noise (MKT, PAY, special chars, masking).

aug_v3 — combines both patterns for stronger variability.

✔️ Generated Simple Category Labels

Rule-based mapping assigns high-level categories: Transfer, Shopping, Cash Withdrawal, Other.

✔️ Added Metadata Columns

Added SOURCE, ORIGINAL_ROW_ID, AMOUNT, IS_CREDIT, IS_DEBIT to track provenance and support supervised learning.

✔️ Exported a Training-Ready CSV

Each original row now expands into 4 rows
(1 original + 3 augmented variants)

Final output is a clean, model-ready dataset for BERT/LLM training.
