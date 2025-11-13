### ✅ Synthetic Data Generation Summary

- **Read your original Excel as-is** and kept every column unchanged  
  (`Account No`, `DATE`, `TRANSACTION DETAILS`, `CHQ.NO.`, `VALUE DATE`,  
  `WITHDRAWAL AMT`, `DEPOSIT AMT`, `BALANCE AMT`).

- **Created a new field: `RAW_TEXT`**  
  - Built directly from `TRANSACTION DETAILS` + amount (if present).  
  - Represents the “true” transaction string before noise.

- **Generated 3 AI-based synthetic noisy variants for each row (`SYN_RAW_TEXT`)**  
  - Added realistic banking noise (UPI/POS/IMPS/NEFT tags).  
  - Injected abbreviations, masked digits, and random punctuation.  
  - Added marketplace suffixes (MKT, PAY, MKTPLC).  
  - Simulated messy real-world merchant strings.

- **Produced `CLEAN_TEXT`**  
  - Lowercased, punctuation removed, numbers normalized.  
  - Ideal input format for BERT training.

- **Inferred a simple `CATEGORY` label**  
  - Keyword-based heuristic mapping (e.g., Amazon → Shopping,  
    UPI → Transfer, Shell → Fuel).  
  - Provides starter labels for model training.

- **Added metadata fields**  
  - `SOURCE` = `"original"` or `"aug_v1" / "aug_v2" / "aug_v3"`.  
  - `ORIGINAL_ROW_ID` links augmented samples to the base row.

- **Exported everything to a training-ready CSV**  
  - File saved as **`synthetic_augmented_transactions.csv`**.
