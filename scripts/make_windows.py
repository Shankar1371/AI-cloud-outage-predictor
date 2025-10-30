"""
1) Read raw provider traces and normalize to a common schema.
2) Resample to 1-min; compute event counts per minute.
3) Build labels for horizon H.
4) Save interim parquet and processed torch tensors.
"""