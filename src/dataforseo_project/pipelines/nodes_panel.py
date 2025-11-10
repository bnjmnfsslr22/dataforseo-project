import pandas as pd

def build_keyword_panel(serp_df, content_df, llm_df, struct_df):
    df = (serp_df
          .merge(content_df[["keyword","url","body","url_depth"]], on=["keyword","url"], how="left")
          .merge(llm_df, on=["keyword","url"], how="left")
          .merge(struct_df, on=["keyword","url"], how="left"))
    
    df["rank_score"] = -df["rank_absolute"].astype(float)
    df["top10"] = (df["rank_absolute"] <= 10).astype(int)
    return df
