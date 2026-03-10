from datasets import load_dataset

def load_data():
    df_train = load_dataset("zh-plus/tiny-imagenet", split="train").to_pandas()
    df_val = load_dataset("zh-plus/tiny-imagenet", split="validation").to_pandas()
    df_test = load_dataset("zh-plus/tiny-imagenet", split="test").to_pandas()
    return df_train, df_val, df_test