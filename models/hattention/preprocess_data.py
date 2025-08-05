from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from typing import Dict, Iterator


def shard_with_streaming(save_dir: str) -> None:

    def _stream_fn() -> Iterator[Dict[str, str]]:

        stream = load_dataset(
            "togethercomputer/Long-Data-Collections",
            data_dir="pretrain",
            split="train",
            streaming=True)

        for example in stream:
            yield {"text": example["text"]}

    print("=>  Building Arrow dataset with a uniform schema …")
    ds = Dataset.from_generator(_stream_fn)
    print(ds)

    print(f"=> Saving shards to {save_dir}")
    ds.save_to_disk(save_dir, max_shard_size="500MB")


def count_tokens(data_files: str, tokenizer_name: str, batch_size: int = 1000) -> int:
    dataset = load_dataset("arrow", data_files=data_files)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(examples) -> Dict[str, int]:
        tokenized = tokenizer(examples["text"], return_length=True)
        return {"length": tokenized.length}

    token_lengths = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        desc="Counting tokens")
    total_tokens = sum(token_lengths["train"]["length"])
    return total_tokens


if __name__ == "__main__":
    shard_with_streaming("/export/share/data/Long-Data-Collections-preprocessed")
