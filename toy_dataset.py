from pathlib import Path
from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize

# download tokenizer
nltk.download("punkt")

# dataset configuration
DATASET_ID = "wmt/wmt14"
DATASET_CONFIG = "fr-en"

SRC_LANGUAGE = "fr"
TGT_LANGUAGE = "en"

MAX_SEQUENCE_LEN = 50
RANDOM_SEED = 42

# desired dataset sizes
TRAIN_SIZE_TARGET = 10000
VALID_SIZE_TARGET = 1000
TEST_SIZE_TARGET = 1000

# number of examples sampled before filtering
TRAIN_SAMPLE_POOL = 20000
VALID_SAMPLE_POOL = 3000
TEST_SAMPLE_POOL = 3000

# directory for raw files
RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# load dataset
dataset_full = load_dataset(DATASET_ID, DATASET_CONFIG)

# shuffle and sample candidate examples
train_candidates = dataset_full["train"].shuffle(seed=RANDOM_SEED).select(range(TRAIN_SAMPLE_POOL))
valid_candidates = dataset_full["validation"].shuffle(seed=RANDOM_SEED).select(range(VALID_SAMPLE_POOL))
test_candidates = dataset_full["test"].shuffle(seed=RANDOM_SEED).select(range(TEST_SAMPLE_POOL))


def clean_text(text):
    # remove extra whitespace
    return " ".join(text.strip().split())


def keep_pair(example):
    # filter sentence pairs by length
    src_text = clean_text(example["translation"][SRC_LANGUAGE])
    tgt_text = clean_text(example["translation"][TGT_LANGUAGE])

    src_tokens = word_tokenize(src_text)
    tgt_tokens = word_tokenize(tgt_text)

    return len(src_tokens) <= MAX_SEQUENCE_LEN and len(tgt_tokens) <= MAX_SEQUENCE_LEN


# filter long sentences
train_filtered_data = train_candidates.filter(keep_pair)
valid_filtered_data = valid_candidates.filter(keep_pair)
test_filtered_data = test_candidates.filter(keep_pair)

print("After filtering:")
print("Train:", len(train_filtered_data))
print("Valid:", len(valid_filtered_data))
print("Test :", len(test_filtered_data))

# check that we still have enough examples
if len(train_filtered_data) < TRAIN_SIZE_TARGET:
    raise ValueError(f"Not enough filtered train examples. Found {len(train_filtered_data)}, need {TRAIN_SIZE_TARGET}.")
if len(valid_filtered_data) < VALID_SIZE_TARGET:
    raise ValueError(f"Not enough filtered validation examples. Found {len(valid_filtered_data)}, need {VALID_SIZE_TARGET}.")
if len(test_filtered_data) < TEST_SIZE_TARGET:
    raise ValueError(f"Not enough filtered test examples. Found {len(test_filtered_data)}, need {TEST_SIZE_TARGET}.")

# select final dataset sizes
train_final_split = train_filtered_data.select(range(TRAIN_SIZE_TARGET))
valid_final_split = valid_filtered_data.select(range(VALID_SIZE_TARGET))
test_final_split = test_filtered_data.select(range(TEST_SIZE_TARGET))

print("\nFinal dataset sizes:")
print("Train:", len(train_final_split))
print("Valid:", len(valid_final_split))
print("Test :", len(test_final_split))


def compute_average_lengths(dataset_split):
    # compute average sentence lengths
    src_lengths = []
    tgt_lengths = []

    for example in dataset_split:
        src_text = clean_text(example["translation"][SRC_LANGUAGE])
        tgt_text = clean_text(example["translation"][TGT_LANGUAGE])

        src_tokens = word_tokenize(src_text)
        tgt_tokens = word_tokenize(tgt_text)

        src_lengths.append(len(src_tokens))
        tgt_lengths.append(len(tgt_tokens))

    return sum(src_lengths) / len(src_lengths), sum(tgt_lengths) / len(tgt_lengths)


avg_src_len_train, avg_tgt_len_train = compute_average_lengths(train_final_split)

print("\nAverage sentence length (train split):")
print(f"Source ({SRC_LANGUAGE}): {avg_src_len_train:.2f}")
print(f"Target ({TGT_LANGUAGE}): {avg_tgt_len_train:.2f}")


def build_vocabulary(dataset_split, language):
    # build vocabulary from tokenized sentences
    vocab_set = set()

    for example in dataset_split:
        text = clean_text(example["translation"][language])
        tokens = word_tokenize(text)
        vocab_set.update(tokens)

    return vocab_set


# compute vocabulary sizes
src_vocab_set = build_vocabulary(train_final_split, SRC_LANGUAGE)
tgt_vocab_set = build_vocabulary(train_final_split, TGT_LANGUAGE)

print("\nVocabulary sizes (training split):")
print(f"Source vocab size: {len(src_vocab_set)}")
print(f"Target vocab size: {len(tgt_vocab_set)}")


def save_parallel(dataset_split, src_file, tgt_file):
    # save parallel sentences into files
    with open(src_file, "w", encoding="utf-8") as fs, open(tgt_file, "w", encoding="utf-8") as ft:
        for example in dataset_split:
            src_text = clean_text(example["translation"][SRC_LANGUAGE])
            tgt_text = clean_text(example["translation"][TGT_LANGUAGE])

            fs.write(src_text + "\n")
            ft.write(tgt_text + "\n")


# write dataset splits to files
save_parallel(train_final_split, RAW_DATA_DIR / "train.fr", RAW_DATA_DIR / "train.en")
save_parallel(valid_final_split, RAW_DATA_DIR / "valid.fr", RAW_DATA_DIR / "valid.en")
save_parallel(test_final_split, RAW_DATA_DIR / "test.fr", RAW_DATA_DIR / "test.en")

print("\nFiles saved successfully in data/raw/")
print(f"Saved: {RAW_DATA_DIR / 'train.fr'}")
print(f"Saved: {RAW_DATA_DIR / 'train.en'}")
print(f"Saved: {RAW_DATA_DIR / 'valid.fr'}")
print(f"Saved: {RAW_DATA_DIR / 'valid.en'}")
print(f"Saved: {RAW_DATA_DIR / 'test.fr'}")
print(f"Saved: {RAW_DATA_DIR / 'test.en'}")