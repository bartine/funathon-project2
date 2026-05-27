# %%
import mlflow
from dotenv import load_dotenv

load_dotenv(override=True)
print('Fim Load')
# %%
# Ler o DF de um ficheiro parquet
import polars as pl

df = pl.read_parquet("https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet")

print(df.head())
print(f"Total rows: {len(df)}")

# %%
# Count Unque NACE Codes
n_classes = df['code'].n_unique()
print(f'Numero de classes do code {n_classes}')

# %%
# Question 1 — Split the dataset into train / validation / test sets
from sklearn.model_selection import train_test_split

train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=42)
val_df, test_df  = train_test_split(tmp_df, test_size=0.50, random_state=42)

X_train, y_train = train_df["label"].to_numpy(), train_df["code"].to_numpy()
X_val, y_val = val_df["label"].to_numpy(), val_df["code"].to_numpy()
X_test, y_test = test_df["label"].to_numpy(), test_df["code"].to_numpy()

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# %%
# 3.2 Question 2 — Encode the labels
# Label Encoder: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(train_df['code'].to_numpy())

# %%
# Verificar Existencia todos os codigos

all_codes = set(df['code'])
train_codes = set(train_df['code'])
missing = all_codes - train_codes

if missing:
    print(f"WARNING: {len(missing)} code(s) missing from training set: {missing}")
else:
    print(f"OK — all {len(all_codes)} codes appear in the training set.")


# %%
# 3.3 Question 3 — Prepare the labels to use them with ttc
from torchTextClassifiers.value_encoder import ValueEncoder

value_encoder = ValueEncoder(label_encoder=encoder)
print('fim encoding')
# %%
# 4 Applying a tokenizer
# 4.1 Why tokenization?
# 4.2 Why subword tokenization?
# 4.3 Question 1 — Train the tokenizer and inspect a sample
from torchTextClassifiers.tokenizers import WordPieceTokenizer
# X_train
tokenizer = WordPieceTokenizer(vocab_size=5000, output_dim=10)
tokenizer.train(X_train)

print("Output tensor size:", tokenizer.tokenize(X_train[0]).input_ids.shape)
print("Vocabulary size:", tokenizer.vocab_size)

# Look at an example of tokenization
print("Raw text", X_train[0])
print(
    "Tokens id:",
    tokenizer.tokenize(X_train[0]).input_ids.squeeze(0)
)
print(
    "Tokens:",
    tokenizer.tokenizer.convert_ids_to_tokens(
        tokenizer.tokenize(X_train[0]).input_ids.squeeze(0)
    )
)

# %%
# 
print(X_train[24])

# %%
