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
# 5 Model architecture
'''
A text classification model in torchTextClassifiers is built from three components (on top of the ValueEncoder and the tokenizer seen above):

TextEmbedder: converts token IDs into dense vectors (embeddings) of size embedding_dim.
NoteEmbeddings, a key concept in NLP models
CategoricalVariableNet (optional): encodes additional categorical features and merges them with the text representation. Not used here.

ClassificationHead: projects the final representation onto num_classes dimensions.
The highest value determines the predicted class.

To learn more about the building blocks of the torchTextClassifiers package, 
please visit the documentation.

These are configured through ModelConfig and assembled automatically by 
torchTextClassifiers. 
You can see some ModelConfig examples here and for torchTextClassifiers here.

Key training hyperparameters:

num_epochs: how many times the model sees the full training set.
batch_size: how many examples are processed at once before a weight update.
lr (learning rate): how large each weight update step is.
patience_early_stopping: stop training if the validation loss has not improved for this many epochs, to avoid overfitting.
'''
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers

embedding_dim = 96

model_config = ModelConfig(
    embedding_dim=embedding_dim,
    num_classes=n_classes,) # n de classes de codificação que quermeos prever

ttc = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
    value_encoder=value_encoder,
)

# %%
# 6.2 Question 2 — Prepare training
training_config = TrainingConfig(
    num_epochs=1,
    batch_size=128,
    lr=5 * 1e-4,
    patience_early_stopping=5,
)
# %%
# Para testar de GEMINI (ver no dashboard)
import mlflow
import os

# 1. Fetch the secret address injected by SSPCLOUD
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

# 2. Tell MLflow to send data to that server instead of your local disk
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
    print(f"Connected to SSPCLOUD MLflow server at: {tracking_uri}")
else:
    print("Warning: MLFLOW_TRACKING_URI not found. Are both services running in the same namespace?")

# 3. Now set your experiment and run your training!
mlflow.set_experiment("funathon-2026-project2")
mlflow.pytorch.autolog()
# %%
# 6.3 Question 3 — Train on a small subsample
'''
mlflow.set_experiment(...): sets the active experiment (created automatically if it does not exist).
Each call inside mlflow.start_run() creates a new run inside that experiment, so successive training attempts are kept separate and comparable.
If MLFLOW_TRACKING_URI is set in your environment (e.g. via .env), metrics are forwarded to that remote server automatically.

'''
mlflow.set_experiment("funathon-2026-project2")
mlflow.pytorch.autolog()

with mlflow.start_run() as run:
    # This should take approximately 1-2mn
    ttc.train(
        X_train,
        y_train,
        training_config=training_config,
        X_val=X_val,
        y_val=y_val,
        verbose=True,
    )

    mlflow.log_artifacts(
        training_config.save_path,   # local folder produced by ttc.train()
        artifact_path="model_artifacts",
    )
# %%
# 7 Prediction and explainability
# 7.1 Question 0 — Load the pretrained model from MLflow
import s3fs

fs = s3fs.S3FileSystem(
    anon=True,  # public bucket
    endpoint_url="https://minio.lab.sspcloud.fr",
)

local_dir = "./mlflow-artifacts/"
fs.get(
    "projet-funathon/diffusion/mlflow-artifacts/",
    local_dir,
    recursive=True,
)
# Rebuild the torchTextClassifiers object from the downloaded files
ttc = torchTextClassifiers.load(local_dir)

ttc.pytorch_model.eval()


# %%
# 7.2 Question 1 — Generate top-5 predictions with confidence scores
import random

random_indices = random.sample(range(len(X_test)), 3)
example_texts = X_test[random_indices]
example_true_codes = y_test[random_indices]
print(example_texts)
top_k = 5
results = ttc.predict(example_texts, top_k=top_k, explain_with_captum=True)
for i, text in enumerate(example_texts):
    predicted_codes = [results["prediction"][i][k] for k in range(top_k)]
    confidence = [results["confidence"][i][k].item() for k in range(top_k)]
    print(f"\nText: {text}")
    print(f"  True code: {example_true_codes[i]}")
    for code, conf in zip(predicted_codes, confidence):
        print(f"  {code}  (confidence: {conf:.3f})")

# %%
# 7.3 Question 2 — Visualise word attributions for the top prediction
from torchTextClassifiers.utilities.plot_explainability import (
    map_attributions_to_char, map_attributions_to_word,
    plot_attributions_at_char, plot_attributions_at_word, figshow,
)

text_idx = 0
top_k_idx = 0
text_sample         = example_texts[text_idx]
offsets             = results["offset_mapping"][text_idx]
word_ids            = results["word_ids"][text_idx]
predicted_code = results["prediction"][text_idx][top_k_idx]

attributions  = results["captum_attributions"][text_idx][top_k_idx] # (seq_len,)

words, word_attributions = map_attributions_to_word(
    attributions.unsqueeze(0), text_sample, word_ids, offsets
)
char_attributions = map_attributions_to_char(attributions.unsqueeze(0), offsets, text_sample)

titles = [f"Attributions for NACE code {predicted_code}"]

figshow(plot_attributions_at_char(
    text=text_sample, attributions_per_char=char_attributions, titles=titles,
)[0])

figshow(plot_attributions_at_word(
    text=text_sample, words=words.values(), attributions_per_word=word_attributions, titles=titles,
)[0])
# %%
# 7.4 Question 3 — Evaluate accuracy on the test set
results_test = ttc.predict(X_test, top_k=1)
preds    = results_test["prediction"].squeeze(1)
accuracy = (preds == y_test).mean()
print(f"Test accuracy: {accuracy:.4f} ({int(accuracy * len(y_test))}/{len(y_test)} correct)")
# %%
