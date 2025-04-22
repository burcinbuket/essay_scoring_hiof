
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Subtract, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

output_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'Outputs'))

# === Output directories ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = output_dir + f"/llama_embeddings_ternary/val__{timestamp}"
os.makedirs(output_dir + '/h5models/', exist_ok=True)
os.makedirs(output_dir + '/csvs/', exist_ok=True)
os.makedirs(output_dir + '/figs/', exist_ok=True)

# === Load CSV ===
embedding_path = '/Users/burcinbo/Documents/GitRepo_burcinbuket/essay_scoring/essay_scoring/Dataset/hiof_data/llama_embeddings_512/' + 'llama_embeddings_512_50k.csv'
label_path = '/Users/burcinbo/Documents/GitRepo_burcinbuket/essay_scoring/essay_scoring/Dataset/hiof_data/paired_essays_datasets/' + 'paired_essays_50k.csv'

embed_df = pd.read_csv(embedding_path)
label_df = pd.read_csv(label_path)

# Merge labels into embeddings
embed_df = pd.merge(
    embed_df,
    label_df[["text_id_1", "text_id_2", "mean_score_1", "mean_score_2", "score_diff"]],
    on=["text_id_1", "text_id_2"],
    how="left"
)

# Recalculate better essay from mean scores
embed_df["true_label"] = embed_df.apply(
    lambda row: 1 if row["mean_score_1"] > row["mean_score_2"] else (2 if row["mean_score_2"] > row["mean_score_1"] else 0),
    axis=1
)

# === Extract features and labels ===
X1 = embed_df.iloc[:, 2:2050].astype(np.float32).values
X2 = embed_df.iloc[:, 2050:4098].astype(np.float32).values
y_raw = embed_df["true_label"].values
y_cat = to_categorical(y_raw, num_classes=3)

# === Split with index tracking to map test examples correctly ===
indices = np.arange(len(X1))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

X1_train, X1_test = X1[train_idx], X1[test_idx]
X2_train, X2_test = X2[train_idx], X2[test_idx]
y_train, y_test = y_cat[train_idx], y_cat[test_idx]

# === Define Siamese model ===
input_shape = (2048,)

def siamese_encoder(x):
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    return x

input_a = Input(shape=input_shape, name="input_essay1")
input_b = Input(shape=input_shape, name="input_essay2")

encoded_a = siamese_encoder(input_a)
encoded_b = siamese_encoder(input_b)

diff = Subtract(name="embedding_difference")([encoded_a, encoded_b])
output = Dense(3, activation="softmax", name="classification")(diff)

model = Model(inputs=[input_a, input_b], outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# === Callbacks ===
checkpoint_path = output_dir + '/h5models/best_model.h5'
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)

# === Training ===
history = model.fit(
    [X1_train, X2_train], y_train,
    validation_data=([X1_test, X2_test], y_test),
    epochs=50,
    #epochs=5,
    batch_size=32,
    callbacks=[early_stop, model_checkpoint]
)

# === Load best model ===
best_model = load_model(checkpoint_path)

# === Prediction and evaluation ===
predictions = best_model.predict([X1_test, X2_test])
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# === Save predictions with detailed info ===
filtered_test_df = embed_df.iloc[test_idx].reset_index(drop=True)

results_df = pd.DataFrame({
    "text_id_1": filtered_test_df["text_id_1"],
    "text_id_2": filtered_test_df["text_id_2"],
    "mean_score_1": filtered_test_df["mean_score_1"],
    "mean_score_2": filtered_test_df["mean_score_2"],
    "score_diff": filtered_test_df["score_diff"],
    "true_label": true_classes,
    "predicted_label": predicted_classes,
    "prob_essay1_better": predictions[:, 1],
    "prob_essay2_better": predictions[:, 2],
})
results_df.to_csv(output_dir + '/csvs/predictions.csv', index=False)

# === Metrics ===
acc = accuracy_score(true_classes, predicted_classes)
conf_matrix = confusion_matrix(true_classes, predicted_classes)
report_df = pd.DataFrame(classification_report(true_classes, predicted_classes, output_dict=True)).transpose()

report_df.to_csv(output_dir + '/csvs/classification_report.csv')
pd.DataFrame({"metric": ["accuracy"], "value": [acc]}).to_csv(output_dir + "/csvs/summary_metrics.csv", index=False)
pd.DataFrame(conf_matrix,
             index=["True_0", "True_1", "True_2"],
             columns=["Pred_0", "Pred_1", "Pred_2"]).to_csv(output_dir + "/csvs/confusion_matrix.csv")

# === ROC curves ===
y_test_bin = label_binarize(true_classes, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]
fpr, tpr, roc_auc = {}, {}, {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.title('ROC Curve - One-vs-Rest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig(output_dir + '/figs/roc_curve.png')
plt.close()

# === Loss plot ===
plt.style.use("ggplot")
plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Training/Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(output_dir + '/figs/loss_plot.png')
plt.close()

K.clear_session()
print("✅ Training complete. Results saved to:", output_dir)
