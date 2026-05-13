import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout
import os
import pickle

# Set file path
file_path = r'c:\Users\evlat\OneDrive\Desktop\makine öğrenmesi\Covid Data.csv'

print("Loading data...")
df = pd.read_csv(file_path)

# 1. Feature Engineering: Create Target Variable 'DEATH'
df['DEATH'] = df['DATE_DIED'].apply(lambda x: 0 if x == '9999-99-99' else 1)
df.drop('DATE_DIED', axis=1, inplace=True)

# 2. Data Cleaning
df['INTUBED'] = df['INTUBED'].replace([97, 98, 99], 2)
df['ICU'] = df['ICU'].replace([97, 98, 99], 2)
df['PREGNANT'] = df['PREGNANT'].replace([97, 98, 99], 2)

cols_to_check = ['PNEUMONIA', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 
                 'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 
                 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO']

for col in cols_to_check:
    df = df[df[col] < 98]

print(f"Cleaned data shape: {df.shape}")

# --- EDA Section ---
print("Generating EDA visualizations...")

# Target Distribution
plt.figure(figsize=(8, 8))
df['DEATH'].value_counts().plot.pie(autopct='%1.1f%%', colors=['skyblue', 'salmon'], explode=(0, 0.1), labels=['Survived', 'Died'])
plt.title('Distribution of Deaths (Target Variable)')
plt.savefig('target_distribution.png')

# Correlation Heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.savefig('correlation_heatmap.png')

# 3. Train/Test Split
X = df.drop('DEATH', axis=1)
y = df['DEATH']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- NEW: Activation & Optimizer Comparison (Small sample for speed) ---
print("Comparing Activation Functions and Optimizers...")
sample_idx = np.random.choice(len(X_train_scaled), 50000, replace=False)
X_sample = X_train_scaled[sample_idx]
y_sample = y_train.iloc[sample_idx]

def compare_models(param_type, values):
    results = {}
    for val in values:
        print(f"Training with {param_type}={val}...")
        m = Sequential([
            Dense(32, activation='relu' if param_type=='optimizer' else val, input_shape=(X_train_scaled.shape[1],)),
            Dense(16, activation='relu' if param_type=='optimizer' else val),
            Dense(1, activation='sigmoid')
        ])
        opt = val if param_type=='optimizer' else 'adam'
        m.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        h = m.fit(X_sample, y_sample, epochs=5, batch_size=2048, validation_split=0.2, verbose=0)
        results[val] = h.history['val_loss']
    
    plt.figure(figsize=(10, 6))
    for val, loss in results.items():
        plt.plot(loss, label=val)
    plt.title(f'Model Comparison: {param_type}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.savefig(f'{param_type}_comparison.png')

compare_models('activation', ['relu', 'sigmoid', 'tanh'])
compare_models('optimizer', ['adam', 'sgd', 'rmsprop'])

# 5. Build Final ANN Model
print("Building and training final model...")
from tensorflow.keras.layers import Input
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall()])

# 6. Train Final Model
history = model.fit(
    X_train_scaled, y_train, 
    epochs=10, 
    batch_size=1024, 
    validation_split=0.2,
    verbose=1
)

# 7. Final Performance Analysis
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy History')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss History')
plt.legend()
plt.savefig('accuracy_loss.png')

# 8. Advanced Metrics
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

# Probability Distribution
plt.figure(figsize=(10, 6))
plt.hist(y_pred_prob, bins=50, color='purple', alpha=0.7)
plt.title('Predicted Probability Distribution')
plt.xlabel('Probability of Death')
plt.ylabel('Frequency')
plt.savefig('probability_dist.png')

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='PR Curve', color='green')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('pr_curve.png')

# ROC-AUC
auc_score = roc_auc_score(y_test, y_pred_prob)
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.legend()
plt.savefig('roc_auc_curve.png')

# Threshold Analysis
thresholds = np.linspace(0.1, 0.9, 9)
accs, f1s, recs, precs = [], [], [], []
for t in thresholds:
    yp = (y_pred_prob > t).astype(int)
    accs.append(accuracy_score(y_test, yp))
    f1s.append(f1_score(y_test, yp))
    recs.append(recall_score(y_test, yp))
    precs.append(precision_score(y_test, yp))

plt.figure(figsize=(10, 6))
plt.plot(thresholds, accs, label='Accuracy', marker='o')
plt.plot(thresholds, f1s, label='F1 Score', marker='s')
plt.plot(thresholds, recs, label='Recall', marker='^')
plt.plot(thresholds, precs, label='Precision', marker='d')
plt.title('Metric variation by Threshold')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.savefig('threshold_analysis.png')

# 9. Hidden Space Analysis (PCA & t-SNE)
print("Analyzing Hidden Space...")
# Extract output from the last hidden layer (index -2 because index -1 is the output layer)
hidden_layer_model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
hidden_features = hidden_layer_model.predict(X_test_scaled[:5000]) # 5000 samples for visualization
y_subset = y_test.iloc[:5000]

# PCA
pca = PCA(n_components=2)
pca_res = pca.fit_transform(hidden_features)
plt.figure(figsize=(10, 8))
plt.scatter(pca_res[y_subset==0, 0], pca_res[y_subset==0, 1], alpha=0.5, label='Survived', c='blue')
plt.scatter(pca_res[y_subset==1, 0], pca_res[y_subset==1, 1], alpha=0.5, label='Died', c='red')
plt.title('Hidden Space PCA Visualization')
plt.legend()
plt.savefig('hidden_space_pca.png')

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, max_iter=300)
tsne_res = tsne.fit_transform(hidden_features)
plt.figure(figsize=(10, 8))
plt.scatter(tsne_res[y_subset==0, 0], tsne_res[y_subset==0, 1], alpha=0.5, label='Survived', c='green')
plt.scatter(tsne_res[y_subset==1, 0], tsne_res[y_subset==1, 1], alpha=0.5, label='Died', c='orange')
plt.title('Hidden Space t-SNE Visualization')
plt.legend()
plt.savefig('hidden_space_tsne.png')

# 10. Feature Importance
correlations = df.corr()['DEATH'].sort_values(ascending=False)
plt.figure(figsize=(10, 8))
correlations.drop('DEATH').plot(kind='bar', color='darkred')
plt.title('Feature Correlation with Death outcome')
plt.savefig('feature_importance.png')

print("\nAll 15+ visualizations generated successfully!")
print(f"Final ROC-AUC: {auc_score:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 11. Save Model and Scaler for Web App
print("\nSaving model and scaler...")
model.save('covid_model.keras')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Saved covid_model.keras and scaler.pkl")
