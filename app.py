from flask import Flask, render_template, request
import gunicorn
import torch
import joblib
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
preprocessor = joblib.load('preprocessor_pipeline.pkl')

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        embeddings = self.encoder(x)
        reconstructed = self.decoder(embeddings)
        return reconstructed, embeddings

data = pd.read_csv('Hiring_dataset.csv')
input_dim = preprocessor.transform(data).shape[1]
model = Autoencoder(input_dim)
model.load_state_dict(torch.load('recommendermodel.pth', map_location='cpu'))
model.eval()

# Load the embeddings and the original dataset
embeddings = np.load('embeddings.npy')
candidates_df = data


@app.route('/home')
def myhome():
    return render_template('home.html')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    candidate = {key: request.form[key] for key in request.form}
    sample_df = pd.DataFrame([candidate])
    sample_processed = preprocessor.transform(sample_df)

    # Convert to  tensor
    sample_tensor = torch.tensor(sample_processed, dtype=torch.float32)

    # Generate embedding for the sample
    with torch.no_grad():
        _, sample_embedding = model(sample_tensor)

    # Computing cosine similarity
    sample_embedding_np = sample_embedding.numpy()
    similarities = cosine_similarity(sample_embedding_np, embeddings)[0]

    # Get top matches
    top_indices = np.argsort(similarities)[::-1][:5]

    candidates = []
    for idx in top_indices:
        similarity_score = similarities[idx]

        # Get candidate details from the original dataset
        candidate_details = candidates_df.iloc[idx].to_dict()

        # Append details with similarity
        candidates.append({
            "index": idx,
            "similarity": f"{similarity_score:.4f}",
            "details": candidate_details
        })

    return render_template('result.html', candidates=candidates, top_n=5)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
