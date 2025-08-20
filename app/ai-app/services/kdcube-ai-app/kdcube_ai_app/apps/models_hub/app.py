import sys
import shutil
import s3fs
import time, os, json
import torch
import numpy as np
from flask import Flask, request, jsonify, Response
from typing import List, Any

from huggingface_hub import HfApi, HfFolder

import requests, threading
from functools import wraps

from flask_cors import CORS  # For CORS support
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, PegasusTokenizer, pipeline
from sentence_transformers import SentenceTransformer

from collections import OrderedDict

DEVICE = "mps"

inference_cache = OrderedDict()
inference_cache_lock = threading.Lock()

dataset_cache = OrderedDict()
dataset_search_engine_cache_lock = threading.Lock()

search_engine_cache = OrderedDict()
search_engine_cache_lock = threading.Lock()

MAX_CACHE_SIZE = 2
MAX_DS_CACHE_SIZE = 2
MAX_SEARCH_ENGINE_CACHE_SIZE = 5

import pandas as pd

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        return super().default(obj)

from dotenv import load_dotenv, find_dotenv
def load_env():
    _ = load_dotenv(find_dotenv())
load_env()

EF_API_KEY = os.environ.get("EF_API_KEY")

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return jsonify({"message": "Authentication token is missing"}), 401

        parts = auth_header.split()
        if parts[0].lower() != "bearer" or len(parts) != 2:
            return jsonify({"message": "Invalid token header format"}), 401

        token = parts[1]
        if token != EF_API_KEY:
            return jsonify({"message": "Invalid or expired token"}), 401

        return f(*args, **kwargs)
    return decorated

# ----------------------------------
# CONFIGURATION
# ----------------------------------

CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "~/.cache/llm-server")
S3_MODELS_BASE_PATH = os.environ.get("S3_MODELS_BASE_PATH", "models")

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

if not S3_MODELS_BASE_PATH:
    print("S3_MODELS_BASE_PATH environment variable not set")
    sys.exit(1)
s3 = s3fs.S3FileSystem(anon=False)

DEFAULT_SUMMARIZER_MODEL = "facebook/bart-large-cnn"
DEFAULT_SUMMARIZER_MODEL_2 = "google/pegasus-cnn_dailymail"
# google/pegasus-cnn_dailymail
summarizers = {}

app = Flask(__name__)
# app.json.encoder = NumpyEncoder

CORS(app)

class InferenceDataset:
    """Minimal dataset class for inference with LLMFineTuner"""
    def __init__(self, system_message: str = ""):
        self.system_message = system_message


def get_summarizer(model_name: str):
    """
    Retrieve (or load and cache) a summarization pipeline for the given model_name.
    This version attempts to run on MPS if available.
    """
    if model_name in summarizers:
        return summarizers[model_name]

    # Use MPS if available, otherwise fall back to CPU (-1).
    if torch.backends.mps.is_available():
        device_arg = torch.device("mps")
        print(f"Using MPS device for {model_name}: {device_arg}")
    else:
        device_arg = -1  # CPU
        print(f"MPS not available, using CPU for {model_name}.")

    if model_name == "google/pegasus-cnn_dailymail":
        # tokenizer = PegasusTokenizer.from_pretrained(model_name, use_fast=False, force_download=True)
        tokenizer = PegasusTokenizer.from_pretrained(model_name, use_fast=False)
        print("Loading Pegasus model with ignore_mismatched_sizes=True...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                                      # force_download=True,
                                                      ignore_mismatched_sizes=True)
        print("Pegasus model loaded.")

        # For MPS, sometimes running these models is challenging.
        # You might try to run on MPS, but if issues persist, consider forcing CPU for Pegasus.
        # device_arg = 0 if torch.backends.mps.is_available() and SUMMARIZER_MODEL != "google/pegasus-cnn_dailymail" else -1

    else:
        model = model_name
        tokenizer = model_name
    try:
        summarizer_pipeline = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            device=device_arg
        )
    except Exception as e:
        raise RuntimeError(f"Error loading summarizer {model_name}: {e}")

    summarizers[model_name] = summarizer_pipeline
    return summarizers[model_name]

# ----------------------------------
# FLASK ROUTE: /v1/summarize
# ----------------------------------
@app.route("/v1/summarize", methods=["POST"])
def summarize_endpoint():
    """
    Mimics a summarization API endpoint.
    Expects JSON with fields like:
      {
        "text": "Long text to summarize",
        "max_length": 130,
        "min_length": 30,
        "do_sample": False
      }
    Returns a JSON with the summary.
    """
    data = request.get_json(force=True)
    text = data.get("text", "")
    max_length = data.get("max_length", 130)
    min_length = data.get("min_length", 30)
    do_sample = data.get("do_sample", False)
    model_name = data.get("model", DEFAULT_SUMMARIZER_MODEL)

    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        # Call the summarization pipeline.
        summarizer = get_summarizer(model_name)
        summary_out = summarizer(text, max_length=max_length, min_length=min_length, do_sample=do_sample)
        summary_text = summary_out[0]['summary_text']
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    now = int(time.time())
    response_id = f"local-summary-{now}-{model_name}"

    response_body = {
        "id": response_id,
        "object": "summarization",
        "created": now,
        "summary": summary_text,
        "model": model_name
    }

    return jsonify(response_body)

def _find_best_checkpoint(checkpoint_dir):
    """Find the best checkpoint based on evaluation metrics."""

    if not os.path.exists(checkpoint_dir):
        return None

    best_checkpoint = None
    best_metric = float('inf')  # Assuming lower is better (like loss)

    for checkpoint in os.listdir(checkpoint_dir):
        if checkpoint.startswith('checkpoint-'):
            # Check if there's a trainer_state.json
            state_path = os.path.join(checkpoint_dir, checkpoint, "trainer_state.json")
            if os.path.exists(state_path):
                try:
                    with open(state_path, 'r') as f:
                        state = json.load(f)

                    # Get best metric (usually eval_loss)
                    if 'best_metric' in state:
                        metric = state['best_metric']
                        if metric < best_metric:
                            best_metric = metric
                            best_checkpoint = os.path.join(checkpoint_dir, checkpoint)
                except Exception as e:
                    print(f"Error reading {state_path}: {e}")

    return best_checkpoint

from flask_cors import cross_origin
# ----------------------------------
# FLASK ROUTE: /v1/chat/completions
# ----------------------------------
@app.route("/v1/chat/completions", methods=["POST", "OPTIONS"])
@cross_origin(origin=["http://localhost", "https://ef.demo.kdcube.tech"])
def chat_completions():
    """
    Mimics OpenAI's /v1/chat/completions endpoint.
    Expects JSON with fields like:
      {
        "model": "your-model-name",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 250
      }
    Returns JSON in an OpenAI-compatible structure.
    """
    data = request.get_json(force=True)

    # Extract parameters
    model_name = data.get("model")

    messages = data.get("messages", [])
    temperature = data.get("temperature", 0.7)
    top_p = data.get("top_p", 0.9)
    max_tokens = data.get("max_tokens", 2048)

    def evaluate():
        checkpoint_location = model_name
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_location)
        model = AutoModelForCausalLM.from_pretrained(checkpoint_location).to(DEVICE)
        model.eval()

        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(DEVICE)
        prompt_tokens = input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )

            # Remove the prompt tokens from the generated output
        generated_ids = output_ids[0, prompt_tokens:]
        completion_tokens = generated_ids.shape[0]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text, completion_tokens, prompt_tokens

    try:
        generated_text, completion_tokens, prompt_tokens = evaluate()
    except Exception as ex:
        return jsonify({"error": "Error"}), 500


    now = int(time.time())
    response_id = f"local-{now}-{model_name}"

    usage_data = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens if prompt_tokens else None
    }

    choice_data = {
        "index": 0,
        "message": {
            "role": "assistant",
            "content": generated_text
        },
        "finish_reason": "stop"
    }

    response_body = {
        "id": response_id,
        "object": "chat.completion",
        "created": now,
        "model": f"{model_name}",
        "choices": [choice_data],
        "usage": usage_data
    }

    return jsonify(response_body)

# ----------------------------------
# FLASK ROUTE: /v1/embeddings
# ----------------------------------
@app.route("/serving/v1/embeddings", methods=["POST"])
def embeddings_endpoint():
    """
    Endpoint to compute embeddings for input text using a local SentenceTransformer model.
    The request JSON can contain:
      - "text": the text to embed (required)
      - "size": desired embedding size (optional; default 384)
      - "model": optional model name to override the default for that size.
    Returns the embedding vector.
    """
    data = request.get_json(force=True)
    text = data.get("inputs", "")
    if not text:
        return jsonify({"error": "No text provided."}), 400

    model_name = data.get("model", "sentence-transformers/all-MiniLM-L6-v2")

    # Default model mapping based on embedding size:
    default_models = {
        384: "sentence-transformers/all-MiniLM-L6-v2",
        512: "sentence-transformers/distiluse-base-multilingual-cased",
        768: "sentence-transformers/all-mpnet-base-v2"
    }
    size = None
    if not model_name:
        # Select a model based on desired embedding size or explicit model parameter.
        size = data.get("size", 384)
        model_name = data.get("model", default_models.get(size, "sentence-transformers/all-MiniLM-L6-v2"))
        if not model_name:
            return jsonify({"error": f"Error loading model for dim={size}"})
    size = size or next(iter([k for k, v in default_models.items() if v == model_name]), 0)
    try:
        # Load the embedding model locally (this may take some time the first time)
        embedder = SentenceTransformer(model_name)
        print(f"Embedding model loaded successfully. Output dimension: {embedder.get_sentence_embedding_dimension()}")
    except Exception as e:
        return jsonify({"error": f"Error loading model {model_name}: {str(e)}"}), 500

    try:
        embedding = embedder.encode(text, show_progress_bar=False)
    except Exception as e:
        return jsonify({"error": f"Error computing embedding: {str(e)}"}), 500

    now = int(time.time())
    response_id = f"local-embeddings-{now}-{model_name}"
    response_body = {
        "id": response_id,
        "object": "embedding",
        "created": now,
        "model": model_name,
        "size": size,
        "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
    }
    return jsonify(response_body)

def find_best_checkpoint_s3(s3_checkpoint_dir):
    """
    Find the best checkpoint from S3 based on evaluation metrics.
    Looks for a metrics.json file in each checkpoint directory.

    Args:
        s3_checkpoint_dir: S3 path to directory containing checkpoint subdirectories

    Returns:
        S3 path to the best checkpoint or None if not found
    """
    if not s3.exists(s3_checkpoint_dir):
        print(f"S3 checkpoint directory does not exist: {s3_checkpoint_dir}")
        return None

    best_score = float('-inf')
    best_checkpoint = None

    # List all checkpoint directories in S3
    try:
        checkpoints = [f"{s3_checkpoint_dir}/{path.split('/')[-1]}"
                       for path in s3.ls(s3_checkpoint_dir) if not path.endswith(".json")]
    except Exception as e:
        print(f"Error listing S3 checkpoint directory {s3_checkpoint_dir}: {e}")
        return None
    # Examine each checkpoint
    for checkpoint_path in checkpoints:
        # Look for metrics.json
        metrics_file = f"{checkpoint_path}/metrics.json"
        if not s3.exists(metrics_file):
            continue
        try:
            # Read metrics file from S3
            with s3.open(metrics_file, 'r') as f:
                metrics = json.load(f)

            # Use eval_loss or another appropriate metric
            # Lower is better for loss
            current_score = -metrics.get("eval_loss", float('inf'))

            if current_score > best_score:
                best_score = current_score
                best_checkpoint = checkpoint_path

        except Exception as e:
            print(f"Error reading metrics from {metrics_file}: {e}")
            continue
    return best_checkpoint

def find_best_checkpoint_local(checkpoint_dir):
    """
    Find the best checkpoint based on evaluation metrics from local filesystem.
    Looks for a metrics.json file in each checkpoint directory.

    Args:
        checkpoint_dir: Directory containing checkpoint subdirectories

    Returns:
        Path to the best checkpoint or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return None

    best_score = float('-inf')
    best_checkpoint = None

    # List all checkpoint directories
    for checkpoint in os.listdir(checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)

        # Skip if not a directory
        if not os.path.isdir(checkpoint_path):
            continue

        # Look for metrics.json
        metrics_file = os.path.join(checkpoint_path, "metrics.json")
        if not os.path.exists(metrics_file):
            continue

        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            # Use eval_loss or another appropriate metric
            # Lower is better for loss
            current_score = -metrics.get("eval_loss", float('inf'))

            if current_score > best_score:
                best_score = current_score
                best_checkpoint = checkpoint_path

        except Exception as e:
            print(f"Error reading metrics from {metrics_file}: {e}")
            continue

    return best_checkpoint

def find_latest_checkpoint_local(checkpoint_dir):
    """
    Find the latest checkpoint based on directory modification time.

    Args:
        checkpoint_dir: Directory containing checkpoint subdirectories

    Returns:
        Path to the latest checkpoint or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return None

    checkpoints = []

    # List all checkpoint directories
    for checkpoint in os.listdir(checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)

        # Skip if not a directory
        if not os.path.isdir(checkpoint_path):
            continue

        # Add to list with modification time
        checkpoints.append((checkpoint_path, os.path.getmtime(checkpoint_path)))

    if not checkpoints:
        return None

    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: x[1], reverse=True)

    return checkpoints[0][0]

def download_from_s3(s3_path, local_path):
    """
    Download a directory or file from S3 to a local path.

    Args:
        s3_path: S3 path to download from
        local_path: Local directory to download to
    """
    try:
        # Check if it's a directory
        if s3_path.endswith('/') or len([x for x in s3.ls(s3_path) if x != s3_path]) > 0:
            # It's a directory, download all contents
            for s3_file in s3.find(s3_path):
                # Calculate relative path
                rel_path = s3_file.replace(s3_path, '').lstrip('/')
                local_file_path = os.path.join(local_path, rel_path)

                # Create directory if needed
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # Download file
                if not s3_file.endswith('/'):  # Skip directory entries
                    s3.get(s3_file, local_file_path)
                    print(f"Downloaded {s3_file} to {local_file_path}")
        else:
            # It's a single file
            local_file_path = os.path.join(local_path, os.path.basename(s3_path))
            s3.get(s3_path, local_file_path)
            print(f"Downloaded {s3_path} to {local_file_path}")
    except Exception as e:
        print(f"Error downloading from S3 {s3_path} to {local_path}: {e}")
        raise e

# ----------------------------------
# MAIN ENTRY POINT
# ----------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=False)

