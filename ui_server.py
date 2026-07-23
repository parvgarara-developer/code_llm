"""
CodeLLM UI Backend Server.
Provides a lightweight HTTP server (zero external dependencies) to serve the web UI 
and run model inference using PyTorch and the trained checkpoints.
"""

import os
import sys
import json
import time
import urllib.parse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

# Import PyTorch safely (since it might not be installed or available yet)
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import SentencePiece safely
try:
    import sentencepiece as spm
    HAS_SENTENCEPIECE = True
except ImportError:
    HAS_SENTENCEPIECE = False

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

# Placeholder for loaded models
LOADED_MODELS = {}

class CORSRequestHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200, "OK")
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == '/api/status':
            self.handle_status()
        elif path == '/api/models':
            self.handle_models()
        else:
            # Serve static files from the 'ui' directory
            self.serve_static(path)

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == '/api/generate':
            self.handle_generate()
        else:
            self.send_error(404, "Not Found")

    def serve_static(self, path):
        if path == '/' or path == '':
            path = '/index.html'
        
        # Prevent directory traversal and normalize path
        parts = [p for p in path.replace('\\', '/').split('/') if p and p != '..' and p != '.']
        file_path = BASE_DIR / 'ui'
        for part in parts:
            file_path = file_path / part

        if not file_path.exists() or file_path.is_dir():
            self.send_error(404, "File Not Found")
            return

        # Determine content type
        content_types = {
            '.html': 'text/html',
            '.css': 'text/css',
            '.js': 'application/javascript',
            '.json': 'application/json',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.svg': 'image/svg+xml',
            '.ico': 'image/x-icon'
        }
        ext = file_path.suffix.lower()
        content_type = content_types.get(ext, 'application/octet-stream')

        try:
            content = file_path.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Internal Server Error: {str(e)}")

    def handle_status(self):
        status = {
            "torch_available": HAS_TORCH,
            "cuda_available": torch.cuda.is_available() if HAS_TORCH else False,
            "gpu_name": torch.cuda.get_device_name(0) if (HAS_TORCH and torch.cuda.is_available()) else None,
            "sentencepiece_available": HAS_SENTENCEPIECE,
            "models_available": self.get_available_models()
        }
        self.send_json_response(200, status)

    def handle_models(self):
        models = self.get_available_models()
        self.send_json_response(200, models)

    def handle_generate(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            req = json.loads(post_data.decode('utf-8'))
        except Exception:
            self.send_json_response(400, {"error": "Invalid JSON format"})
            return

        model_name = req.get('model', 'gpt_opencode')
        prompt = req.get('prompt', '')
        temperature = float(req.get('temperature', 0.8))
        top_k = req.get('top_k', 40)
        top_p = req.get('top_p', 0.95)
        max_tokens = int(req.get('max_tokens', 256))
        repetition_penalty = float(req.get('repetition_penalty', 1.15))

        if not prompt:
            self.send_json_response(400, {"error": "Prompt cannot be empty"})
            return

        if not HAS_TORCH:
            self.send_json_response(500, {"error": "PyTorch is not installed in the current Python environment."})
            return

        # Attempt to run generation
        try:
            start_time = time.time()
            text, logs = self.generate_code(
                model_name, prompt, max_tokens, temperature, top_k, top_p, repetition_penalty
            )
            elapsed = time.time() - start_time
            
            # Simple estimate of tokens generated
            tokens_generated = len(text.split()) # rough estimate
            tokens_per_sec = tokens_generated / max(elapsed, 0.001)

            self.send_json_response(200, {
                "text": text,
                "stats": {
                    "time_taken_sec": round(elapsed, 3),
                    "tokens_generated": tokens_generated,
                    "tokens_per_sec": round(tokens_per_sec, 2),
                    "device": "CUDA" if torch.cuda.is_available() else "CPU"
                },
                "logs": logs
            })
        except Exception as e:
            self.send_json_response(500, {
                "error": f"Generation failed: {str(e)}",
                "logs": [f"[ERROR] {str(e)}"]
            })

    def send_json_response(self, status_code, data):
        response_bytes = json.dumps(data).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(response_bytes))
        self.end_headers()
        self.wfile.write(response_bytes)

    def get_available_models(self):
        available = {}
        
        # 1. gpt_opencode (legacy character-level)
        opencode_ckpt = BASE_DIR / "checkpoints" / "gpt_opencode.pt"
        opencode_cfg = BASE_DIR / "checkpoints" / "gpt_opencode_config.json"
        vocab_path = BASE_DIR / "data" / "processed" / "char_vocab.json"
        
        available["gpt_opencode"] = {
            "name": "GPT OpenCode (Legacy Character-level)",
            "exists": opencode_ckpt.exists() and opencode_cfg.exists() and vocab_path.exists(),
            "type": "character",
            "details": f"Checkpoint: {'Found' if opencode_ckpt.exists() else 'Missing'}, Vocab: {'Found' if vocab_path.exists() else 'Missing'}"
        }

        # 2. gpt_codesearchnet (legacy character-level)
        csn_ckpt = BASE_DIR / "checkpoints" / "gpt_codesearchnet.pt"
        # Uses same config structure as opencode or similar
        available["gpt_codesearchnet"] = {
            "name": "GPT CodeSearchNet (Legacy Character-level)",
            "exists": csn_ckpt.exists() and vocab_path.exists(),
            "type": "character",
            "details": f"Checkpoint: {'Found' if csn_ckpt.exists() else 'Missing'}, Vocab: {'Found' if vocab_path.exists() else 'Missing'}"
        }

        # 3. gpt_subword (subword BPE)
        subword_best = BASE_DIR / "checkpoints" / "gpt_subword_best.pt"
        subword_last = BASE_DIR / "checkpoints" / "gpt_subword.pt"
        subword_cfg = BASE_DIR / "checkpoints" / "gpt_subword_config.json"
        tokenizer_path = BASE_DIR / "data" / "processed" / "code_bpe.model"

        subword_exists = (subword_best.exists() or subword_last.exists()) and subword_cfg.exists() and tokenizer_path.exists()
        available["gpt_subword"] = {
            "name": "GPT Subword (SentencePiece BPE)",
            "exists": subword_exists,
            "type": "subword",
            "details": f"Checkpoint: {'Found' if (subword_best.exists() or subword_last.exists()) else 'Missing'}, Tokenizer: {'Found' if tokenizer_path.exists() else 'Missing'}"
        }

        return available

    def generate_code(self, model_name, task_prompt, max_new_tokens, temperature, top_k, top_p, repetition_penalty):
        logs = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logs.append(f"[INFO] Using device: {device.upper()}")

        if model_name == "gpt_subword":
            if not HAS_SENTENCEPIECE:
                raise ImportError("sentencepiece is required to run the subword model.")
            
            # Load Subword Model
            cfg_file = BASE_DIR / "checkpoints" / "gpt_subword_config.json"
            if not cfg_file.exists():
                raise FileNotFoundError("gpt_subword_config.json not found in checkpoints/")
            
            cfg_d = json.loads(cfg_file.read_text(encoding="utf-8"))
            
            # Import GPT from model.py
            from model import GPT, GPTConfig
            cfg = GPTConfig.from_dict(cfg_d)

            ckpt_path = BASE_DIR / "checkpoints" / "gpt_subword_best.pt"
            if not ckpt_path.exists():
                ckpt_path = BASE_DIR / "checkpoints" / "gpt_subword.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError("gpt_subword_best.pt or gpt_subword.pt not found.")

            logs.append(f"[INFO] Loading subword checkpoint from {ckpt_path.name}...")
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = GPT(cfg).to(device)
            model.load_state_dict(state["model"])
            model.eval()

            tokenizer_path = BASE_DIR / "data" / "processed" / "code_bpe.model"
            if not tokenizer_path.exists():
                raise FileNotFoundError("code_bpe.model tokenizer not found.")

            logs.append("[INFO] Loading SentencePiece tokenizer...")
            sp = spm.SentencePieceProcessor()
            sp.load(str(tokenizer_path))

            prompt = f"### Instruction:\n{task_prompt}\n\n### Response:\n"
            logs.append("[INFO] Tokenizing prompt...")
            ids = sp.encode(prompt)
            logs.append(f"[INFO] Prompt token count: {len(ids)} tokens")

            x = torch.tensor([ids], dtype=torch.long, device=device)
            logs.append(f"[INFO] Running generation (max_tokens={max_new_tokens}, temp={temperature}, top_k={top_k}, top_p={top_p})...")
            
            # Run generate
            y = model.generate(
                x, max_new_tokens=max_new_tokens, temperature=temperature,
                top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                eos_id=cfg_d["eos_id"]
            )
            out = y[0].tolist()
            gen = out[len(ids):]
            if cfg_d["eos_id"] in gen:
                gen = gen[: gen.index(cfg_d["eos_id"])]
            
            decoded = sp.decode(ids + gen)
            # Extract only the response section
            response_marker = "### Response:\n"
            if response_marker in decoded:
                generated_code = decoded.split(response_marker)[1]
            else:
                generated_code = decoded
                
            logs.append("[INFO] Code generation finished.")
            return generated_code, logs

        else:
            # Load Legacy Character-level model (gpt_opencode or gpt_codesearchnet)
            from model_gpt import GPTModel
            
            ckpt_name = "gpt_opencode.pt" if model_name == "gpt_opencode" else "gpt_codesearchnet.pt"
            ckpt_path = BASE_DIR / "checkpoints" / ckpt_name
            config_path = BASE_DIR / "checkpoints" / "gpt_opencode_config.json"
            vocab_path = BASE_DIR / "data" / "processed" / "char_vocab.json"

            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint file {ckpt_name} not found.")
            if not config_path.exists():
                raise FileNotFoundError("gpt_opencode_config.json not found.")
            if not vocab_path.exists():
                raise FileNotFoundError("char_vocab.json not found.")

            logs.append(f"[INFO] Loading legacy model checkpoint from {ckpt_name}...")
            model_config = json.loads(config_path.read_text(encoding="utf-8"))
            
            # Workaround if config file does not contain vocab_size
            vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
            stoi = vocab["stoi"]
            itos = {i: ch for ch, i in stoi.items()}
            
            model_config["vocab_size"] = len(stoi)
            
            model = GPTModel(**model_config).to(device)
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()

            prompt = f"### Instruction:\n{task_prompt}\n\n### Response:\n"
            filtered_prompt = "".join(ch for ch in prompt if ch in stoi)
            if not filtered_prompt:
                raise ValueError("Prompt contains no characters from the training vocabulary.")

            logs.append(f"[INFO] Tokenizing prompt at character-level: {len(filtered_prompt)} characters")
            idx = torch.tensor([[stoi[ch] for ch in filtered_prompt]], dtype=torch.long, device=device)

            logs.append(f"[INFO] Running generation (max_tokens={max_new_tokens}, temp={temperature}, top_k={top_k})...")
            
            with torch.no_grad():
                output_ids = model.generate(
                    idx,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k
                )

            tokens = output_ids[0].tolist()
            text = "".join(itos.get(int(i), "") for i in tokens)
            
            END_TOKEN = "<|end|>"
            if END_TOKEN in text:
                text = text.split(END_TOKEN)[0]

            response_marker = "### Response:\n"
            if response_marker in text:
                generated_code = text.split(response_marker)[1]
            else:
                generated_code = text

            logs.append("[INFO] Code generation finished.")
            return generated_code, logs


def run_server(port=5000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, CORSRequestHandler)
    print(f"CodeLLM UI backend running on http://localhost:{port}")
    print("Open the UI by opening 'ui/index.html' in your browser.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


if __name__ == "__main__":
    port = 5000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
