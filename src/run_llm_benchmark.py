import json
import os
import time
from huggingface_hub import snapshot_download
import numpy as np
import onnxruntime as ort
import psutil
from transformers import AutoTokenizer


def get_ram_usage_mb() -> float:
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)


def prepare_local_onnx_model(
    repo_id: str = "onnx-community/gemma-4-E2B-it-qat-mobile-ONNX",
    local_dir: str = "models/gemma-4-e2b-onnx",
) -> str:
    os.makedirs(local_dir, exist_ok=True)
    decoder_file = os.path.join(local_dir, "onnx", "decoder_model_merged_q2f16.onnx")
    embed_file = os.path.join(local_dir, "onnx", "embed_tokens_q2f16.onnx")
    if os.path.exists(decoder_file) and os.path.exists(embed_file):
        return local_dir

    print(f"Téléchargement des poids texte ONNX depuis {repo_id} (~2,3 Go)...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        ignore_patterns=["*audio_encoder*", "*vision_encoder*", "*.md"],
    )
    return local_dir


class Gemma4OnnxRunner:
    """Exécuteur autoregressif pur ONNX Runtime CPU pour Gemma 4 E2B QAT."""

    def __init__(self, model_dir: str, num_threads: int = 4):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        embed_path = os.path.join(model_dir, "onnx", "embed_tokens_q2f16.onnx")
        decoder_path = os.path.join(model_dir, "onnx", "decoder_model_merged_q2f16.onnx")

        self.embed_sess = ort.InferenceSession(
            embed_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        self.decoder_sess = ort.InferenceSession(
            decoder_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )

        self.embed_output_names = [out.name for out in self.embed_sess.get_outputs()]
        self.decoder_inputs_meta = {inp.name: inp for inp in self.decoder_sess.get_inputs()}
        self.decoder_output_names = [out.name for out in self.decoder_sess.get_outputs()]

    def _init_past_key_values(self, batch_size: int = 1) -> dict:
        past = {}
        for name, meta in self.decoder_inputs_meta.items():
            if name.startswith("past_key_values."):
                shape = []
                for idx, dim in enumerate(meta.shape):
                    if idx == 0:
                        shape.append(batch_size)
                    elif isinstance(dim, int) and dim > 0:
                        shape.append(dim)
                    else:
                        shape.append(0)
                dtype = np.float16 if "float16" in meta.type else np.float32
                past[name] = np.zeros(tuple(shape), dtype=dtype)
        return past

    def generate(
        self,
        prompt_text: str,
        max_new_tokens: int = 20,
        stream_progress: bool = True,
    ) -> tuple[str, int, int]:
        messages = [
            {
                "role": "user",
                "content": f"Answer briefly in Kinyarwanda: {prompt_text}",
            }
        ]

        if getattr(self.tokenizer, "chat_template", None):
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = (
                f"<start_of_turn>user\n{messages[0]['content']}<end_of_turn>\n<start_of_turn>model\n"
            )

        input_ids = self.tokenizer(formatted, return_tensors="np")["input_ids"].astype(np.int64)
        prompt_len = int(input_ids.shape[1])

        past_kv = self._init_past_key_values(batch_size=1)
        generated_tokens = []

        eos_ids = set()
        if self.tokenizer.eos_token_id is not None:
            eos_ids.add(int(self.tokenizer.eos_token_id))
        end_turn_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(end_turn_id, int) and end_turn_id >= 0:
            eos_ids.add(end_turn_id)

        current_ids = input_ids
        total_seq_len = prompt_len

        if stream_progress:
            print(f"  Tokens (max {max_new_tokens}) : ", end="", flush=True)

        for _ in range(max_new_tokens):
            step_len = current_ids.shape[1]

            # 1. Embeddings principaux + per_layer_inputs
            embed_outputs = self.embed_sess.run(None, {"input_ids": current_ids})
            embed_dict = dict(zip(self.embed_output_names, embed_outputs))

            # 2. Entrées du décodeur Gemma 4
            feed = {
                "inputs_embeds": embed_dict["inputs_embeds"],
                "per_layer_inputs": embed_dict["per_layer_inputs"],
                "attention_mask": np.ones((1, total_seq_len), dtype=np.int64),
                "position_ids": np.arange(
                    total_seq_len - step_len, total_seq_len, dtype=np.int64
                )[None, :],
                "num_logits_to_keep": np.array(1, dtype=np.int64),
            }
            feed.update(past_kv)

            # 3. Inférence ONNX
            outputs = self.decoder_sess.run(None, feed)
            logits = outputs[0]

            for out_name, out_val in zip(self.decoder_output_names[1:], outputs[1:]):
                past_name = out_name.replace("present.", "past_key_values.", 1)
                if past_name in past_kv:
                    past_kv[past_name] = out_val

            next_token_id = int(np.argmax(logits[0, -1, :]))
            if next_token_id in eos_ids:
                break

            generated_tokens.append(next_token_id)
            if stream_progress:
                print(".", end="", flush=True)

            current_ids = np.array([[next_token_id]], dtype=np.int64)
            total_seq_len += 1

        if stream_progress:
            print()

        answer_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return answer_text, prompt_len, len(generated_tokens)


def run_llm_benchmark(
    eval_json_path: str = "data/eval_dataset.json",
    output_results_path: str = "results/llm_benchmark_results.json",
    repo_id: str = "onnx-community/gemma-4-E2B-it-qat-mobile-ONNX",
    max_new_tokens: int = 20,
):
    os.makedirs(os.path.dirname(output_results_path), exist_ok=True)

    with open(eval_json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    prompts = dataset.get("llm_prompts", [])
    if not prompts:
        raise ValueError(
            "Aucun prompt trouvé dans la clé 'llm_prompts' de data/eval_dataset.json."
        )

    model_dir = prepare_local_onnx_model(repo_id=repo_id)

    ram_before_load = get_ram_usage_mb()
    print(f"RAM initiale du processus : {ram_before_load} MB")
    print("Chargement du modèle Gemma 4 E2B QAT via ONNX Runtime pur (CPU)...")

    load_start = time.perf_counter()
    runner = Gemma4OnnxRunner(model_dir=model_dir, num_threads=4)
    load_time_sec = round(time.perf_counter() - load_start, 2)
    ram_after_load = get_ram_usage_mb()

    print(f"Modèle chargé en {load_time_sec}s | RAM après chargement : {ram_after_load} MB")
    print("-" * 60)

    results_per_prompt = []
    total_tokens_generated = 0
    total_generation_time_sec = 0.0
    peak_ram_mb = ram_after_load

    for item in prompts:
        prompt_id = item["id"]
        prompt_text = item["prompt_rw"]

        print(f"[{prompt_id}/{len(prompts)}] {item['category']}...")
        start_time = time.perf_counter()
        answer_text, prompt_tokens, completion_tokens = runner.generate(
            prompt_text=prompt_text,
            max_new_tokens=max_new_tokens,
            stream_progress=True,
        )
        elapsed_sec = time.perf_counter() - start_time

        tokens_per_sec = (
            round(completion_tokens / elapsed_sec, 2) if elapsed_sec > 0 else 0.0
        )

        current_ram = get_ram_usage_mb()
        peak_ram_mb = max(peak_ram_mb, current_ram)

        total_tokens_generated += completion_tokens
        total_generation_time_sec += elapsed_sec

        print(
            f"  Latence : {round(elapsed_sec, 2)}s | Tokens : {completion_tokens} "
            f"({tokens_per_sec} tok/s) | RAM : {current_ram} MB"
        )
        print(f"  Q : {prompt_text}")
        print(f"  R : {answer_text}\n")

        results_per_prompt.append({
            "id": prompt_id,
            "category": item["category"],
            "prompt_rw": prompt_text,
            "translation_fr": item.get("translation_fr", ""),
            "response_rw": answer_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_sec": round(elapsed_sec, 3),
            "tokens_per_sec": tokens_per_sec,
            "ram_mb": current_ram,
        })

        # Sauvegarde incrémentale après chaque prompt
        avg_tps = (
            round(total_tokens_generated / total_generation_time_sec, 2)
            if total_generation_time_sec > 0
            else 0.0
        )
        partial_summary = {
            "model_id": repo_id,
            "base_family": "Google Gemma 4 QAT (E2B ONNX quantifié pour déploiement CPU/disque contraint)",
            "stage": "OFFLINE_QUANTIZED_ONNX_CPU",
            "hardware": "Windows 10 Pro | Intel Core i5-1145G7 | 16GB RAM (CPU-only)",
            "metrics": {
                "model_load_time_sec": load_time_sec,
                "ram_before_load_mb": ram_before_load,
                "ram_after_load_mb": ram_after_load,
                "peak_ram_mb": peak_ram_mb,
                "total_prompts_completed": len(results_per_prompt),
                "total_tokens_generated": total_tokens_generated,
                "total_generation_time_sec": round(total_generation_time_sec, 2),
                "average_latency_per_prompt_sec": round(
                    total_generation_time_sec / len(results_per_prompt), 2
                ),
                "average_tokens_per_sec": avg_tps,
            },
            "prompts_results": results_per_prompt,
        }
        with open(output_results_path, "w", encoding="utf-8") as out:
            json.dump(partial_summary, out, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("RÉSULTATS BENCHMARK LLM (Gemma 4 E2B QAT ONNX CPU)")
    print(f"- Débit moyen   : {partial_summary['metrics']['average_tokens_per_sec']} tokens/seconde")
    print(
        f"- Latence moy.  : {partial_summary['metrics']['average_latency_per_prompt_sec']}s / prompt"
    )
    print(f"- Pic RAM       : {peak_ram_mb} MB")
    print(f"- Sauvegardé    : {output_results_path}")


if __name__ == "__main__":
    run_llm_benchmark()