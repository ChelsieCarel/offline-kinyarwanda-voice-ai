## Audio (.wav) $\rightarrow$ Mbaza-ASR $\rightarrow$ Gemma 4 E2B QAT ONNX


import argparse
import gc
import json
import os
import time
import warnings

# Désactiver les logs verbeux et forcer le mode hors ligne strict
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
warnings.filterwarnings("ignore")

import nemo.collections.asr as nemo_asr
import numpy as np
import onnxruntime as ort
import psutil
import soundfile as sf
import torch
from transformers import AutoTokenizer


def get_ram_usage_mb() -> float:
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)


def find_cached_nemo_model() -> str:
    """Localise automatiquement le fichier .nemo de Mbaza-ASR déjà présent dans le cache local."""
    hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    for root, _, files in os.walk(hf_cache):
        for file in files:
            if file.endswith(".nemo") and "mbaza" in root.lower():
                return os.path.join(root, file)
    # Fallback générique sur tout fichier .nemo en cache
    for root, _, files in os.walk(hf_cache):
        for file in files:
            if file.endswith(".nemo"):
                return os.path.join(root, file)
    raise FileNotFoundError(
        "Aucun modèle .nemo trouvé dans le cache Hugging Face local. "
        "Exécute d'abord src/run_asr_benchmark.py."
    )


class OfflineVoiceAssistantPipeline:
    """Pipeline vocal hors ligne : Audio (.wav) -> Mbaza-ASR -> Gemma 4 E2B ONNX."""

    def __init__(
        self,
        llm_model_dir: str = "models/gemma-4-e2b-onnx",
        num_threads: int = 4,
    ):
        self.llm_model_dir = llm_model_dir
        self.num_threads = num_threads
        self.asr_model = None
        self.tokenizer = None
        self.embed_sess = None
        self.decoder_sess = None

    def load_asr(self) -> float:
        nemo_path = find_cached_nemo_model()
        t0 = time.perf_counter()
        self.asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
            restore_path=nemo_path,
            map_location=torch.device("cpu"),
        )
        self.asr_model.eval()
        return round(time.perf_counter() - t0, 2)

    def load_llm(self) -> float:
        embed_path = os.path.join(self.llm_model_dir, "onnx", "embed_tokens_q2f16.onnx")
        decoder_path = os.path.join(
            self.llm_model_dir, "onnx", "decoder_model_merged_q2f16.onnx"
        )
        if not (os.path.exists(embed_path) and os.path.exists(decoder_path)):
            raise FileNotFoundError(
                f"Fichiers ONNX introuvables dans {self.llm_model_dir}. "
                "Exécute d'abord src/run_llm_benchmark.py."
            )

        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_model_dir, local_files_only=True
        )

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.num_threads
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.embed_sess = ort.InferenceSession(
            embed_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        self.decoder_sess = ort.InferenceSession(
            decoder_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )

        self.embed_output_names = [out.name for out in self.embed_sess.get_outputs()]
        self.decoder_inputs_meta = {
            inp.name: inp for inp in self.decoder_sess.get_inputs()
        }
        self.decoder_output_names = [
            out.name for out in self.decoder_sess.get_outputs()
        ]
        return round(time.perf_counter() - t0, 2)

    def transcribe_audio(self, audio_path: str) -> tuple[str, float, float]:
        """Transcrit un fichier WAV 16 kHz et renvoie (texte, durée_audio_s, latence_asr_s)."""
        info = sf.info(audio_path)
        audio_duration_sec = round(info.duration, 2)

        t0 = time.perf_counter()
        with torch.no_grad():
            transcriptions = self.asr_model.transcribe([audio_path], verbose=False)
        asr_latency_sec = round(time.perf_counter() - t0, 3)

        # Compatibilité selon la version de NeMo (liste de strings ou d'objets Hypothesis)
        first_res = transcriptions[0]
        transcript_text = (
            first_res.text if hasattr(first_res, "text") else str(first_res)
        ).strip()

        return transcript_text, audio_duration_sec, asr_latency_sec

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

    def generate_response(
        self, transcript_text: str, max_new_tokens: int = 24
    ) -> tuple[str, int, float, float]:
        """Génère la réponse LLM et renvoie (réponse, nb_tokens, latence_llm_s, tokens_par_sec)."""
        messages = [
            {
                "role": "user",
                "content": f"Answer briefly in Kinyarwanda: {transcript_text}",
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

        input_ids = self.tokenizer(formatted, return_tensors="np")["input_ids"].astype(
            np.int64
        )
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

        t0 = time.perf_counter()
        print("  Génération LLM : ", end="", flush=True)
        for _ in range(max_new_tokens):
            step_len = current_ids.shape[1]

            embed_outputs = self.embed_sess.run(None, {"input_ids": current_ids})
            embed_dict = dict(zip(self.embed_output_names, embed_outputs))

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
            print(".", end="", flush=True)

            current_ids = np.array([[next_token_id]], dtype=np.int64)
            total_seq_len += 1
        print()

        llm_latency_sec = round(time.perf_counter() - t0, 3)
        completion_tokens = len(generated_tokens)
        tokens_per_sec = (
            round(completion_tokens / llm_latency_sec, 2)
            if llm_latency_sec > 0
            else 0.0
        )
        answer_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()

        return answer_text, completion_tokens, llm_latency_sec, tokens_per_sec

    def unload_asr(self):
        """Libère la RAM du modèle ASR avant d'exécuter le LLM sur machines contraintes."""
        self.asr_model = None
        gc.collect()


def run_end_to_end_pipeline(
    audio_path: str | None = None,
    eval_json_path: str = "data/eval_dataset.json",
    output_path: str = "results/pipeline_e2e_results.json",
    max_new_tokens: int = 24,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    reference_text = ""
    if not audio_path:
        if os.path.exists(eval_json_path):
            with open(eval_json_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            samples = dataset.get("asr_samples", [])
            if samples:
                first_sample = samples[0]
                for key in ("audio_path", "path", "file", "audio_filepath", "wav_path"):
                    if key in first_sample:
                        audio_path = first_sample[key]
                        break
                reference_text = first_sample.get(
                    "reference_text", first_sample.get("text", first_sample.get("transcript", ""))
                )

        # Fallback automatique : prend le premier fichier .wav présent dans data/
        if not audio_path and os.path.isdir("data"):
            for root, _, files in os.walk("data"):
                wavs = sorted(f for f in files if f.lower().endswith(".wav"))
                if wavs:
                    audio_path = os.path.join(root, wavs[0])
                    break

    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError(f"Fichier audio introuvable : {audio_path}")

    print("=" * 65)
    print("PIPELINE VOCAL 100 % HORS LIGNE : AUDIO -> MBAZA-ASR -> GEMMA 4")
    print("=" * 65)
    print(f"Fichier audio d'entrée : {audio_path}")
    ram_start = get_ram_usage_mb()
    print(f"RAM initiale           : {ram_start} MB\n")

    pipeline = OfflineVoiceAssistantPipeline()

    # Étape 1 : Chargement et exécution ASR
    print("[Étape 1/2] Chargement de Mbaza-ASR-Afrivoice-660h (CPU)...")
    asr_load_sec = pipeline.load_asr()
    ram_after_asr = get_ram_usage_mb()
    print(f"  ASR chargé en {asr_load_sec}s | RAM : {ram_after_asr} MB")

    transcript_text, audio_dur_sec, asr_latency_sec = pipeline.transcribe_audio(
        audio_path
    )
    rtf = round(asr_latency_sec / audio_dur_sec, 3) if audio_dur_sec > 0 else 0.0
    print(f"  Durée audio   : {audio_dur_sec}s | Latence ASR : {asr_latency_sec}s (RTF: {rtf})")
    if reference_text:
        print(f"  Référence     : {reference_text}")
    print(f"  Transcription : {transcript_text}\n")

    # Libération propre de la RAM ASR avant de charger le LLM
    pipeline.unload_asr()

    # Étape 2 : Chargement et exécution LLM
    print("[Étape 2/2] Chargement de Gemma 4 E2B QAT ONNX (CPU)...")
    llm_load_sec = pipeline.load_llm()
    ram_after_llm = get_ram_usage_mb()
    print(f"  LLM chargé en {llm_load_sec}s | RAM : {ram_after_llm} MB")

    answer_text, completion_tokens, llm_latency_sec, tokens_per_sec = (
        pipeline.generate_response(
            transcript_text=transcript_text, max_new_tokens=max_new_tokens
        )
    )
    peak_ram_mb = max(ram_after_asr, ram_after_llm, get_ram_usage_mb())
    total_inference_latency_sec = round(asr_latency_sec + llm_latency_sec, 3)

    print(f"  Latence LLM   : {llm_latency_sec}s ({completion_tokens} tokens @ {tokens_per_sec} tok/s)")
    print(f"  Réponse LLM   : {answer_text}\n")

    result = {
        "pipeline_mode": "100%_OFFLINE_CPU_SEQUENTIAL_MEMORY_MANAGED",
        "hardware": "Windows 10 Pro | Intel Core i5-1145G7 | 16GB RAM (CPU-only)",
        "input_audio": {
            "file": audio_path,
            "duration_sec": audio_dur_sec,
            "reference_text": reference_text,
        },
        "stage_1_asr": {
            "model": "Mbaza-ASR-Afrivoice-660h",
            "load_time_sec": asr_load_sec,
            "transcription": transcript_text,
            "inference_latency_sec": asr_latency_sec,
            "rtf": rtf,
            "ram_mb": ram_after_asr,
        },
        "stage_2_llm": {
            "model": "onnx-community/gemma-4-E2B-it-qat-mobile-ONNX",
            "load_time_sec": llm_load_sec,
            "response_text": answer_text,
            "completion_tokens": completion_tokens,
            "inference_latency_sec": llm_latency_sec,
            "tokens_per_sec": tokens_per_sec,
            "ram_mb": ram_after_llm,
        },
        "end_to_end_summary": {
            "total_inference_latency_sec": total_inference_latency_sec,
            "peak_ram_mb": peak_ram_mb,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 65)
    print("SYNTHÈSE BOUT-EN-BOUT (AUDIO -> ASR -> LLM)")
    print(f"- Latence inférence totale : {total_inference_latency_sec}s (ASR: {asr_latency_sec}s + LLM: {llm_latency_sec}s)")
    print(f"- Pic RAM contrôlé         : {peak_ram_mb} MB")
    print(f"- Résultat sauvegardé      : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline vocal hors ligne Kinyarwanda (Audio -> Mbaza-ASR -> Gemma 4 ONNX)"
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Chemin vers un fichier audio .wav (16 kHz mono). Par défaut, détecte automatiquement le 1er échantillon.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=24,
        help="Nombre maximum de tokens générés par le LLM (défaut : 24).",
    )
    args = parser.parse_args()

    run_end_to_end_pipeline(audio_path=args.audio, max_new_tokens=args.max_tokens)