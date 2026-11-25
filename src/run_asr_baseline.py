import json
import os
import time
import jiwer
import nemo.collections.asr as nemo_asr
import psutil
import torch


def get_ram_usage_mb() -> float:
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)


def run_asr_baseline(
    eval_json_path: str = "data/eval_dataset.json",
    output_results_path: str = "results/asr_baseline_results.json",
    model_id: str = "DigitalUmuganda/Mbaza-ASR-Afrivoice-660h",
):
    os.makedirs(os.path.dirname(output_results_path), exist_ok=True)

    with open(eval_json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    asr_samples = dataset.get("asr_samples", [])
    if not asr_samples:
        raise ValueError("Aucun échantillon audio trouvé dans eval_dataset.json.")

    ram_before_load = get_ram_usage_mb()
    print(f"RAM initiale du processus : {ram_before_load} MB")
    print(f"Chargement du modèle {model_id} sur CPU (~500 Mo au 1er lancement)...")

    load_start = time.perf_counter()
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name=model_id, map_location=torch.device("cpu")
    )
    asr_model.eval()
    load_time_sec = round(time.perf_counter() - load_start, 2)
    ram_after_load = get_ram_usage_mb()

    print(f"Modèle chargé en {load_time_sec}s | RAM après chargement : {ram_after_load} MB")
    print("-" * 60)

    results_per_clip = []
    all_references = []
    all_predictions = []
    total_audio_sec = 0.0
    total_inference_sec = 0.0
    peak_ram_mb = ram_after_load

    for item in asr_samples:
        audio_path = item["audio_file"]
        ref_text = item["reference_text"].strip().lower()
        duration = item["duration_sec"]

        start_inf = time.perf_counter()
        transcriptions = asr_model.transcribe([audio_path])
        inf_time = time.perf_counter() - start_inf

        raw_pred = transcriptions[0]
        pred_text = (
            raw_pred.text.strip().lower()
            if hasattr(raw_pred, "text")
            else str(raw_pred).strip().lower()
        )

        clip_wer = round(jiwer.wer(ref_text, pred_text) * 100, 2)
        clip_rtf = round(inf_time / duration, 3)

        current_ram = get_ram_usage_mb()
        peak_ram_mb = max(peak_ram_mb, current_ram)

        total_audio_sec += duration
        total_inference_sec += inf_time
        all_references.append(ref_text)
        all_predictions.append(pred_text)

        print(
            f"[{item['id']:02d}/{len(asr_samples)}] Durée: {duration}s | "
            f"Inférence: {round(inf_time, 2)}s (RTF: {clip_rtf}) | WER: {clip_wer}%"
        )
        print(f"  REF : {ref_text}")
        print(f"  PRED: {pred_text}\n")

        results_per_clip.append({
            "id": item["id"],
            "audio_file": audio_path,
            "duration_sec": duration,
            "inference_time_sec": round(inf_time, 3),
            "rtf": clip_rtf,
            "wer_percent": clip_wer,
            "reference_text": ref_text,
            "predicted_text": pred_text,
        })

    global_wer = round(jiwer.wer(all_references, all_predictions) * 100, 2)
    global_cer = round(jiwer.cer(all_references, all_predictions) * 100, 2)
    global_rtf = round(total_inference_sec / total_audio_sec, 3)

    summary = {
        "model_id": model_id,
        "stage": "BASELINE_FP32_CPU",
        "hardware": "Windows 10 Pro | Intel Core i5-1145G7 | 16GB RAM (CPU-only)",
        "metrics": {
            "model_load_time_sec": load_time_sec,
            "ram_before_load_mb": ram_before_load,
            "ram_after_load_mb": ram_after_load,
            "peak_ram_mb": peak_ram_mb,
            "total_audio_duration_sec": round(total_audio_sec, 2),
            "total_inference_time_sec": round(total_inference_sec, 2),
            "average_latency_per_clip_sec": round(
                total_inference_sec / len(asr_samples), 3
            ),
            "global_rtf": global_rtf,
            "global_wer_percent": global_wer,
            "global_cer_percent": global_cer,
        },
        "clips": results_per_clip,
    }

    with open(output_results_path, "w", encoding="utf-8") as out:
        json.dump(summary, out, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("RÉSULTATS BASELINE ASR (Mbaza-ASR-Afrivoice-660h)")
    print(f"- WER Global : {global_wer} % | CER Global : {global_cer} %")
    print(f"- RTF Moyen  : {global_rtf} (< 1.0 = plus rapide que le temps réel)")
    print(f"- Pic RAM    : {peak_ram_mb} MB")
    print(f"- Sauvegardé : {output_results_path}")


if __name__ == "__main__":
    run_asr_baseline()