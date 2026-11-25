import json
import os


def lock_eval_dataset(json_path: str = "data/eval_dataset.json"):
  if not os.path.exists(json_path):
    raise FileNotFoundError(
        f"Le fichier {json_path} est introuvable. Vérifie le chemin."
    )

  with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

  # 5 consignes fixes en Kinyarwanda pour évaluer la qualité, le raisonnement et la latence du LLM hors ligne
  data["llm_prompts"] = [
      {
          "id": 1,
          "category": "Conseil pratique (Agriculture)",
          "prompt_rw": (
              "Ni izihe nama eshatu z'ingenzi waha umuhinzi w'ibigori mu gihe"
              " cy'izuba ryinshi?"
          ),
          "translation_fr": (
              "Quels sont trois conseils essentiels à donner à un cultivateur"
              " de maïs en période de sécheresse ?"
          ),
          "expected_criteria": (
              "Pertinence agronomique, clarté en kinyarwanda, structure en 3"
              " points."
          ),
      },
      {
          "id": 2,
          "category": "Vulgarisation / Simplification",
          "prompt_rw": (
              "Sobanura mu magambo make kandi yumvikana akamaro ko gukoresha"
              " ifumbire y'imborera ku butaka."
          ),
          "translation_fr": (
              "Explique en termes simples et courts l'importance d'utiliser de"
              " l'engrais organique pour le sol."
          ),
          "expected_criteria": (
              "Langage accessible, exactitude sur la fertilité des sols."
          ),
      },
      {
          "id": 3,
          "category": "Orientation / Premiers secours",
          "prompt_rw": (
              "Umwana afite umuriro mwinshi nijoro kandi ivuriro riri kure. Ni"
              " iki umubyeyi yakora mu gihe ategereje kujya kwa muganga?"
          ),
          "translation_fr": (
              "Un enfant a une forte fièvre la nuit et le centre de santé est"
              " loin. Que doit faire le parent en attendant de consulter ?"
          ),
          "expected_criteria": (
              "Mesures sûres (hydratation, déshabiller légèrement), aucune"
              " prescription médicale risquée."
          ),
      },
      {
          "id": 4,
          "category": "Compréhension / Extraction d'information",
          "prompt_rw": (
              "Soma iyi nteruro hanyuma uvuge ibikenewe: 'Ejo mu gitondo"
              " abahinzi bo mu karere ka Musanze bazahura saa mbiri kugira ngo"
              " bahabwe imbuto z'ibirayi.'"
          ),
          "translation_fr": (
              "Lis cette phrase et indique ce qui est nécessaire : 'Demain"
              " matin, les agriculteurs du district de Musanze se réuniront à"
              " 8h pour recevoir des semences de pommes de terre.'"
          ),
          "expected_criteria": (
              "Extraction exacte : lieu (Musanze), heure (8h / saa mbiri),"
              " objet (semences de pommes de terre)."
          ),
      },
      {
          "id": 5,
          "category": "Communication / Sensibilisation",
          "prompt_rw": (
              "Andika ubutumwa bugufi bwo gushishikariza urubyiruko kwitabira"
              " ikoranabuhanga mu buhinzi."
          ),
          "translation_fr": (
              "Rédige un court message pour encourager les jeunes à adopter la"
              " technologie dans l'agriculture."
          ),
          "expected_criteria": "Ton motivant, vocabulaire naturel en kinyarwanda.",
      },
  ]

  # Mise à jour des métadonnées pour tracer le verrouillage du dataset
  if "dataset_metadata" in data:
    data["dataset_metadata"]["total_llm_prompts"] = len(data["llm_prompts"])
    data["dataset_metadata"]["status"] = (
        "LOCKED_FOR_BASELINE_AND_OPTIMIZED_EVAL"
    )

  with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

  print(f"Jeu de données verrouillé avec succès dans {json_path} !")
  print(f"- Extraits audio ASR : {len(data.get('asr_samples', []))}")
  print(f"- Consignes fixes LLM : {len(data['llm_prompts'])}")


if __name__ == "__main__":
  lock_eval_dataset()