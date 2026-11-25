# Pipeline Vocal 100 % Hors Ligne pour le Kinyarwanda (Mbaza-ASR + Gemma 4 ONNX)

Prototype d'assistant vocal local et autonome conçu pour fonctionner **sans connexion Internet** et **sans GPU** sur des environnements matériels contraints (laptops de terrain, postes communautaires, appareils embarqués).

Le système enchaîne localement :
1. **Reconnaissance vocale (ASR) :** `DigitalUmuganda/Mbaza-ASR-Afrivoice-660h` (NeMo Conformer/CTC BPE).
2. **Compréhension et génération (LLM) :** `onnx-community/gemma-4-E2B-it-qat-mobile-ONNX` (architecture Google Gemma 4 QAT quantifiée exécutée via ONNX Runtime pur sur CPU).

---

## 1. Architecture du Projet

```text
offline_voice_ai/
├── data/
│   ├── audio/                     # Échantillons audio Kinyarwanda (.wav, 16 kHz mono)
│   └── eval_dataset.json          # Jeu d'évaluation verrouillé (transcriptions & 5 prompts)
├── results/
│   ├── asr_benchmark_results.json # Métriques détaillées ASR (WER, RTF, RAM)
│   ├── llm_benchmark_results.json # Métriques détaillées LLM (tok/s, latence, RAM)
│   └── pipeline_e2e_results.json  # Trace d'exécution complète Audio -> ASR -> LLM
├── src/
│   ├── run_asr_benchmark.py       # Évaluation autonome du modèle Mbaza-ASR
│   ├── run_llm_benchmark.py       # Évaluation autonome du modèle Gemma 4 E2B ONNX
│   └── pipeline.py                # Pipeline intégré 100 % hors ligne avec gestion mémoire
├── requirements.txt
└── README.md


Décisions d'ingénierie et gestion des contraintes matériellesEnvironnement matériel cible : Windows 10 Pro, Intel Core i5-1145G7 (CPU-only), 16 Go RAM, 8 Go d'espace disque disponible.Sélection de la variante Gemma 4 QAT (E2B ONNX, ~2,29 Go) : Face à la contrainte stricte de stockage (8 Go libres) et à l'absence de GPU dédié, le checkpoint massif 12B non quantifié (~24 Go) a été écarté au profit de la variante Gemma 4 E2B QAT quantifiée au format ONNX (decoder_model_merged_q2f16.onnx + embed_tokens_q2f16.onnx, en excluant les encodeurs audio/vision non requis).Gestion séquentielle de la RAM (unload_asr) : Dans src/pipeline.py, le modèle ASR est déchargé de la mémoire (gc.collect()) immédiatement après la transcription avant d'initialiser les sessions ONNX du LLM, plafonnant le pic de consommation mémoire global à 2,64 Go.Exécution 100 % hors ligne : Les variables d'environnement HF_HUB_OFFLINE=1 et TRANSFORMERS_OFFLINE=1 garantissent qu'aucune requête réseau n'est émise lors de l'inférence bout-en-bout.2. Résultats Expérimentaux (Benchmarks sur CPU)A. Benchmark ASR (DigitalUmuganda/Mbaza-ASR-Afrivoice-660h)MétriqueValeur mesuréeObservationWER moyen (Word Error Rate)4,96 %Excellente fidélité acoustique et orthographique en kinyarwandaRTF moyen (Real-Time Factor)0,187 (0,083 sur test_02)~5× à 12× plus rapide que le temps réel sur CPUEmpreinte RAM (Pic)1 580 MB (1,58 Go)Léger et stable sur CPUB. Benchmark LLM (onnx-community/gemma-4-E2B-it-qat-mobile-ONNX, 5 prompts Kinyarwanda)MétriqueValeur mesuréeObservationTemps de chargement ONNX8,65 sInitialisation des sessions embed_tokens + decoder (35 couches)Débit moyen de génération0,21 tok/s (jusqu'à 0,27 tok/s)Goulot lié aux conversions logicielles FP16 $\rightarrow$ FP32 sur CPU x86Latence moyenne / prompt89,87 s (20 tokens max)Mesurée en exécution autoregressif pure ONNX RuntimeEmpreinte RAM (Pic)1 867,85 MB (1,87 Go)Compatible avec les appareils à mémoire limitéeC. Exécution Bout-en-Bout (src/pipeline.py sur kin_afrivoice_test_02.wav, 14,88 s)Étape du PipelineTemps de chargementLatence d'inférenceEmpreinte RAM1. Transcription (Mbaza-ASR)2,32 s1,238 s (RTF : 0,083)1 574,07 MB2. Génération (Gemma 4 E2B ONNX)4,73 s72,931 s (20 tokens @ 0,27 tok/s)2 198,15 MBTotal Bout-en-Bout7,05 s74,169 sPic : 2 644,41 MB (2,64 Go)Exemple de transcription ASR produite (kin_afrivoice_test_02.wav) :"inkoko zifite ibara ry'umweru zirimo kurya bakoresheje agakoresho kabugenewe mu kuzigaburira ako gakoresho gafite ibara ry'umweru hasi gafite ibara ry'umutuku"3. Analyse Critique et Perspectives d'OptimisationCompromis Taille vs Maîtrise Linguistique Basse Ressource :Si la variante compacte E2B (2B paramètres) valide la faisabilité mémoire (< 2 Go RAM), sa capacité réduite entraîne des répétitions lexicales en kinyarwanda et un repli ponctuel vers l'anglais sur des tâches complexes d'extraction d'entités (Prompt 4).Recommandation de production : Déployer la variante Gemma 4 E4B ou 12B QAT sur serveur local communautaire ou smartphone équipé d'un NPU dédié.Accélération Matérielle (NPU/GPU vs CPU x86) :Le modèle q2f16 repose sur des tenseurs float16 optimisés pour les accélérateurs mobiles (Android NNAPI / QNN / Metal). Sur processeur x86 sans NPU, ONNX Runtime convertit chaque opération FP16 en FP32, réduisant le débit à ~0,21–0,27 tok/s. Sur smartphone mobile moderne ou via quantification INT4 pure CPU (GGUF Q4_0), un gain de vitesse de $10\times$ à $25\times$ est attendu.4. Installation et ExécutionPowerShell# 1. Créer et activer l'environnement virtuel
python -m venv env
.\env\Scripts\Activate.ps1

## 2. Installer les dépendances
pip install -r requirements.txt

## 3. Exécuter les benchmarks individuels (télécharge et met en cache les modèles)
python src/run_asr_benchmark.py
python src/run_llm_benchmark.py

## 4. Exécuter le pipeline complet 100 % hors ligne
python src/pipeline.py --audio data/audio/kin_afrivoice_test_02.wav --max-tokens 20