<!---
Copyright 2026 EGen Team. All rights reserved.

Licensed under the MIT License.
-->

<div align="center">
    <img src="../../docs/assets/banner.png" alt="THL Banner" width="100%"/>
</div>
<br>

<p align="center">
    <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
    <img src="https://img.shields.io/badge/vram-4GB-orange.svg" alt="VRAM Optimized">
    <a href="https://github.com/EGen-V/Transformer-Hierarchical-Layers/actions"><img src="https://github.com/EGen-V/Transformer-Hierarchical-Layers/workflows/Tests/badge.svg" alt="Tests"></a>
</p>

<h1 align="center">🤗 THL: Transformer Hierarchical Layers</h1>

<p align="center">
    <a href="README_AR.md">العربية</a> |
    <a href="../../README.md">English</a> |
    <a href="README_ES.md">Español</a> |
    <a href="README_FR.md">Français</a> |
    <a href="README_zh-hans.md">简体中文</a>
</p>

<h3 align="center">
    Modèles Récurrents Hiérarchiques de Pointe pour Matériel à Faibles Ressources
</h3>

<p align="center">
    THL est un graphe de calcul récurrent hiérarchique, strictement non-Transformer, conçu pour exécuter de grands modèles de langage sur <b>4 Go de VRAM</b> et des appareils mobiles.
</p>

---

**THL** résout le problème spécifique de l'**explosion de la mémoire cache KV** dans les Transformers en utilisant une **Mémoire Indépendante de la Longueur de Séquence** (mémoire O(1) par couche). Il atteint des performances compétitives avec les Transformers tout en permettant l'inférence sur du matériel grand public.

## ⚡ Pourquoi utiliser THL ?

1.  **Mémoire Bornée (O(1))** : Oubliez le cache KV en O(T). THL utilise une mémoire à emplacements fixes (`J=1024`), permettant une génération de contexte infinie sans planter votre GPU.
2.  **Récurrence Hiérarchique** : Des niveaux GRU à échelles multiples traitent l'information à différentes fréquences ($\tau_k$), capturant efficacement la syntaxe locale et la sémantique globale.
3.  **Inférence Faible VRAM** : Le **Moteur d'Inférence par Couches** intégré permet d'exécuter des modèles de plus de 7B paramètres sur <4 Go de VRAM.
4.  **Routage Épars** : Le routage Top-K multi-têtes garantit que les souvenirs pertinents sont consultés sans traiter l'historique complet.

## 🛠️ Installation

```bash
# Cloner le dépôt
git clone https://github.com/EGen-V/Transformer-Hierarchical-Layers.git
cd Core

# Installer les dépendances
pip install -r requirements.txt
pip install .
```

## 🚀 Tour Rapide

### 1. Modélisation de Langue Basique

Instanciez facilement un modèle et exécutez une passe avant :

```python
import torch
from thl.config import THLConfig
from thl.model import THLModel

# Configurer pour 4 Go de VRAM
config = THLConfig(
    num_tiers=3,
    memory_slots=1024,
    dim=768
)

model = THLModel(config)
input_ids = torch.randint(0, 50257, (1, 32))
logits, state = model(input_ids)
```

### 2. Génération Faible VRAM (Streaming)

Exécutez de plus grands modèles en streamant les couches vers le GPU une par une :

```python
from thl.inference.layered import LayeredInferenceEngine
from thl.inference.state import InferenceState

engine = LayeredInferenceEngine(model, device="cuda")
state = InferenceState.init(1, config, model.tiers, model.memory_bank)

# Étape de génération d'un seul token
token = torch.tensor([123])
logit, state = engine.step(token, state)
```

## 🏗️ Architecture

| Composant | Symbole | Description |
|-----------|---|-------------|
| **Banque de Mémoire** | $M_t$ | Matrice de taille fixe ($J \times d$) conservant le contexte à long terme. |
| **Routeur Épars** | $r_t$ | Mécanisme de routage Top-K pour lire les emplacements pertinents. |
| **Niveaux Hiérarchiques** | $s_t^{(k)}$ | Pile de cellules récurrentes mises à jour à intervalles exponentiels $\tau=2^k$. |
| **Écrivain de Nouveauté** | $w_t$ | Mécanisme à porte pour écrire uniquement les nouvelles informations en mémoire. |

## 🧪 Performance Vérifiée

Nous testons THL rigoureusement. Lancez la suite vous-même :
```bash
./scripts/run_tests.sh
```

## 📜 Licence

Ce projet est sous licence MIT.
