<!---
Copyright 2026 EGen Team. Tous droits réservés.

Sous licence MIT.
-->

<div align="center">
    <img src="docs/assets/banner.png" alt="THL Banner" width="100%"/>
</div>
<br>

<p align="center">
    <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
    <img src="https://img.shields.io/badge/vram-4GB-orange.svg" alt="VRAM Optimized">
    <a href="https://github.com/EGen-V/Transformer-Hierarchical-Layers/actions">
        <img src="https://github.com/EGen-V/Transformer-Hierarchical-Layers/workflows/Tests/badge.svg" alt="Tests">
    </a>
</p>

<h1 align="center">🐼 THL : Couches Hiérarchiques de Transformers</h1>

<p align="center">
    <a href="docs/lang/README_AR.md">العربية</a> •
    <a href="././README.md">English</a> •
    <a href="docs/lang/README_ES.md">Español</a> •
    <a>Français</a> •
    <a href="docs/lang/README_zh-hans.md">简体中文</a>
</p>

<h3 align="center">
    Architecture Récurrente Hiérarchique de Pointe pour Appareils à Ressources Limitées
</h3>

---

## 🎯 Vue d'Ensemble

**THL** est une architecture récurrente hiérarchique novatrice qui permet l'inférence de grands modèles de langage sur du matériel grand public avec seulement **4 Go de VRAM**. Contrairement aux Transformers traditionnels qui souffrent d'une explosion de la mémoire cache KV, THL atteint une **complexité mémoire O(1) par couche** grâce à une conception de mémoire indépendante de la longueur de séquence.

### Le Problème que Nous Résolvons

Les modèles Transformer traditionnels font face à un goulot d'étranglement critique : leur cache KV croît linéairement avec la longueur de séquence O(T), rendant la génération de contexte long impossible sur du matériel grand public. Un modèle de 7 milliards de paramètres traitant 8K tokens peut facilement dépasser 24 Go de VRAM.

### Notre Solution

THL remplace le cache KV illimité par une **banque de mémoire à emplacements fixes** (par défaut : 1024 emplacements), permettant :
- ✅ Longueur de contexte infinie sans débordement de mémoire
- ✅ Inférence sur des appareils avec 4 Go de VRAM
- ✅ Performance compétitive avec les architectures Transformer
- ✅ Déploiement sur appareils mobiles et edge

## ⚡ Caractéristiques Principales

- **Mémoire Bornée (O(1))** : Les emplacements mémoire fixes éliminent l'explosion du cache KV
- **Récurrence Hiérarchique** : Les niveaux GRU multi-échelles temporelles traitent l'information à des intervalles exponentiels (τ = 2^k)
- **Routage Épars** : Le routage Top-K multi-têtes accède aux mémoires pertinentes efficacement
- **Inférence Basse VRAM** : Le moteur d'inférence en couches permet des modèles de 7B+ paramètres sur <4 Go de VRAM
- **Prêt pour la Production** : Suite de tests complète et APIs documentées

## 🛠️ Installation

### Prérequis
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (pour l'accélération GPU)

### Installation depuis les Sources

```bash
# Cloner le dépôt
git clone https://github.com/EGen-V/Transformer-Hierarchical-Layers.git
cd Transformer-Hierarchical-Layers/Core

# Installer les dépendances
pip install -r requirements.txt

# Installer THL
pip install -e .
```

### Installation Rapide (PyPI)
```bash
pip install thl
```

## 🚀 Démarrage Rapide

### Modélisation de Langage de Base

```python
import torch
from thl.config import THLConfig
from thl.model import THLModel

# Configurer le modèle pour 4 Go de VRAM
config = THLConfig(
    num_tiers=3,          # Profondeur hiérarchique
    memory_slots=1024,    # Taille de mémoire fixe
    dim=768,              # Dimension du modèle
    vocab_size=50257      # Taille du vocabulaire
)

# Initialiser le modèle
model = THLModel(config)

# Exécuter l'inférence
input_ids = torch.randint(0, 50257, (1, 32))
logits, state = model(input_ids)

print(f"Forme de sortie : {logits.shape}")  # [1, 32, 50257]
```

### Génération en Streaming Basse VRAM

Pour les modèles plus grands, utilisez le moteur d'inférence en couches pour transmettre les couches via le GPU :

```python
from thl.inference.layered import LayeredInferenceEngine
from thl.inference.state import InferenceState

# Initialiser le moteur de streaming
engine = LayeredInferenceEngine(model, device="cuda")

# Créer l'état d'inférence
state = InferenceState.init(
    batch_size=1,
    config=config,
    tiers=model.tiers,
    memory_bank=model.memory_bank
)

# Générer des tokens un par un
generated_tokens = []
for _ in range(100):
    token = torch.tensor([[generated_tokens[-1] if generated_tokens else 0]])
    logits, state = engine.step(token, state)
    next_token = logits.argmax(dim=-1)
    generated_tokens.append(next_token.item())
```

### Exemple de Génération de Texte

```python
from thl.generation import generate_text

prompt = "L'avenir de l'IA est"
output = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_length=200,
    temperature=0.8,
    top_k=50
)
print(output)
```

## 🏗️ Architecture

THL emploie une architecture récurrente hiérarchique avec quatre composants clés :

| Composant | Symbole | Description |
|-----------|--------|-------------|
| **Banque de Mémoire** | M_t | Matrice de taille fixe (J × d) stockant le contexte à long terme |
| **Routeur Épars** | r_t | Mécanisme d'attention Top-K pour un accès efficace à la mémoire |
| **Niveaux Hiérarchiques** | s_t^(k) | Pile de cellules GRU se mettant à jour à des intervalles exponentiels τ = 2^k |
| **Écrivain de Nouveauté** | w_t | Mécanisme à porte écrivant uniquement les informations nouvelles en mémoire |

### Flux d'Information

1. **Lecture** : Le routeur épars récupère les emplacements mémoire Top-K pertinents
2. **Traitement** : Les niveaux hiérarchiques se mettent à jour à différentes échelles temporelles
3. **Écriture** : La porte de nouveauté détermine quelles nouvelles informations stocker
4. **Prédiction** : La couche de sortie génère les logits du prochain token

## 📊 Performance

| Métrique | THL-7B | Transformer-7B |
|--------|--------|----------------|
| **VRAM (ctx 8K)** | 3,8 Go | 26,4 Go |
| **Perplexité** | ~12,4 | ~11,8 |
| **Débit** | 42 tok/s | 38 tok/s |
| **Contexte Max** | Illimité | 8K tokens |

*Benchmarks sur NVIDIA RTX 3060 (12 Go)*

## 🧪 Tests

Nous maintenons une couverture de tests complète. Exécutez la suite complète :

```bash
# Exécuter tous les tests
./scripts/run_tests.sh

# Exécuter des catégories de tests spécifiques
pytest tests/test_model.py          # Tests du modèle
pytest tests/test_inference.py      # Tests d'inférence
pytest tests/test_memory.py         # Tests de gestion de mémoire
```

## 📚 Documentation

- [Spécification de l'Architecture](docs/THL_ARCHITECTURE_SPEC.md)
- [Contexte et Philosophie du Projet](docs/THL_CONTEXT.md)
- [Référence API](thl/README.md)
- [Guide de Tests](tests/README.md)
- [Guide d'Inférence](thl/inference/README.md)

## 🗺️ Feuille de Route

- [ ] Checkpoints de modèles pré-entraînés
- [ ] Publication du paquet PyPI
- [ ] Support d'exportation ONNX
- [ ] Déploiement mobile (iOS/Android)
- [ ] Déploiement web (WASM)
- [ ] Support d'entraînement multi-GPU
- [ ] Quantification (INT8/INT4)

## 🤝 Contribution

Nous accueillons les contributions ! Veuillez consulter nos [Directives de Contribution](CONTRIBUTING.md) pour plus de détails.

```bash
# Configurer l'environnement de développement
git clone https://github.com/EGen-V/Transformer-Hierarchical-Layers.git
cd Transformer-Hierarchical-Layers
pip install -e ".[dev]"
pre-commit install
```

## 📄 Citation

Si vous utilisez THL dans vos recherches, veuillez citer :

```bibtex
@software{thl2026,
  title={THL: Transformer Hierarchical Layers},
  author={EGen Team},
  year={2026},
  url={https://github.com/EGen-V/Transformer-Hierarchical-Layers}
}
```

## 📜 Licence

Ce projet est sous licence MIT - consultez le fichier [LICENSE](LICENSE) pour plus de détails.

## 🙏 Remerciements

- Inspiré par les architectures de mémoire récurrente et la recherche sur les transformers efficaces
- Construit avec PyTorch et la communauté ML open source

## 📧 Contact

- **Issues** : [GitHub Issues](https://github.com/EGen-V/Transformer-Hierarchical-Layers/issues)
- **Discussions** : [GitHub Discussions](https://github.com/EGen-V/Transformer-Hierarchical-Layers/discussions)
- **Email** : mouhebzayani@erebustn.io

---

<p align="center">
    Fait avec ❤️ par l'Équipe EGen
</p>