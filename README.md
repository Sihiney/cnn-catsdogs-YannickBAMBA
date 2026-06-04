# CNN Cats vs Dogs — From Scratch vs Transfer Learning

**Auteur :** Yannick BAMBA
**Classe :** Master 1 IA
**Cours :** Deep Learning - DIT (Dakar Institute of Technology)  
**Période :** Juin 2026

---

## Objectif

Comparer deux approches de classification d'images (chats vs chiens) :
- **Expérience A** : CNN entraîné from scratch (architecture personnalisée)
- **Expérience B** : Transfer Learning (ResNet18 pré-entraîné ImageNet)

Montrer l'impact du transfer learning sur la convergence, la performance
et la robustesse par rapport à un entraînement from scratch.

---

## Résultats

| Modèle | Accuracy | Précision | Recall | Loss val |
|---|---|---|---|---|
| Scratch SGD | 0.734 | 0.741 | 0.722 | 0.560 |
| Scratch Adam | 0.777 | 0.752 | 0.826 | 0.481 |
| Transfer SGD | **0.992** | **0.992** | **0.993** | **0.020** |
| Transfer Adam | 0.990 | 0.993 | 0.987 | 0.024 |

**Conclusion principale :** Le transfer learning (ResNet18) atteint 98.7%
dès la première époque, là où le CNN from scratch démarre à 61.6%.
L'écart final est de +21.5 points d'accuracy.

---

## Environnement

### Prérequis

```bash
pip install -r requirements.txt
```

### requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
pandas>=2.0.0
```

### Matériel utilisé

- MacBook Pro Apple Silicon (M3 Pro)
- Accélérateur : MPS (Metal Performance Shaders)
- Python 3.12 / Anaconda

---

## Organisation des données

Le dataset utilisé est le jeu **Cats vs Dogs** (Kaggle).

### Structure attendue

```
Cat_Dog_data/
├── train/
│   ├── cat/     (11 250 images .jpg)
│   └── dog/     (11 250 images .jpg)
└── test/
    ├── cat/     (1 250 images .jpg)
    └── dog/     (1 250 images .jpg)
```

### Téléchargement

1. Télécharger le dataset sur [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats)
2. Extraire et placer le dossier `Cat_Dog_data/` à la racine du projet
3. **Ne pas pousser les données sur GitHub** (voir `.gitignore`)

> **Note :** Le dossier `test/` fourni est utilisé comme ensemble de validation
> pendant l'entraînement. Un split 80/10/10 serait préférable pour une
> évaluation plus rigoureuse - Nous avons identifié cela comme piste d'amélioration.

---

## Lancer l'entraînement

Ouvrir et exécuter **toutes les cellules** de `notebook.ipynb` dans l'ordre.

### Paramètres clés

| Paramètre | Valeur | Description |
|---|---|---|
| `SEED` | 42 | Seed de reproductibilité |
| `IMG_SIZE` | 224 | Taille des images (pixels) |
| `BATCH_SIZE` | 128 | Taille des mini-batches |
| `N_EPOCHS` | 15 | Nombre d'époques |
| `DEVICE` | mps/cuda/cpu | Détection automatique |

### Expérience A — CNN from scratch

- Architecture : 3 blocs Conv→BN→ReLU→MaxPool (32→64→128 canaux)
- Régularisation : Dropout2d(0.25) + BatchNorm après chaque bloc
- Optimiseurs testés : SGD (lr=0.01) et Adam (lr=1e-3)
- Scheduler : StepLR(step_size=5, gamma=0.5)

### Expérience B — Transfer Learning

- Base : ResNet18 pré-entraîné ImageNet (fine-tuning complet)
- Classifieur : Dropout(0.5) + Linear(512→2)
- Optimiseurs testés : SGD (lr=1e-3) et Adam (lr=1e-4)
- Scheduler : StepLR(step_size=5, gamma=0.5)

---

## Évaluation et rechargement du modèle

Les meilleurs checkpoints sont sauvegardés automatiquement pendant
l'entraînement :

```
scratch_sgd_best.pth   (val_acc=0.734)
scratch_adam_best.pth  (val_acc=0.777)
tl_sgd_best.pth        (val_acc=0.992)
tl_adam_best.pth       (val_acc=0.990)
```

Pour évaluer un modèle sauvegardé, exécuter la cellule 4.3 du notebook.
Elle recharge automatiquement le checkpoint et rapporte :
accuracy, précision, recall et matrice de confusion.

---

## Analyse des résultats

### Transfer Learning vs Scratch

ResNet18 part avec un avantage décisif grâce aux poids pré-entraînés
sur 1,2 million d'images ImageNet. Dès l'époque 1, il atteint 98.7%
d'accuracy là où le CNN from scratch démarre à 61.6% (niveau quasi
aléatoire). L'écart final est de +21.5 points.

### SGD vs Adam

Résultat contre-intuitif : SGD bat Adam en transfer learning (0.992 vs
0.990) alors qu'Adam bat SGD en scratch (0.777 vs 0.734). Explication :
les poids pré-entraînés sont déjà bien calibrés — Adam, avec son
adaptation dynamique du learning rate, risque de les déstabiliser.
SGD avec momentum est plus conservateur et préserve mieux la
connaissance acquise.

### Erreurs typiques

Sur 2 500 images de test, le meilleur modèle (Transfer SGD, 99.2%)
commet ~20 erreurs réparties en 3 catégories :
1. Images ambiguës (flou, angle inhabituel, gros plan)
2. Images multi-sujets (chat et chien dans le même cadre)
3. Images hors distribution (logos, dessins animés)

### Limites et pistes d'amélioration

- **Split train/val/test** (80/10/10) pour une évaluation plus rigoureuse
- **EfficientNet-B0** pour un meilleur ratio performance/paramètres
- **CosineAnnealingLR** pour une convergence plus douce
- **Nettoyage du dataset** (suppression des images corrompues/logos)
- **Journalisation TensorBoard** pour le suivi en temps réel

---

## Structure du dépôt

```
cnn-catsdogs-YannickBAMBA /
├── notebook.ipynb        # Notebook principal
├── requirements.txt      # Dépendances Python
├── .gitignore            # Exclusions Git
└── README.md             # Ce fichier
```

---

## GPU

L'entraînement utilise automatiquement :
- **CUDA** si GPU NVIDIA disponible
- **MPS** si Mac Apple Silicon (M1/M2/M3)
- **CPU** sinon

Détection automatique dans la cellule 0.3 du notebook.
Entraînement réalisé sur Mac M3 Pro (MPS) — ~175s par époque.

---

## Journalisation

Les métriques sont tracées via matplotlib dans le notebook (cellule 4.1).
L'intégration TensorBoard est identifiée comme piste d'amélioration
pour un suivi en temps réel lors de futurs entraînements.
