# Rapport technique — Module RGB pour la détection de deepfakes

## 1. Introduction

Ce document décrit le module **RGBStreamResNet**, développé dans le cadre d’un système de détection de deepfakes.  
Ce module constitue la branche **visuelle (spatiale)** du système global, qui sera ensuite combiné avec un module fréquentiel (FFT) dans une architecture de fusion.

L’objectif principal est d’extraire des caractéristiques visuelles pertinentes à partir d’images de visages afin de distinguer les contenus réels des contenus générés ou manipulés.

---

## 2. Rôle du module RGB

Le module RGB a pour fonction de transformer une image en une représentation numérique compacte appelée **vecteur de caractéristiques (embedding)**.

### Ce que le module apprend :

- textures de peau et détails fins du visage
- formes et structures faciales
- incohérences visuelles (artefacts de génération)
- variations d’éclairage et de compression

### Sorties possibles :

- un vecteur de caractéristiques de taille **512 (features vector)**
- une prédiction binaire **REAL / FAKE**

---

## 3. Architecture du modèle

Le modèle est basé sur **ResNet-18**, un réseau de neurones convolutionnel pré-entraîné.

### Modification principale :

- Suppression de la couche de classification originale (1000 classes)
- Remplacement par une couche personnalisée :

```text
Dropout → Linear(in_features → 512)
```
