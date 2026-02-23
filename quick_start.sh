#!/bin/bash
# Script de démarrage rapide pour l'entraînement du classificateur

echo "=================================================="
echo "Pipeline Presence Detector - Quick Start"
echo "=================================================="
echo ""



# Analyser le dataset
echo "2. Analyse du dataset..."
 .env/bin/python analyze_dataset.py

if [ $? -ne 0 ]; then
    echo "❌ Erreur lors de l'analyse du dataset"
    exit 1
fi

echo ""
read -p "Voulez-vous lancer l'entraînement maintenant ? (o/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Oo]$ ]]; then
    echo ""
    echo "3. Démarrage de l'entraînement..."
    echo "   (Cela peut prendre plusieurs heures selon votre matériel)"
    echo ""
    
    .env/bin/python pipeline_presence_detector.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "=================================================="
        echo "✓ Entraînement terminé avec succès !"
        echo "=================================================="
        echo ""
        echo "Fichiers générés :"
        echo "  - best_pipeline_classifier.pth (modèle)"
        echo "  - training_history.png (graphiques)"
        echo "  - test_results.json (résultats)"
        echo ""
        echo "Pour faire une prédiction :"
        echo "   .env/bin/python predict_pipeline_presence.py --input your_file.npz"
    else
        echo ""
        echo "❌ Erreur lors de l'entraînement"
        exit 1
    fi
else
    echo ""
    echo "Entraînement annulé."
    echo "Pour lancer l'entraînement plus tard :"
    echo "   .env/bin/python pipeline_presence_detector.py"
fi
