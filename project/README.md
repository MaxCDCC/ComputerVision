# Projet HPE — Triangulation 3D et reprojection

Ce dossier contient le script de triangulation et d'évaluation de reprojection pour le dataset annoté dans `hpe_04.coco`.

## Objectif

- calculer la position 3D d'un joueur à partir des caméras synchronisées
- reprojeter les points 3D dans chaque vue
- comparer la reprojection avec les annotations 2D

## Utilisation

1. Vérifier que le dataset et les fichiers de calibration existent :
   - `hpe_04.coco/train/_annotations.coco.json`
   - `material4project-20260603T095832Z-3-001/material4project/3D Pose Estimation Material/camera_data_with_Rvecs/camera_data`

2. Exécuter le script depuis la racine du workspace :

```bash
python project/triangulation.py \
  --annotations hpe_04.coco/train/_annotations.coco.json \
  --calib-root material4project-20260603T095832Z-3-001/material4project/3D\ Pose\ Estimation\ Material/camera_data_with_Rvecs/camera_data \
  --output project/triangulation_results.json
```

3. Résultats

- `project/triangulation_results.json` contient les positions 3D triangulées et les erreurs de reprojection.

## Notes

- le script utilise les vues `out1`, `out2`, `out3`, `out4`, `out5`, `out7`
- la correspondance entre vue et calibration est définie dans `triangulation.py`
- le script ne triangule les points que si le même point est visible dans au moins deux caméras
