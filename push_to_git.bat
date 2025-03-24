@echo off
cd /d "C:\Users\vince\Documents\Université PSL\Paris_Dauphine-PSL\Apprentissage_supervisé\Projet\donnees_projet_apprentissage_supervise"
echo 📁 Déplacement dans le dossier des données...

git add .
echo ✅ Tous les fichiers de données ajoutés à Git.

git commit -m "💾 Mise à jour automatique des fichiers de données"
echo ✅ Commit effectué.

git push origin main
echo 🚀 Données envoyées sur le dépôt GitHub public !

pause