"""
Script de déploiement et maintenance pour le projet RAG.

Ce script facilite le déploiement, la maintenance et la mise à jour
du système RAG en production.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any

def create_environment():
    """Crée un environnement virtuel pour le projet."""
    print("🐍 Création de l'environnement virtuel...")
    
    venv_path = Path("venv")
    if venv_path.exists():
        print("⚠️ Environnement virtuel existant trouvé")
        response = input("Voulez-vous le recréer ? (o/N): ").strip().lower()
        if response in ['o', 'oui', 'y', 'yes']:
            shutil.rmtree(venv_path)
        else:
            print("✅ Utilisation de l'environnement existant")
            return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Environnement virtuel créé")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de la création de l'environnement: {e}")
        return False

def install_dependencies():
    """Installe les dépendances du projet."""
    print("📦 Installation des dépendances...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ Fichier requirements.txt manquant")
        return False
    
    try:
        # Déterminer le chemin de pip selon l'OS
        if os.name == 'nt':  # Windows
            pip_path = Path("venv/Scripts/pip")
        else:  # Linux/Mac
            pip_path = Path("venv/bin/pip")
        
        subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
        print("✅ Dépendances installées")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'installation: {e}")
        return False

def setup_environment_file():
    """Configure le fichier d'environnement."""
    print("🔧 Configuration du fichier .env...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✅ Fichier .env existant trouvé")
        return True
    
    if not env_example.exists():
        print("❌ Fichier .env.example manquant")
        return False
    
    # Copier le fichier d'exemple
    shutil.copy(env_example, env_file)
    print("✅ Fichier .env créé depuis .env.example")
    
    print("\n⚠️ IMPORTANT: Configurez votre clé API Groq dans le fichier .env")
    print("   1. Ouvrez le fichier .env")
    print("   2. Remplacez 'your_key_here' par votre vraie clé API Groq")
    print("   3. Obtenez votre clé sur: https://console.groq.com/")
    
    return True

def create_directories():
    """Crée les répertoires nécessaires."""
    print("📁 Création des répertoires...")
    
    directories = [
        "data/documents",
        "chroma_db",
        "logs"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")
    
    return True

def run_tests():
    """Exécute les tests d'installation."""
    print("🧪 Exécution des tests...")
    
    test_file = Path("test_installation.py")
    if not test_file.exists():
        print("⚠️ Fichier de test manquant")
        return True
    
    try:
        # Déterminer le chemin de python selon l'OS
        if os.name == 'nt':  # Windows
            python_path = Path("venv/Scripts/python")
        else:  # Linux/Mac
            python_path = Path("venv/bin/python")
        
        result = subprocess.run([str(python_path), "test_installation.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Tests passés avec succès")
            print(result.stdout)
        else:
            print("⚠️ Certains tests ont échoué")
            print(result.stdout)
            print(result.stderr)
        
        return True
    except Exception as e:
        print(f"❌ Erreur lors des tests: {e}")
        return False

def cleanup_old_data():
    """Nettoie les anciennes données si nécessaire."""
    print("🧹 Nettoyage des anciennes données...")
    
    chroma_db_path = Path("chroma_db")
    if chroma_db_path.exists():
        response = input("Voulez-vous nettoyer la base vectorielle existante ? (o/N): ").strip().lower()
        if response in ['o', 'oui', 'y', 'yes']:
            shutil.rmtree(chroma_db_path)
            chroma_db_path.mkdir()
            print("✅ Base vectorielle nettoyée")
        else:
            print("✅ Base vectorielle conservée")
    
    return True

def show_startup_instructions():
    """Affiche les instructions de démarrage."""
    print("\n" + "=" * 60)
    print("🚀 PROJET RAG DÉPLOYÉ AVEC SUCCÈS")
    print("=" * 60)
    
    print("\n📋 Instructions de démarrage:")
    
    print("\n1. 🐍 Activer l'environnement virtuel:")
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Linux/Mac
        print("   source venv/bin/activate")
    
    print("\n2. 🔑 Configurer la clé API Groq:")
    print("   - Ouvrez le fichier .env")
    print("   - Remplacez 'your_key_here' par votre clé API")
    print("   - Obtenez votre clé sur: https://console.groq.com/")
    
    print("\n3. 📚 Ajouter des documents PDF:")
    print("   - Placez vos fichiers PDF dans data/documents/")
    print("   - L'application les détectera automatiquement")
    
    print("\n4. 🌐 Lancer l'application:")
    print("   streamlit run app.py")
    
    print("\n5. 🧪 Ou utiliser le script de démarrage rapide:")
    print("   python start.py")
    
    print("\n💡 Conseils:")
    print("   - Consultez le README.md pour plus d'informations")
    print("   - Utilisez python test_installation.py pour vérifier l'installation")
    print("   - L'application sera accessible sur http://localhost:8501")

def main():
    """Fonction principale de déploiement."""
    print("🚀 DÉPLOIEMENT DU PROJET RAG")
    print("=" * 40)
    
    steps = [
        ("Création de l'environnement virtuel", create_environment),
        ("Installation des dépendances", install_dependencies),
        ("Configuration du fichier .env", setup_environment_file),
        ("Création des répertoires", create_directories),
        ("Nettoyage des anciennes données", cleanup_old_data),
        ("Exécution des tests", run_tests)
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        try:
            if not step_func():
                print(f"❌ Échec: {step_name}")
                return
        except Exception as e:
            print(f"❌ Erreur lors de {step_name}: {e}")
            return
    
    show_startup_instructions()

if __name__ == "__main__":
    main()
