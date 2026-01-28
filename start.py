"""
Script de démarrage rapide pour le projet RAG.

Ce script facilite le démarrage de l'application en vérifiant
la configuration et en lançant Streamlit automatiquement.
"""

import os
import sys
import subprocess
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError


def check_environment():
    """Vérifie l'environnement et la configuration."""
    print("🔍 Vérification de l'environnement...")
    
    # Vérifier Python
    python_version = sys.version_info
    if python_version < (3, 8):
        print("❌ Python 3.8+ requis")
        return False
    print(f"✅ Python {python_version.major}.{python_version.minor}")
    
    # Vérifier le fichier .env
    env_file = Path("mon_rag_project/.env")
    if not env_file.exists():
        print("⚠️ Fichier .env manquant")
        print("   Copiez .env.example vers .env et configurez votre clé API Groq")
        return False
    print("✅ Fichier .env trouvé")
    
    # Vérifier la clé API
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_key_here":
            print("⚠️ Clé API Groq non configurée")
            print("   Configurez GROQ_API_KEY dans le fichier .env")
            return False
        print("✅ Clé API Groq configurée")
    except ImportError:
        print("❌ python-dotenv non installé")
        return False
    
    return True

def check_dependencies():
    """Vérifie que les dépendances sont installées."""
    print("\n📦 Vérification des dépendances...")
    
    required_packages = [
        "streamlit", "langchain", "chromadb", "groq",
        "sentence_transformers", "PyPDF2", "python-dotenv"
    ]

    missing_packages = []
    for pkg in required_packages:
        try:
            v = version(pkg)
            print(f"✅ {pkg} ({v})")
        except PackageNotFoundError:
            print(f"❌ {pkg}")
            missing_packages.append(pkg)
    
    if missing_packages:
        print(f"\n⚠️ Packages manquants: {', '.join(missing_packages)}")
        print("   Installez avec: pip install -r requirements.txt")
        return False
    
    return True


def check_documents():
    """Vérifie la présence de documents PDF."""
    print("\n📚 Vérification des documents...")
    
    documents_dir = Path("mon_rag_project/data/documents")
    if not documents_dir.exists():
        documents_dir.mkdir(parents=True, exist_ok=True)
        print("✅ Dossier documents créé")
    
    pdf_files = list(documents_dir.glob("*.pdf"))
    if pdf_files:
        print(f"✅ {len(pdf_files)} document(s) PDF trouvé(s)")
        for pdf_file in pdf_files:
            print(f"   📄 {pdf_file.name}")
    else:
        print("⚠️ Aucun document PDF trouvé")
        print("   Placez vos fichiers PDF dans data/documents/")
        print("   L'application fonctionnera mais sans contenu à analyser")
    
    return True

def start_streamlit():
    """Démarre l'application Streamlit."""
    print("\n🚀 Démarrage de l'application Streamlit...")

    try:
        # chemin absolu vers app.py (même dossier que ce start.py)
        app_path = Path(__file__).resolve().parent / "app.py"
        if not app_path.exists():
            print(f"❌ Fichier app.py manquant : {app_path}")
            return False

        print("🌐 Ouverture de l'application dans le navigateur...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)

    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors du démarrage de Streamlit: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️ Arrêt de l'application")
        return True

    return True


def main():
    """Fonction principale."""
    print("🚀 DÉMARRAGE DU PROJET RAG")
    print("=" * 40)
    
    # Vérifications
    if not check_environment():
        print("\n❌ Problème de configuration détecté")
        print("   Corrigez les erreurs avant de continuer")
        return
    
    if not check_dependencies():
        print("\n❌ Dépendances manquantes")
        print("   Installez avec: pip install -r requirements.txt")
        return
    
    check_documents()
    
    # Demander confirmation
    print("\n" + "=" * 40)
    response = input("Voulez-vous démarrer l'application ? (o/N): ").strip().lower()
    
    if response in ['o', 'oui', 'y', 'yes']:
        start_streamlit()
    else:
        print("👋 Démarrage annulé")
        print("\n💡 Pour démarrer manuellement:")
        print("   streamlit run app.py")

if __name__ == "__main__":
    main()
