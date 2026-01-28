"""
Script de test pour valider l'installation du projet RAG.

Ce script teste tous les composants du système pour s'assurer
que l'installation est correcte et que tous les modules fonctionnent.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Teste l'importation de tous les modules."""
    print("🔍 Test des importations...")
    
    try:
        # Test des dépendances externes
        import streamlit
        print("✅ Streamlit")
        
        import langchain
        print("✅ LangChain")
        
        import chromadb
        print("✅ ChromaDB")
        
        import groq
        print("✅ Groq")
        
        import sentence_transformers
        print("✅ Sentence Transformers")
        
        import PyPDF2
        print("✅ PyPDF2")
        
        import dotenv
        print("✅ Python-dotenv")
        
        # Test des modules du projet
        from src import config
        print("✅ Module config")
        
        from src import document_loader
        print("✅ Module document_loader")
        
        from src import vectorstore
        print("✅ Module vectorstore")
        
        from src import llm_handler
        print("✅ Module llm_handler")
        
        from src import rag_pipeline
        print("✅ Module rag_pipeline")
        
        return True
        
    except ImportError as e:
        print(f"❌ Erreur d'importation: {e}")
        return False

def test_configuration():
    """Teste la configuration du projet."""
    print("\n🔧 Test de la configuration...")
    
    try:
        from src.config import config
        
        # Test de la structure des dossiers
        if config.DOCUMENTS_DIR.exists():
            print("✅ Dossier documents")
        else:
            print("⚠️ Dossier documents manquant (sera créé automatiquement)")
        
        if config.CHROMA_PERSIST_DIR.exists():
            print("✅ Dossier chroma_db")
        else:
            print("⚠️ Dossier chroma_db manquant (sera créé automatiquement)")
        
        # Test de la clé API
        if config.GROQ_API_KEY:
            print("✅ Clé API Groq configurée")
        else:
            print("⚠️ Clé API Groq manquante (configurez-la dans .env)")
        
        # Affichage de la configuration
        print(f"📄 Taille chunks: {config.CHUNK_SIZE}")
        print(f"🔗 Overlap: {config.CHUNK_OVERLAP}")
        print(f"🔍 Top-K: {config.TOP_K_RETRIEVAL}")
        print(f"🤖 Modèle LLM: {config.LLM_MODEL}")
        print(f"🧠 Modèle embedding: {config.EMBEDDING_MODEL}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur de configuration: {e}")
        return False

def test_document_loader():
    """Teste le chargeur de documents."""
    print("\n📚 Test du chargeur de documents...")
    
    try:
        from src.document_loader import document_loader
        
        # Test du résumé des documents
        summary = document_loader.get_document_summary()
        print(f"📄 {summary['total_files']} document(s) trouvé(s)")
        
        if summary['total_files'] > 0:
            for file_info in summary['files']:
                if 'error' in file_info:
                    print(f"⚠️ {file_info['filename']}: {file_info['error']}")
                else:
                    print(f"✅ {file_info['filename']}: {file_info['pages']} pages")
        else:
            print("ℹ️ Aucun document PDF trouvé dans data/documents/")
            print("   Placez vos fichiers PDF dans ce dossier pour les tester")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur du chargeur de documents: {e}")
        return False

def test_llm_handler():
    """Teste le gestionnaire LLM."""
    print("\n🤖 Test du gestionnaire LLM...")
    
    try:
        from src.llm_handler import llm_handler
        
        # Test de connexion
        test_result = llm_handler.test_connection()
        
        if test_result['status'] == 'success':
            print("✅ Connexion Groq réussie")
            print(f"🤖 Modèle: {test_result['model']}")
            print(f"📝 Réponse test: {test_result['test_response']}")
        else:
            print(f"❌ Erreur de connexion: {test_result['message']}")
            return False
        
        # Informations du modèle
        model_info = llm_handler.get_model_info()
        if model_info['status'] == 'ready':
            print(f"✅ Modèle prêt: {model_info['model_name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur du gestionnaire LLM: {e}")
        return False

def test_vectorstore():
    """Teste le gestionnaire de base vectorielle."""
    print("\n🗄️ Test de la base vectorielle...")
    
    try:
        from src.vectorstore import vectorstore_manager
        
        # Test de l'initialisation des embeddings
        if vectorstore_manager.embeddings:
            print("✅ Modèle d'embedding initialisé")
        else:
            print("❌ Erreur d'initialisation des embeddings")
            return False
        
        # Informations sur la base vectorielle
        info = vectorstore_manager.get_vectorstore_info()
        print(f"📊 Statut: {info['status']}")
        
        if info['status'] == 'ready':
            print(f"📚 {info['document_count']} documents dans la base")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur de la base vectorielle: {e}")
        return False

def test_streamlit_app():
    """Teste l'application Streamlit."""
    print("\n🌐 Test de l'application Streamlit...")
    
    try:
        # Vérifier que app.py existe
        app_file = Path("app.py")
        if app_file.exists():
            print("✅ Fichier app.py trouvé")
        else:
            print("❌ Fichier app.py manquant")
            return False
        
        # Test d'importation de l'app
        import importlib.util
        spec = importlib.util.spec_from_file_location("app", "app.py")
        app_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_module)
        print("✅ Application Streamlit importable")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur de l'application Streamlit: {e}")
        return False

def main():
    """Fonction principale de test."""
    print("🧪 Test d'installation du projet RAG")
    print("=" * 50)
    
    tests = [
        ("Importations", test_imports),
        ("Configuration", test_configuration),
        ("Chargeur de documents", test_document_loader),
        ("Gestionnaire LLM", test_llm_handler),
        ("Base vectorielle", test_vectorstore),
        ("Application Streamlit", test_streamlit_app)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erreur dans le test {test_name}: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSÉ" if result else "❌ ÉCHEC"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Résultat: {passed}/{total} tests passés")
    
    if passed == total:
        print("🎉 Tous les tests sont passés ! Le projet est prêt à être utilisé.")
        print("\n🚀 Pour démarrer l'application:")
        print("   streamlit run app.py")
    else:
        print("⚠️ Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
        print("\n💡 Conseils de dépannage:")
        print("   1. Installez les dépendances: pip install -r requirements.txt")
        print("   2. Configurez votre clé API Groq dans le fichier .env")
        print("   3. Placez des fichiers PDF dans data/documents/")

if __name__ == "__main__":
    main()
