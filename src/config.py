"""
Configuration du projet RAG.

Ce module centralise tous les paramètres de configuration du système RAG,
incluant les chemins de fichiers, les modèles utilisés, et les paramètres
de traitement des documents.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# Charger les variables d'environnement
load_dotenv()


class Config:
    """Configuration centralisée du projet RAG."""
    
    # ==================== Chemins de fichiers ====================
    # Répertoire racine du projet
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # Répertoire des documents PDF
    DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
    
    # Répertoire de persistance ChromaDB
    CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"
    
    # ==================== Configuration API ====================
    # Clé API Groq (obligatoire)
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    
    # Modèle LLM à utiliser
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

    
    # ==================== Configuration des embeddings ====================
    # Modèle d'embedding (léger et efficace)
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", 
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # ==================== Configuration du découpage de documents ====================
    # Taille des chunks (caractères)
    CHUNK_SIZE: int = 1000
    
    # Overlap entre chunks (caractères)
    CHUNK_OVERLAP: int = 200
    
    # ==================== Configuration de la récupération ====================
    # Nombre de documents similaires à récupérer
    TOP_K_RETRIEVAL: int = 4
    
    # Seuil de similarité minimum (optionnel)
    SIMILARITY_THRESHOLD: float = 0.7
    
    # ==================== Configuration Streamlit ====================
    # Titre de l'application
    APP_TITLE: str = "🔬 RAG - Recherche dans Documents Scientifiques"
    
    # Description de l'application
    APP_DESCRIPTION: str = """
    Posez des questions sur vos documents PDF scientifiques et obtenez des réponses 
    précises avec les sources citées.
    """
    
    # ==================== Messages d'erreur ====================
    ERROR_MESSAGES = {
        "missing_api_key": "❌ Clé API Groq manquante. Veuillez configurer GROQ_API_KEY dans le fichier .env",
        "documents_not_found": "❌ Aucun document PDF trouvé dans le dossier data/documents/",
        "chroma_error": "❌ Erreur lors de la création/chargement de la base vectorielle",
        "llm_error": "❌ Erreur lors de la génération de la réponse",
        "embedding_error": "❌ Erreur lors de la création des embeddings"
    }
    
    # ==================== Messages de succès ====================
    SUCCESS_MESSAGES = {
        "documents_loaded": "✅ Documents chargés avec succès",
        "embeddings_created": "✅ Embeddings créés et sauvegardés",
        "vectorstore_ready": "✅ Base vectorielle prête"
    }
    
    @classmethod
    def validate_config(cls) -> bool:
        """
        Valide la configuration du projet.
        
        Returns:
            bool: True si la configuration est valide, False sinon.
        """
        # Vérifier la présence de la clé API
        if not cls.GROQ_API_KEY:
            print(cls.ERROR_MESSAGES["missing_api_key"])
            return False
        
        # Créer les répertoires nécessaires
        cls.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        
        return True
    
    @classmethod
    def get_document_files(cls) -> list[Path]:
        """
        Récupère la liste des fichiers PDF dans le répertoire des documents.
        
        Returns:
            list[Path]: Liste des chemins vers les fichiers PDF.
        """
        if not cls.DOCUMENTS_DIR.exists():
            return []
        
        # Rechercher tous les fichiers PDF
        pdf_files = list(cls.DOCUMENTS_DIR.glob("*.pdf"))
        return pdf_files
    
    @classmethod
    def print_config(cls) -> None:
        """Affiche la configuration actuelle."""
        print("🔧 Configuration du projet RAG:")
        print(f"  📁 Documents: {cls.DOCUMENTS_DIR}")
        print(f"  🗄️ ChromaDB: {cls.CHROMA_PERSIST_DIR}")
        print(f"  🤖 Modèle LLM: {cls.LLM_MODEL}")
        print(f"  🧠 Modèle Embedding: {cls.EMBEDDING_MODEL}")
        print(f"  📄 Taille chunks: {cls.CHUNK_SIZE} caractères")
        print(f"  🔗 Overlap: {cls.CHUNK_OVERLAP} caractères")
        print(f"  🔍 Top-K: {cls.TOP_K_RETRIEVAL}")
        print(f"  🔑 API Key: {'✅ Configurée' if cls.GROQ_API_KEY else '❌ Manquante'}")


# Instance globale de configuration
config = Config()


if __name__ == "__main__":
    # Test de la configuration
    print("Test de la configuration...")
    config.print_config()
    
    if config.validate_config():
        print("✅ Configuration valide")
        documents = config.get_document_files()
        print(f"📚 {len(documents)} document(s) trouvé(s)")
        for doc in documents:
            print(f"  - {doc.name}")
    else:
        print("❌ Configuration invalide")
