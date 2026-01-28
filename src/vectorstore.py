"""
Module de gestion de la base de données vectorielle ChromaDB.

Ce module gère la création, le chargement et la persistance de la base
vectorielle ChromaDB utilisée pour stocker les embeddings des documents.
"""

import logging
import time
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document
# ⚠️ Utiliser le package dédié (évite le warning de dépréciation)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from .config import config

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Gestionnaire de la base de données vectorielle ChromaDB.

    Cette classe gère la création, le chargement et la persistance
    des embeddings dans ChromaDB avec une interface LangChain.
    """

    def __init__(self):
        """Initialise le gestionnaire de base vectorielle."""
        self.embeddings = None
        self.vectorstore: Optional[Chroma] = None
        self.collection_name = "rag_documents"
        self._initialize_embeddings()

    # --------------------------- Embeddings ---------------------------

    def _initialize_embeddings(self) -> None:
        """
        Initialise le modèle d'embedding.

        Raises:
            Exception: Si le modèle d'embedding ne peut pas être chargé.
        """
        try:
            logger.info(f"Chargement du modèle d'embedding: {config.EMBEDDING_MODEL}")

            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},  # Force CPU pour compatibilité
                encode_kwargs={"normalize_embeddings": True},
            )

            logger.info("✅ Modèle d'embedding chargé avec succès")

        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle d'embedding: {e}")
            raise Exception(config.ERROR_MESSAGES["embedding_error"])

    # --------------------------- Utils internes ---------------------------

    def _vectorstore_exists(self) -> bool:
        """
        Vérifie si une base vectorielle existe déjà.

        Returns:
            bool: True si la base existe, False sinon.
        """
        try:
            if not config.CHROMA_PERSIST_DIR.exists():
                return False
            chroma_files = list(config.CHROMA_PERSIST_DIR.glob("*"))
            return len(chroma_files) > 0
        except Exception:
            return False

    def _ensure_loaded(self) -> None:
        """
        Recharge la base si l'instance en mémoire est absente.
        Utile après un rerun Streamlit (les singletons sont réinitialisés).
        """
        if self.vectorstore is None and self._vectorstore_exists():
            try:
                self.load_existing_vectorstore()
            except Exception as e:
                logger.warning(f"Impossible de recharger la base vectorielle: {e}")

    # --------------------------- Création / Chargement ---------------------------

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Crée une nouvelle base vectorielle à partir des documents.

        Args:
            documents (List[Document]): Liste des documents à indexer.

        Returns:
            Chroma: Base vectorielle ChromaDB.

        Raises:
            Exception: Si la création échoue.
        """
        try:
            logger.info(f"Création de la base vectorielle avec {len(documents)} documents...")

            # S'assurer que le répertoire de persistance existe
            config.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

            # ⚠️ Création + indexation à partir des documents
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,  # bon paramètre
                persist_directory=str(config.CHROMA_PERSIST_DIR),
                collection_name=self.collection_name,
                collection_metadata={"hnsw:space": "cosine"},
            )

            # Avec Chroma récent, la persistance est automatique, mais on garde l’appel
            try:
                self.vectorstore.persist()
            except Exception:
                # Certaines versions loggent un warning si persist() est inutile
                pass

            logger.info(config.SUCCESS_MESSAGES["embeddings_created"])
            logger.info(f"Base vectorielle sauvegardée dans: {config.CHROMA_PERSIST_DIR}")

            # Log de vérification
            try:
                count = self.vectorstore._collection.count()
                logger.info(f"📊 Chroma count (après création): {count}")
            except Exception:
                pass

            return self.vectorstore

        except Exception as e:
            logger.exception("Erreur lors de la création de la base vectorielle")
            raise Exception(config.ERROR_MESSAGES["chroma_error"])

    def load_existing_vectorstore(self) -> Optional[Chroma]:
        """
        Charge une base vectorielle existante depuis le disque.

        Returns:
            Optional[Chroma]: Base vectorielle chargée ou None si elle n'existe pas.

        Raises:
            Exception: Si le chargement échoue.
        """
        try:
            if not self._vectorstore_exists():
                logger.info("Aucune base vectorielle existante trouvée")
                return None

            logger.info("Chargement de la base vectorielle existante...")

            # ⚠️ Chargement sans documents (réattache la collection persistée)
            self.vectorstore = Chroma(
                persist_directory=str(config.CHROMA_PERSIST_DIR),
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
                collection_metadata={"hnsw:space": "cosine"},
            )

            # Vérifier que la base contient des documents
            doc_count = self.vectorstore._collection.count()
            if doc_count == 0:
                logger.warning("La base vectorielle existe mais est vide")
                return None

            logger.info(f"✅ Base vectorielle chargée avec {doc_count} documents")
            return self.vectorstore

        except Exception as e:
            logger.exception("Erreur lors du chargement de la base vectorielle")
            raise Exception(config.ERROR_MESSAGES["chroma_error"])

    def get_or_create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Récupère une base vectorielle existante ou en crée une nouvelle.

        Args:
            documents (List[Document]): Documents à utiliser si création nécessaire.

        Returns:
            Chroma: Base vectorielle prête à l'emploi.
        """
        try:
            existing_vectorstore = self.load_existing_vectorstore()

            if existing_vectorstore is not None:
                self.vectorstore = existing_vectorstore
                logger.info(config.SUCCESS_MESSAGES["vectorstore_ready"])
                return self.vectorstore

            logger.info("Création d'une nouvelle base vectorielle...")
            return self.create_vectorstore(documents)

        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de la base vectorielle: {e}")
            raise

    # --------------------------- Recherches ---------------------------

    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """
        Effectue une recherche de similarité dans la base vectorielle.
        """
        self._ensure_loaded()
        if self.vectorstore is None:
            raise Exception("Base vectorielle non initialisée")

        try:
            k = k or config.TOP_K_RETRIEVAL
            logger.info(f"Recherche de similarité pour: '{query[:50]}...'")
            results = self.vectorstore.similarity_search(query=query, k=k)
            logger.info(f"Trouvé {len(results)} document(s) similaire(s)")
            return results

        except Exception as e:
            logger.error(f"Erreur lors de la recherche de similarité: {e}")
            raise Exception(f"Erreur de recherche: {str(e)}")

    def similarity_search_with_score(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """
        Effectue une recherche de similarité avec scores.
        Retourne des tuples (Document, distance).
        """
        self._ensure_loaded()
        if self.vectorstore is None:
            raise Exception("Base vectorielle non initialisée")

        try:
            k = k or config.TOP_K_RETRIEVAL
            logger.info(f"Recherche de similarité avec scores pour: '{query[:50]}...'")
            results = self.vectorstore.similarity_search_with_score(query=query, k=k)
            logger.info(f"Trouvé {len(results)} document(s) similaire(s)")
            return results

        except Exception as e:
            logger.error(f"Erreur lors de la recherche de similarité avec scores: {e}")
            raise Exception(f"Erreur de recherche: {str(e)}")

    # --------------------------- Infos / Maintenance ---------------------------

    def get_vectorstore_info(self) -> Dict[str, Any]:
        """
        Récupère les informations sur la base vectorielle.
        """
        self._ensure_loaded()
        if self.vectorstore is None:
            return {
                "status": "not_initialized",
                "document_count": 0,
                "collection_name": self.collection_name,
            }

        try:
            doc_count = self.vectorstore._collection.count()
            logger.info(f"📊 Chroma count (info): {doc_count}")
            return {
                "status": "ready",
                "document_count": doc_count,
                "collection_name": self.collection_name,
                "persist_directory": str(config.CHROMA_PERSIST_DIR),
                "embedding_model": config.EMBEDDING_MODEL,
            }

        except Exception as e:
            logger.error(f"Erreur lors de la récupération des infos: {e}")
            return {"status": "error", "error": str(e), "document_count": 0}

    def delete_vectorstore(self) -> bool:
        """
        Supprime la base vectorielle existante.
        (avec petite tolérance Windows si le fichier est verrouillé)
        """
        try:
            if not self._vectorstore_exists():
                logger.info("Aucune base vectorielle à supprimer")
                return True

            logger.info("Suppression de la base vectorielle...")

            # Libérer l'instance en mémoire
            self.vectorstore = None

            # Réessayer la suppression si le fichier est temporairement verrouillé (Windows)
            for attempt in range(3):
                try:
                    shutil.rmtree(config.CHROMA_PERSIST_DIR)
                    logger.info("✅ Base vectorielle supprimée")
                    return True
                except Exception as e:
                    logger.warning(f"Suppression bloquée (tentative {attempt+1}/3): {e}")
                    time.sleep(0.8)

            logger.error("Échec de la suppression après plusieurs tentatives")
            return False

        except Exception as e:
            logger.error(f"Erreur lors de la suppression: {e}")
            return False

    def refresh_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Recrée complètement la base vectorielle avec de nouveaux documents.
        """
        try:
            logger.info("Rafraîchissement de la base vectorielle...")
            self.delete_vectorstore()
            return self.create_vectorstore(documents)

        except Exception as e:
            logger.error(f"Erreur lors du rafraîchissement: {e}")
            raise


# Instance globale du gestionnaire
vectorstore_manager = VectorStoreManager()


if __name__ == "__main__":
    # Test du module
    print("Test du module VectorStoreManager...")

    try:
        info = vectorstore_manager.get_vectorstore_info()
        print(f"📊 Statut de la base vectorielle: {info['status']}")

        if info["status"] == "ready":
            print(f"📚 {info['document_count']} documents dans la base")
            print(f"🗄️ Collection: {info['collection_name']}")
            print(f"🧠 Modèle: {info['embedding_model']}")

        # Test de recherche si la base existe
        if vectorstore_manager.vectorstore is not None:
            print("\n🔍 Test de recherche...")
            results = vectorstore_manager.similarity_search("test", k=2)
            print(f"Résultats trouvés: {len(results)}")

    except Exception as e:
        print(f"❌ Erreur: {e}")
