"""
Module principal du pipeline RAG.

Ce module orchestre l'ensemble du processus RAG en combinant le chargement
des documents, la création d'embeddings, la recherche vectorielle et la
génération de réponses avec le LLM.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
#from langchain.schema import Document
from langchain_core.documents import Document


from .config import config
from .document_loader import document_loader
from .vectorstore import vectorstore_manager
from .llm_handler import llm_handler

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Pipeline principal RAG (Retrieval-Augmented Generation).
    
    Cette classe orchestre l'ensemble du processus RAG en gérant :
    - Le chargement et le traitement des documents
    - La création et la gestion de la base vectorielle
    - La recherche de documents pertinents
    - La génération de réponses avec le LLM
    """
    
    def __init__(self):
        """Initialise le pipeline RAG."""
        self.is_initialized = False
        self.documents_loaded = False
        self.vectorstore_ready = False
        
    def initialize(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Initialise le pipeline RAG complet.
        
        Args:
            force_refresh (bool): Force la recréation de la base vectorielle.
            
        Returns:
            Dict[str, Any]: Statut de l'initialisation avec détails.
        """
        try:
            logger.info("🚀 Initialisation du pipeline RAG...")
            
            # Validation de la configuration
            if not config.validate_config():
                raise Exception("Configuration invalide")
            
            # Charger les documents
            documents = self._load_documents()
            self.documents_loaded = True
            
            # Initialiser la base vectorielle
            self._initialize_vectorstore(documents, force_refresh)
            self.vectorstore_ready = True
            
            # Tester le LLM
            self._test_llm()
            
            self.is_initialized = True
            
            logger.info("✅ Pipeline RAG initialisé avec succès")
            
            return {
                'status': 'success',
                'message': 'Pipeline RAG initialisé',
                'documents_count': len(documents),
                'vectorstore_ready': True,
                'llm_ready': True
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'documents_count': 0,
                'vectorstore_ready': False,
                'llm_ready': False
            }
    
    def _load_documents(self) -> List[Document]:
        """
        Charge tous les documents PDF disponibles.
        
        Returns:
            List[Document]: Liste des documents chargés et découpés.
            
        Raises:
            Exception: Si le chargement échoue.
        """
        try:
            logger.info("📚 Chargement des documents...")
            documents = document_loader.load_all_documents()
            
            if not documents:
                raise Exception("Aucun document n'a pu être chargé")
            
            logger.info(f"✅ {len(documents)} chunks de documents chargés")
            return documents
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des documents: {e}")
            raise
    
    def _initialize_vectorstore(self, documents: List[Document], force_refresh: bool = False) -> None:
        """
        Initialise la base vectorielle.
        
        Args:
            documents (List[Document]): Documents à indexer.
            force_refresh (bool): Force la recréation.
            
        Raises:
            Exception: Si l'initialisation échoue.
        """
        try:
            logger.info("🗄️ Initialisation de la base vectorielle...")
            
            if force_refresh:
                logger.info("🔄 Rafraîchissement forcé de la base vectorielle")
                vectorstore_manager.refresh_vectorstore(documents)
            else:
                vectorstore_manager.get_or_create_vectorstore(documents)
            
            logger.info("✅ Base vectorielle prête")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de la base vectorielle: {e}")
            raise
    
    def _test_llm(self) -> None:
        """
        Teste la connexion avec le LLM.
        
        Raises:
            Exception: Si le test échoue.
        """
        try:
            logger.info("🤖 Test du LLM...")
            test_result = llm_handler.test_connection()
            
            if test_result['status'] != 'success':
                raise Exception(f"Test LLM échoué: {test_result['message']}")
            
            logger.info("✅ LLM opérationnel")
            
        except Exception as e:
            logger.error(f"Erreur lors du test du LLM: {e}")
            raise
    
    def ask_question(self, question: str, include_sources: bool = True) -> Dict[str, Any]:
        """
        Pose une question au système RAG.
        
        Args:
            question (str): Question à poser.
            include_sources (bool): Inclure les sources dans la réponse.
            
        Returns:
            Dict[str, Any]: Réponse avec sources et métadonnées.
            
        Raises:
            Exception: Si le pipeline n'est pas initialisé ou si une erreur survient.
        """
        if not self.is_initialized:
            raise Exception("Pipeline RAG non initialisé. Appelez initialize() d'abord.")
        
        try:
            logger.info(f"❓ Question: {question[:100]}...")
            
            # Étape 1: Recherche de documents pertinents
            relevant_docs = self._retrieve_relevant_documents(question)
            
            if not relevant_docs:
                return {
                    'answer': "Désolé, je n'ai pas trouvé d'informations pertinentes dans les documents pour répondre à votre question.",
                    'sources': [],
                    'metadata': {
                        'question': question,
                        'documents_found': 0,
                        'confidence': 0.0
                    }
                }
            
            # Étape 2: Génération de la réponse
            answer = self._generate_answer(question, relevant_docs)
            
            # Étape 3: Préparation des sources
            sources = self._prepare_sources(relevant_docs) if include_sources else []
            
            # Étape 4: Calcul de la confiance
            confidence = self._calculate_confidence(relevant_docs)
            
            logger.info(f"✅ Réponse générée (confiance: {confidence:.2f})")
            
            return {
                'answer': answer,
                'sources': sources,
                'metadata': {
                    'question': question,
                    'documents_found': len(relevant_docs),
                    'confidence': confidence,
                    'answer_length': len(answer)
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la réponse: {e}")
            raise Exception(f"Erreur de génération: {str(e)}")
    
    def _retrieve_relevant_documents(self, question: str) -> List[Document]:
        """
        Récupère les documents les plus pertinents pour une question.
        
        Args:
            question (str): Question de l'utilisateur.
            
        Returns:
            List[Document]: Documents les plus pertinents.
        """
        try:
            # Recherche avec scores pour évaluer la pertinence
            results_with_scores = vectorstore_manager.similarity_search_with_score(
                question, 
                k=config.TOP_K_RETRIEVAL
            )
            
            # Filtrer par seuil de similarité si configuré
            filtered_results = []
            for doc, score in results_with_scores:
                # Convertir la distance en similarité (ChromaDB utilise la distance)
                similarity = 1 / (1 + score)
                
                if similarity >= config.SIMILARITY_THRESHOLD:
                    filtered_results.append(doc)
                else:
                    logger.info(f"Document filtré (similarité: {similarity:.3f} < {config.SIMILARITY_THRESHOLD})")
            
            # Retourner au moins le document le plus pertinent même si sous le seuil
            if not filtered_results and results_with_scores:
                filtered_results = [results_with_scores[0][0]]
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des documents: {e}")
            return []
    
    def _generate_answer(self, question: str, relevant_docs: List[Document]) -> str:
        """
        Génère une réponse basée sur les documents pertinents.
        
        Args:
            question (str): Question de l'utilisateur.
            relevant_docs (List[Document]): Documents pertinents.
            
        Returns:
            str: Réponse générée par le LLM.
        """
        try:
            # Extraire le contenu des documents
            context_documents = [doc.page_content for doc in relevant_docs]
            
            # Créer le prompt RAG
            prompt = llm_handler.create_rag_prompt(question, context_documents)
            
            # Générer la réponse
            answer = llm_handler.generate_response(prompt)
            
            return answer
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la réponse: {e}")
            raise
    
    def _prepare_sources(self, relevant_docs: List[Document]) -> List[Dict[str, Any]]:
        """
        Prépare les informations des sources pour l'affichage.
        
        Args:
            relevant_docs (List[Document]): Documents pertinents.
            
        Returns:
            List[Dict[str, Any]]: Liste des sources avec métadonnées.
        """
        sources = []
        
        for doc in relevant_docs:
            metadata = doc.metadata
            
            source_info = {
                'filename': metadata.get('source', 'Document inconnu'),
                'page_number': metadata.get('page_number', 'N/A'),
                'chunk_id': metadata.get('chunk_id', 'N/A'),
                'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                'chunk_size': metadata.get('chunk_size', len(doc.page_content))
            }
            
            sources.append(source_info)
        
        return sources
    
    def _calculate_confidence(self, relevant_docs: List[Document]) -> float:
        """
        Calcule un score de confiance basé sur les documents trouvés.
        
        Args:
            relevant_docs (List[Document]): Documents pertinents.
            
        Returns:
            float: Score de confiance entre 0 et 1.
        """
        if not relevant_docs:
            return 0.0
        
        # Score basé sur le nombre de documents trouvés
        base_confidence = min(len(relevant_docs) / config.TOP_K_RETRIEVAL, 1.0)
        
        # Bonus si plusieurs documents différents
        unique_sources = len(set(doc.metadata.get('source', '') for doc in relevant_docs))
        diversity_bonus = min(unique_sources / 2.0, 0.2)
        
        return min(base_confidence + diversity_bonus, 1.0)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Récupère le statut actuel du pipeline.
        
        Returns:
            Dict[str, Any]: Informations sur le statut du pipeline.
        """
        status = {
            'initialized': self.is_initialized,
            'documents_loaded': self.documents_loaded,
            'vectorstore_ready': self.vectorstore_ready,
            'llm_ready': False
        }
        
        # Test du LLM
        try:
            llm_test = llm_handler.test_connection()
            status['llm_ready'] = llm_test['status'] == 'success'
        except:
            status['llm_ready'] = False
        
        # Informations sur la base vectorielle
        if self.vectorstore_ready:
            vectorstore_info = vectorstore_manager.get_vectorstore_info()
            status['vectorstore_info'] = vectorstore_info
        
        # Résumé des documents
        if self.documents_loaded:
            doc_summary = document_loader.get_document_summary()
            status['document_summary'] = doc_summary
        
        return status
    
    def refresh_documents(self) -> Dict[str, Any]:
        """
        Rafraîchit les documents et la base vectorielle.
        
        Returns:
            Dict[str, Any]: Résultat de l'opération.
        """
        try:
            logger.info("🔄 Rafraîchissement des documents...")
            
            # Réinitialiser le pipeline
            self.is_initialized = False
            self.documents_loaded = False
            self.vectorstore_ready = False
            
            # Réinitialiser avec force
            return self.initialize(force_refresh=True)
            
        except Exception as e:
            logger.error(f"Erreur lors du rafraîchissement: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }


# Instance globale du pipeline
rag_pipeline = RAGPipeline()


if __name__ == "__main__":
    # Test du module
    print("Test du module RAGPipeline...")
    
    try:
        # Initialisation
        init_result = rag_pipeline.initialize()
        print(f"🚀 Initialisation: {init_result['status']}")
        
        if init_result['status'] == 'success':
            print(f"📚 {init_result['documents_count']} documents chargés")
            
            # Test de question
            test_question = "Quels sont les points clés abordés dans les documents ?"
            print(f"\n❓ Question test: {test_question}")
            
            response = rag_pipeline.ask_question(test_question)
            print(f"✅ Réponse générée (confiance: {response['metadata']['confidence']:.2f})")
            print(f"📝 Réponse: {response['answer'][:200]}...")
            print(f"📚 Sources: {len(response['sources'])} document(s)")
            
        else:
            print(f"❌ Erreur: {init_result['message']}")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
