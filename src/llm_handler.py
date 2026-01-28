"""
Module de gestion du modèle de langage avec l'API Groq.

Ce module gère l'interface avec l'API Groq pour générer des réponses
en utilisant le modèle Mixtral-8x7B-32768, optimisé pour la rapidité
et l'efficacité.
"""

import logging
from typing import List, Dict, Any, Optional
from groq import Groq
# from langchain.llms.base import LLM
from langchain_core.language_models import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
#from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import BaseModel, Field

from .config import config

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GroqLLM(LLM, BaseModel):
    """
    Implémentation LangChain du modèle Groq.
    
    Cette classe encapsule l'API Groq dans une interface compatible
    avec LangChain pour une intégration transparente.
    """
    
    client: Optional[Groq] = Field(default=None)
    model_name: str = Field(default=config.LLM_MODEL)
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=2048)
    top_p: float = Field(default=0.9)
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        """Initialise le modèle Groq."""
        super().__init__(**kwargs)
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """
        Initialise le client Groq.
        
        Raises:
            Exception: Si la clé API n'est pas configurée.
        """
        if not config.GROQ_API_KEY:
            raise Exception(config.ERROR_MESSAGES["missing_api_key"])
        
        try:
            self.client = Groq(api_key=config.GROQ_API_KEY)
            logger.info(f"✅ Client Groq initialisé avec le modèle {self.model_name}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du client Groq: {e}")
            raise Exception(f"Erreur d'initialisation Groq: {str(e)}")
    
    @property
    def _llm_type(self) -> str:
        """Retourne le type de LLM."""
        return "groq"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Appelle le modèle Groq avec un prompt.
        
        Args:
            prompt (str): Prompt à envoyer au modèle.
            stop (Optional[List[str]]): Mots d'arrêt.
            run_manager (Optional[CallbackManagerForLLMRun]): Gestionnaire de callbacks.
            **kwargs: Arguments supplémentaires.
            
        Returns:
            str: Réponse générée par le modèle.
            
        Raises:
            Exception: Si l'appel échoue.
        """
        if self.client is None:
            raise Exception("Client Groq non initialisé")
        
        try:
            logger.info(f"Génération de réponse avec Groq...")
            
            # Préparer les paramètres
            generation_params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "stream": False
            }
            
            # Ajouter les mots d'arrêt si spécifiés
            if stop:
                generation_params["stop"] = stop
            
            # Appeler l'API
            response = self.client.chat.completions.create(**generation_params)
            
            # Extraire la réponse
            if response.choices and len(response.choices) > 0:
                generated_text = response.choices[0].message.content
                logger.info(f"✅ Réponse générée ({len(generated_text)} caractères)")
                return generated_text
            else:
                raise Exception("Aucune réponse générée par le modèle")
                
        except Exception as e:
            logger.error(f"Erreur lors de la génération: {e}")
            raise Exception(f"Erreur de génération: {str(e)}")


class LLMHandler:
    """
    Gestionnaire principal du modèle de langage.
    
    Cette classe fournit une interface simplifiée pour interagir
    avec le modèle Groq et gère la configuration des prompts.
    """
    
    def __init__(self):
        """Initialise le gestionnaire LLM."""
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self) -> None:
        """
        Initialise le modèle de langage.
        
        Raises:
            Exception: Si l'initialisation échoue.
        """
        try:
            self.llm = GroqLLM()
            logger.info("✅ Gestionnaire LLM initialisé")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du LLM: {e}")
            raise
    
    def generate_response(self, prompt: str) -> str:
        """
        Génère une réponse à partir d'un prompt.
        
        Args:
            prompt (str): Prompt d'entrée.
            
        Returns:
            str: Réponse générée.
            
        Raises:
            Exception: Si la génération échoue.
        """
        if self.llm is None:
            raise Exception("LLM non initialisé")
        
        try:
            return self.llm._call(prompt)
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération: {e}")
            raise Exception(config.ERROR_MESSAGES["llm_error"])
    
    def create_rag_prompt(self, question: str, context_documents: List[str]) -> str:
        """
        Crée un prompt optimisé pour le RAG.
        
        Args:
            question (str): Question de l'utilisateur.
            context_documents (List[str]): Documents de contexte.
            
        Returns:
            str: Prompt formaté pour le RAG.
        """
        # Concaténer les documents de contexte
        context = "\n\n".join(context_documents)
        
        prompt = f"""Tu es un assistant spécialisé dans l'analyse de documents scientifiques. 
Tu dois répondre aux questions en te basant uniquement sur les documents fournis.

CONTEXTE (Documents sources):
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Réponds de manière précise et détaillée en te basant uniquement sur les informations fournies dans le contexte.
2. Si l'information n'est pas disponible dans les documents, indique-le clairement.
3. Cite les sources en mentionnant le nom du document et le numéro de page quand c'est possible.
4. Structure ta réponse de manière claire et professionnelle.
5. Utilise un langage scientifique approprié.

RÉPONSE:"""
        
        return prompt
    
    def create_summary_prompt(self, documents: List[str]) -> str:
        """
        Crée un prompt pour résumer les documents.
        
        Args:
            documents (List[str]): Documents à résumer.
            
        Returns:
            str: Prompt formaté pour le résumé.
        """
        context = "\n\n".join(documents)
        
        prompt = f"""Tu es un expert en analyse de documents scientifiques. 
Résume les documents suivants en identifiant les points clés, les méthodologies, 
et les conclusions principales.

DOCUMENTS:
{context}

RÉSUMÉ (structurez avec des sections claires):"""
        
        return prompt
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Teste la connexion avec l'API Groq.
        
        Returns:
            Dict[str, Any]: Résultat du test avec informations de statut.
        """
        try:
            if self.llm is None:
                return {
                    'status': 'error',
                    'message': 'LLM non initialisé'
                }
            
            # Test simple
            test_prompt = "Dis 'Hello' en une phrase."
            response = self.generate_response(test_prompt)
            
            return {
                'status': 'success',
                'message': 'Connexion réussie',
                'model': self.llm.model_name,
                'test_response': response,
                'response_length': len(response)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Erreur de connexion: {str(e)}'
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Récupère les informations sur le modèle configuré.
        
        Returns:
            Dict[str, Any]: Informations sur le modèle.
        """
        if self.llm is None:
            return {
                'status': 'not_initialized',
                'model_name': None
            }
        
        return {
            'status': 'ready',
            'model_name': self.llm.model_name,
            'temperature': self.llm.temperature,
            'max_tokens': self.llm.max_tokens,
            'top_p': self.llm.top_p
        }
    
    def update_parameters(self, **kwargs) -> None:
        """
        Met à jour les paramètres du modèle.
        
        Args:
            **kwargs: Paramètres à mettre à jour (temperature, max_tokens, etc.).
        """
        if self.llm is None:
            raise Exception("LLM non initialisé")
        
        try:
            for key, value in kwargs.items():
                if hasattr(self.llm, key):
                    setattr(self.llm, key, value)
                    logger.info(f"Paramètre {key} mis à jour: {value}")
                else:
                    logger.warning(f"Paramètre inconnu: {key}")
                    
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des paramètres: {e}")
            raise


# Instance globale du gestionnaire
llm_handler = LLMHandler()


if __name__ == "__main__":
    # Test du module
    print("Test du module LLMHandler...")
    
    try:
        # Test de connexion
        test_result = llm_handler.test_connection()
        print(f"🔗 Test de connexion: {test_result['status']}")
        
        if test_result['status'] == 'success':
            print(f"🤖 Modèle: {test_result['model']}")
            print(f"📝 Réponse test: {test_result['test_response']}")
        else:
            print(f"❌ Erreur: {test_result['message']}")
        
        # Informations du modèle
        model_info = llm_handler.get_model_info()
        print(f"\n📊 Informations du modèle:")
        print(f"  - Statut: {model_info['status']}")
        if model_info['status'] == 'ready':
            print(f"  - Modèle: {model_info['model_name']}")
            print(f"  - Température: {model_info['temperature']}")
            print(f"  - Max tokens: {model_info['max_tokens']}")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
