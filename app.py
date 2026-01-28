"""
Application Streamlit principale pour le système RAG.

Interface web moderne et intuitive pour interroger des documents PDF
scientifiques en utilisant le système RAG développé.
"""

import streamlit as st
import time
from pathlib import Path
from typing import Dict, Any, List
import logging

# Configuration du logging pour Streamlit
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration de la page Streamlit
st.set_page_config(
    page_title="RAG - Documents Scientifiques",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import des modules du projet
try:
    from src.config import config
    from src.rag_pipeline import rag_pipeline
    #from src.rag_pipeline import rag_pipeline
    from src.document_loader import document_loader
    from src.vectorstore import vectorstore_manager
    from src.llm_handler import llm_handler
except ImportError as e:
    st.error(f"Erreur d'importation: {e}")
    st.stop()

def initialize_session_state():
    """Initialise l'état de la session Streamlit."""
    if 'pipeline_initialized' not in st.session_state:
        st.session_state.pipeline_initialized = False
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'documents_info' not in st.session_state:
        st.session_state.documents_info = {}


def display_header():
    """Affiche l'en-tête de l'application."""
    st.title(config.APP_TITLE)
    st.markdown(config.APP_DESCRIPTION)
    
    # Indicateur de statut
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        pipeline_status = "✅ Initialisé" if st.session_state.pipeline_initialized else "⏳ En attente"
        st.metric("Pipeline RAG", pipeline_status)
    
    with status_col2:
        if st.session_state.pipeline_initialized:
            vectorstore_info = vectorstore_manager.get_vectorstore_info()
            doc_count = vectorstore_info.get('document_count', 0)
            st.metric("Documents", doc_count)
        else:
            st.metric("Documents", "N/A")
    
    with status_col3:
        if st.session_state.pipeline_initialized:
            llm_info = llm_handler.get_model_info()
            model_name = llm_info.get('model_name', 'N/A')
            st.metric("Modèle LLM", model_name.split('/')[-1] if '/' in model_name else model_name)
        else:
            st.metric("Modèle LLM", "N/A")


def display_sidebar():
    """Affiche la barre latérale avec les contrôles."""
    st.sidebar.header("⚙️ Configuration")
    
    # Section de gestion des documents
    st.sidebar.subheader("📚 Documents")
    
    # Afficher les documents disponibles
    pdf_files = config.get_document_files()
    if pdf_files:
        st.sidebar.success(f"✅ {len(pdf_files)} document(s) trouvé(s)")
        with st.sidebar.expander("Voir les documents"):
            for pdf_file in pdf_files:
                st.write(f"📄 {pdf_file.name}")
    else:
        st.sidebar.warning("⚠️ Aucun document PDF trouvé")
        st.sidebar.info("Placez vos fichiers PDF dans le dossier `data/documents/`")
    
    # Bouton d'initialisation
    st.sidebar.subheader("🚀 Initialisation")
    
    if st.sidebar.button("🔄 Initialiser le Pipeline RAG", type="primary"):
        with st.spinner("Initialisation du pipeline RAG..."):
            init_result = rag_pipeline.initialize()
            
            if init_result['status'] == 'success':
                st.session_state.pipeline_initialized = True
                st.sidebar.success("✅ Pipeline initialisé avec succès")
                
                # Mettre à jour les informations des documents
                st.session_state.documents_info = document_loader.get_document_summary()
                
                # Rafraîchir la page
                st.rerun()
            else:
                st.sidebar.error(f"❌ Erreur: {init_result['message']}")
    
    # Bouton de rafraîchissement
    if st.sidebar.button("🔄 Rafraîchir les Documents"):
        with st.spinner("Rafraîchissement des documents..."):
            refresh_result = rag_pipeline.refresh_documents()
            
            if refresh_result['status'] == 'success':
                st.session_state.documents_info = document_loader.get_document_summary()
                st.sidebar.success("✅ Documents rafraîchis")
                st.rerun()
            else:
                st.sidebar.error(f"❌ Erreur: {refresh_result['message']}")
    
    # Section d'informations système
    st.sidebar.subheader("ℹ️ Informations Système")
    
    if st.session_state.pipeline_initialized:
        # Informations sur la base vectorielle
        vectorstore_info = vectorstore_manager.get_vectorstore_info()
        st.sidebar.info(f"🗄️ Base vectorielle: {vectorstore_info.get('document_count', 0)} documents")
        
        # Informations sur le modèle
        llm_info = llm_handler.get_model_info()
        if llm_info['status'] == 'ready':
            st.sidebar.info(f"🤖 Modèle: {llm_info['model_name']}")
        
        # Paramètres de configuration
        with st.sidebar.expander("Paramètres RAG"):
            st.write(f"📄 Taille chunks: {config.CHUNK_SIZE}")
            st.write(f"🔗 Overlap: {config.CHUNK_OVERLAP}")
            st.write(f"🔍 Top-K: {config.TOP_K_RETRIEVAL}")
            st.write(f"🧠 Embeddings: {config.EMBEDDING_MODEL.split('/')[-1]}")
    
    # Section d'aide
    st.sidebar.subheader("❓ Aide")
    st.sidebar.info("""
    **Comment utiliser:**
    1. Placez vos PDFs dans `data/documents/`
    2. Cliquez sur "Initialiser le Pipeline RAG"
    3. Posez vos questions dans la zone de texte
    4. Consultez les réponses avec sources
    """)


def display_conversation_history():
    """Affiche l'historique de conversation."""
    if st.session_state.conversation_history:
        st.subheader("💬 Historique de Conversation")
        
        for i, (question, response) in enumerate(st.session_state.conversation_history[-5:]):
            with st.expander(f"Q{i+1}: {question[:50]}...", expanded=False):
                st.write(f"**Question:** {question}")
                st.write(f"**Réponse:** {response['answer']}")
                
                if response['sources']:
                    st.write("**Sources:**")
                    for source in response['sources']:
                        st.write(f"- 📄 {source['filename']} (page {source['page_number']})")
                
                st.write(f"**Confiance:** {response['metadata']['confidence']:.2f}")


def display_main_interface():
    """Affiche l'interface principale de questions/réponses."""
    if not st.session_state.pipeline_initialized:
        st.warning("⚠️ Veuillez d'abord initialiser le pipeline RAG depuis la barre latérale.")
        return
    
    st.subheader("🤖 Posez votre question")
    
    # Zone de saisie de question
    question = st.text_area(
        "Entrez votre question sur les documents:",
        placeholder="Exemple: Quelles sont les méthodologies utilisées dans cette recherche ?",
        height=100,
        key="question_input"
    )
    
    # Bouton d'envoi
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        submit_button = st.button("🔍 Rechercher", type="primary")
    
    with col2:
        clear_button = st.button("🗑️ Effacer")
    
    if clear_button:
        st.session_state.question_input = ""
        st.rerun()
    
    # Traitement de la question
    if submit_button and question.strip():
        with st.spinner("🔍 Recherche dans les documents..."):
            try:
                # Générer la réponse
                start_time = time.time()
                response = rag_pipeline.ask_question(question.strip())
                processing_time = time.time() - start_time
                
                # Afficher la réponse
                st.subheader("📝 Réponse")
                st.write(response['answer'])
                
                # Afficher les métadonnées
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confiance", f"{response['metadata']['confidence']:.2f}")
                with col2:
                    st.metric("Documents trouvés", response['metadata']['documents_found'])
                with col3:
                    st.metric("Temps de traitement", f"{processing_time:.1f}s")
                
                # Afficher les sources
                if response['sources']:
                    st.subheader("📚 Sources")
                    
                    for i, source in enumerate(response['sources'], 1):
                        with st.expander(f"Source {i}: {source['filename']} (page {source['page_number']})"):
                            st.write(f"**Contenu:** {source['content_preview']}")
                            st.write(f"**Taille du chunk:** {source['chunk_size']} caractères")
                
                # Ajouter à l'historique
                st.session_state.conversation_history.append((question, response))
                
                # Limiter l'historique à 10 entrées
                if len(st.session_state.conversation_history) > 10:
                    st.session_state.conversation_history = st.session_state.conversation_history[-10:]
                
            except Exception as e:
                st.error(f"❌ Erreur lors de la génération de la réponse: {str(e)}")
                logger.error(f"Erreur dans l'interface: {e}")


def display_documents_info():
    """Affiche les informations détaillées sur les documents."""
    if not st.session_state.pipeline_initialized:
        return
    
    if st.session_state.documents_info:
        st.subheader("📊 Informations sur les Documents")
        
        doc_summary = st.session_state.documents_info
        
        # Statistiques générales
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", doc_summary['total_files'])
        
        total_chunks = sum(file_info.get('chunks', 0) for file_info in doc_summary['files'] if 'chunks' in file_info)
        with col2:
            st.metric("Total Chunks", total_chunks)
        
        total_pages = sum(file_info.get('pages', 0) for file_info in doc_summary['files'] if 'pages' in file_info)
        with col3:
            st.metric("Total Pages", total_pages)
        
        # Détails par document
        with st.expander("Détails par document"):
            for file_info in doc_summary['files']:
                if 'error' in file_info:
                    st.error(f"❌ {file_info['filename']}: {file_info['error']}")
                else:
                    st.success(f"✅ {file_info['filename']}")
                    st.write(f"   - Pages: {file_info['pages']}")
                    st.write(f"   - Chunks: {file_info['chunks']}")
                    st.write(f"   - Taille: {file_info['text_length']} caractères")


def main():
    """Fonction principale de l'application."""
    # Initialisation
    initialize_session_state()
    
    # Affichage des composants
    display_header()
    display_sidebar()
    
    # Interface principale
    tab1, tab2, tab3 = st.tabs(["🤖 Questions/Réponses", "💬 Historique", "📊 Informations"])
    
    with tab1:
        display_main_interface()
    
    with tab2:
        display_conversation_history()
    
    with tab3:
        display_documents_info()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🔬 **RAG Project** - Système de recherche dans documents scientifiques | "
        "Développé avec LangChain, ChromaDB et Groq API"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Erreur fatale de l'application: {str(e)}")
        logger.error(f"Erreur fatale: {e}")
