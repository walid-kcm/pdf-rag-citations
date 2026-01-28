"""
Module de chargement et de traitement des documents PDF.

Ce module gère le chargement des fichiers PDF, leur découpage intelligent
en chunks avec overlap, et la préparation des documents pour la création
d'embeddings.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import PyPDF2
import re

#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
#from langchain.schema import Document

from .config import config

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Classe pour charger et traiter les documents PDF.
    
    Cette classe gère le chargement des PDFs, l'extraction du texte,
    et le découpage intelligent en chunks avec préservation du contexte.
    """
    
    def __init__(self):
        """Initialise le chargeur de documents."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def load_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Charge un fichier PDF et extrait son contenu.
        
        Args:
            pdf_path (Path): Chemin vers le fichier PDF.
            
        Returns:
            Dict[str, Any]: Dictionnaire contenant le texte et les métadonnées.
            
        Raises:
            Exception: Si le fichier ne peut pas être lu.
        """
        try:
            logger.info(f"Chargement du PDF: {pdf_path.name}")
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extraction du texte de toutes les pages
                text_content = []
                page_contents = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():  # Ignorer les pages vides
                            text_content.append(page_text)
                            page_contents.append({
                                'page_number': page_num + 1,
                                'content': page_text.strip()
                            })
                    except Exception as e:
                        logger.warning(f"Erreur lors de l'extraction de la page {page_num + 1}: {e}")
                        continue
                
                # Concaténation de tout le texte
                full_text = "\n".join(text_content)
                
                # Nettoyage du texte
                cleaned_text = self._clean_text(full_text)
                
                return {
                    'filename': pdf_path.name,
                    'filepath': str(pdf_path),
                    'full_text': cleaned_text,
                    'page_contents': page_contents,
                    'total_pages': len(pdf_reader.pages),
                    'text_length': len(cleaned_text)
                }
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement du PDF {pdf_path}: {e}")
            raise Exception(f"Impossible de charger le PDF {pdf_path.name}: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """
        Nettoie le texte extrait du PDF.
        
        Args:
            text (str): Texte brut extrait du PDF.
            
        Returns:
            str: Texte nettoyé.
        """
        # Supprimer les caractères de contrôle
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Normaliser les espaces multiples
        text = re.sub(r'\s+', ' ', text)
        
        # Supprimer les lignes trop courtes (probablement des artefacts)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 10:  # Garder seulement les lignes significatives
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def split_document(self, document_data: Dict[str, Any]) -> List[Document]:
        """
        Découpe un document en chunks pour la création d'embeddings.
        
        Args:
            document_data (Dict[str, Any]): Données du document chargé.
            
        Returns:
            List[Document]: Liste des chunks sous forme de Documents LangChain.
        """
        try:
            filename = document_data['filename']
            full_text = document_data['full_text']
            page_contents = document_data['page_contents']
            
            logger.info(f"Découpage du document {filename} en chunks...")
            
            # Créer un mapping page -> contenu pour référence
            page_mapping = {page['page_number']: page['content'] for page in page_contents}
            
            # Découper le texte en chunks
            text_chunks = self.text_splitter.split_text(full_text)
            
            # Créer les documents LangChain avec métadonnées
            documents = []
            
            for i, chunk in enumerate(text_chunks):
                # Déterminer la page approximative du chunk
                page_number = self._find_page_for_chunk(chunk, page_mapping)
                
                # Métadonnées enrichies
                metadata = {
                    'source': filename,
                    'chunk_id': i,
                    'page_number': page_number,
                    'total_chunks': len(text_chunks),
                    'chunk_size': len(chunk)
                }
                
                # Créer le document LangChain
                doc = Document(
                    page_content=chunk,
                    metadata=metadata
                )
                
                documents.append(doc)
            
            logger.info(f"Document {filename} découpé en {len(documents)} chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Erreur lors du découpage du document: {e}")
            raise Exception(f"Erreur lors du découpage: {str(e)}")
    
    def _find_page_for_chunk(self, chunk: str, page_mapping: Dict[int, str]) -> int:
        """
        Détermine la page approximative d'un chunk basé sur son contenu.
        
        Args:
            chunk (str): Contenu du chunk.
            page_mapping (Dict[int, str]): Mapping des numéros de page vers leur contenu.
            
        Returns:
            int: Numéro de page approximatif.
        """
        # Prendre les premiers mots du chunk pour la recherche
        chunk_words = chunk[:100].lower().split()
        
        best_match = 1
        max_overlap = 0
        
        for page_num, page_content in page_mapping.items():
            page_words = page_content[:100].lower().split()
            
            # Calculer l'overlap entre les mots
            overlap = len(set(chunk_words) & set(page_words))
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = page_num
        
        return best_match
    
    def load_all_documents(self) -> List[Document]:
        """
        Charge et traite tous les documents PDF du répertoire.
        
        Returns:
            List[Document]: Liste de tous les chunks de tous les documents.
            
        Raises:
            Exception: Si aucun document n'est trouvé ou si une erreur survient.
        """
        try:
            # Récupérer tous les fichiers PDF
            pdf_files = config.get_document_files()
            
            if not pdf_files:
                raise Exception(config.ERROR_MESSAGES["documents_not_found"])
            
            logger.info(f"Trouvé {len(pdf_files)} fichier(s) PDF à traiter")
            
            all_documents = []
            
            for pdf_file in pdf_files:
                try:
                    # Charger le PDF
                    document_data = self.load_pdf(pdf_file)
                    
                    # Découper en chunks
                    chunks = self.split_document(document_data)
                    
                    all_documents.extend(chunks)
                    
                    logger.info(f"✅ {pdf_file.name}: {len(chunks)} chunks créés")
                    
                except Exception as e:
                    logger.error(f"❌ Erreur avec {pdf_file.name}: {e}")
                    continue
            
            if not all_documents:
                raise Exception("Aucun document n'a pu être traité avec succès")
            
            logger.info(f"🎉 Total: {len(all_documents)} chunks créés à partir de {len(pdf_files)} documents")
            return all_documents
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des documents: {e}")
            raise
    
    def get_document_summary(self) -> Dict[str, Any]:
        """
        Génère un résumé des documents disponibles.
        
        Returns:
            Dict[str, Any]: Résumé des documents avec statistiques.
        """
        pdf_files = config.get_document_files()
        
        summary = {
            'total_files': len(pdf_files),
            'files': []
        }
        
        for pdf_file in pdf_files:
            try:
                document_data = self.load_pdf(pdf_file)
                chunks = self.split_document(document_data)
                
                summary['files'].append({
                    'filename': pdf_file.name,
                    'pages': document_data['total_pages'],
                    'chunks': len(chunks),
                    'text_length': document_data['text_length']
                })
                
            except Exception as e:
                summary['files'].append({
                    'filename': pdf_file.name,
                    'error': str(e)
                })
        
        return summary


# Instance globale du chargeur
document_loader = DocumentLoader()


if __name__ == "__main__":
    # Test du module
    print("Test du module DocumentLoader...")
    
    try:
        # Résumé des documents
        summary = document_loader.get_document_summary()
        print(f"📚 {summary['total_files']} document(s) trouvé(s)")
        
        for file_info in summary['files']:
            if 'error' in file_info:
                print(f"❌ {file_info['filename']}: {file_info['error']}")
            else:
                print(f"✅ {file_info['filename']}: {file_info['pages']} pages, {file_info['chunks']} chunks")
        
        # Chargement complet
        if summary['total_files'] > 0:
            documents = document_loader.load_all_documents()
            print(f"🎉 {len(documents)} chunks au total")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
