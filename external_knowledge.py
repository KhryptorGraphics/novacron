#!/usr/bin/env python3.12
"""
External knowledge integration for the NovaCron Code Memory system.
This module provides connectors to various external knowledge sources like
GitHub, Stack Exchange, and Google Knowledge Graph.
"""

import os
import re
import json
import time
import logging
import requests
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from github import Github, RateLimitExceededException
from stackapi import StackAPI
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import markdown
from bs4 import BeautifulSoup
from qdrant_mcp_utils import QdrantMemory, store_code_in_qdrant

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
STACK_API_KEY = os.environ.get("STACK_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

class RateLimitHandler:
    """Handles rate limiting for various APIs."""
    
    @staticmethod
    def handle_github_rate_limit(github_client):
        """Check and handle GitHub API rate limit."""
        rate_limit = github_client.get_rate_limit()
        core_rate = rate_limit.core
        
        if core_rate.remaining < 10:
            reset_timestamp = core_rate.reset.timestamp()
            sleep_time = reset_timestamp - time.time() + 60  # Add 60s buffer
            if sleep_time > 0:
                logger.warning(f"GitHub API rate limit nearly exceeded. Sleeping for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)
                
    @staticmethod
    def handle_stackexchange_rate_limit(response):
        """Check and handle Stack Exchange API rate limit."""
        if 'backoff' in response:
            backoff_time = int(response['backoff'])
            logger.warning(f"Stack Exchange API requests backoff for {backoff_time} seconds.")
            time.sleep(backoff_time)

class GitHubConnector:
    """Connector for GitHub API to fetch code and documentation."""
    
    def __init__(self, token: Optional[str] = None):
        """Initialize GitHub connector with optional token."""
        self.token = token or GITHUB_TOKEN
        if self.token:
            self.github = Github(self.token)
            logger.info("GitHub connector initialized with token")
        else:
            self.github = Github()
            logger.warning("GitHub connector initialized without token, rate limits will be strict")
    
    def search_repositories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for repositories matching the query.
        
        Args:
            query: Search query in GitHub format
            limit: Maximum number of repositories to return
            
        Returns:
            List of repository information dictionaries
        """
        try:
            RateLimitHandler.handle_github_rate_limit(self.github)
            repositories = []
            
            for repo in self.github.search_repositories(query=query, sort="stars")[:limit]:
                repositories.append({
                    "name": repo.name,
                    "full_name": repo.full_name,
                    "description": repo.description,
                    "url": repo.html_url,
                    "stars": repo.stargazers_count,
                    "language": repo.language
                })
            
            return repositories
        except RateLimitExceededException:
            logger.error("GitHub API rate limit exceeded")
            return []
        except Exception as e:
            logger.error(f"Error searching GitHub repositories: {str(e)}")
            return []
    
    def get_repository_documentation(self, repo_name: str, 
                                    doc_patterns: List[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve documentation files from a GitHub repository.
        
        Args:
            repo_name: Repository name in the format "owner/repo"
            doc_patterns: List of file patterns to consider documentation
            
        Returns:
            List of documentation content dictionaries
        """
        if doc_patterns is None:
            doc_patterns = [
                r"\.md$",           # Markdown files
                r"^docs\/.*\.md$",  # Documentation directory markdown
                r"^README.*$",      # README files
                r"\.rst$"           # ReStructured Text files
            ]
            
        try:
            RateLimitHandler.handle_github_rate_limit(self.github)
            repo = self.github.get_repo(repo_name)
            doc_files = []
            
            # Get default branch
            default_branch = repo.default_branch
            
            # Get all files in the repository's default branch
            files = repo.get_contents("", ref=default_branch)
            
            # Look for documentation files
            while files:
                file_content = files.pop(0)
                
                # If it's a directory, get its contents
                if file_content.type == "dir":
                    files.extend(repo.get_contents(file_content.path, ref=default_branch))
                    continue
                
                # Check if the file matches documentation patterns
                for pattern in doc_patterns:
                    if re.search(pattern, file_content.path, re.IGNORECASE):
                        content = repo.get_contents(file_content.path, ref=default_branch).decoded_content.decode('utf-8')
                        doc_files.append({
                            "repo": repo_name,
                            "path": file_content.path,
                            "content": content,
                            "url": file_content.html_url,
                            "type": os.path.splitext(file_content.path)[1][1:] if "." in file_content.path else "unknown"
                        })
                        break
            
            return doc_files
        except RateLimitExceededException:
            logger.error("GitHub API rate limit exceeded")
            return []
        except Exception as e:
            logger.error(f"Error getting documentation from repository {repo_name}: {str(e)}")
            return []
            
    def search_code(self, query: str, language: Optional[str] = None, 
                   limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for code matching the query.
        
        Args:
            query: Code search query
            language: Optional language filter
            limit: Maximum number of results to return
            
        Returns:
            List of code snippet dictionaries
        """
        try:
            RateLimitHandler.handle_github_rate_limit(self.github)
            
            # Build the search query
            search_query = query
            if language:
                search_query = f"{search_query} language:{language}"
            
            code_results = []
            for code_result in self.github.search_code(query=search_query)[:limit]:
                repo = code_result.repository
                content = code_result.decoded_content.decode('utf-8')
                
                code_results.append({
                    "repo": repo.full_name,
                    "path": code_result.path,
                    "content": content,
                    "url": code_result.html_url,
                    "type": os.path.splitext(code_result.path)[1][1:] if "." in code_result.path else "unknown"
                })
            
            return code_results
        except RateLimitExceededException:
            logger.error("GitHub API rate limit exceeded")
            return []
        except Exception as e:
            logger.error(f"Error searching code on GitHub: {str(e)}")
            return []

class StackExchangeConnector:
    """Connector for Stack Exchange API (primarily Stack Overflow)."""
    
    def __init__(self, api_key: Optional[str] = None, site: str = "stackoverflow"):
        """
        Initialize Stack Exchange connector.
        
        Args:
            api_key: Stack Exchange API key
            site: Stack Exchange site to query (default: stackoverflow)
        """
        self.api_key = api_key or STACK_API_KEY
        self.site = site
        
        if self.api_key:
            self.stack_api = StackAPI(site, key=self.api_key)
            logger.info(f"Stack Exchange connector initialized for {site} with API key")
        else:
            self.stack_api = StackAPI(site)
            logger.warning(f"Stack Exchange connector initialized for {site} without API key, rate limits will be strict")
    
    def search_questions(self, query: str, tagged: Optional[List[str]] = None, 
                        limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for questions matching the query.
        
        Args:
            query: Search query
            tagged: Optional list of tags to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of question dictionaries
        """
        try:
            params = {
                'sort': 'votes',
                'order': 'desc',
                'q': query,
                'pagesize': min(limit, 100),
                'filter': '!-*jbN-o8P3E5'  # Include bodies in the response
            }
            
            if tagged:
                params['tagged'] = ';'.join(tagged)
            
            response = self.stack_api.fetch('search/advanced', **params)
            RateLimitHandler.handle_stackexchange_rate_limit(response)
            
            questions = []
            for item in response['items']:
                # Clean HTML from body
                soup = BeautifulSoup(item.get('body', ''), 'html.parser')
                text_body = soup.get_text()
                
                questions.append({
                    "title": item.get('title'),
                    "body": text_body,
                    "tags": item.get('tags', []),
                    "score": item.get('score', 0),
                    "answer_count": item.get('answer_count', 0),
                    "is_answered": item.get('is_answered', False),
                    "link": item.get('link'),
                    "question_id": item.get('question_id')
                })
            
            return questions
        except Exception as e:
            logger.error(f"Error searching Stack Exchange questions: {str(e)}")
            return []
    
    def get_question_answers(self, question_id: int) -> List[Dict[str, Any]]:
        """
        Get answers for a specific question.
        
        Args:
            question_id: Stack Exchange question ID
            
        Returns:
            List of answer dictionaries
        """
        try:
            response = self.stack_api.fetch('questions/{ids}/answers', ids=[question_id], 
                                          sort='votes', order='desc', 
                                          filter='!-*jbN-o8P3E5')  # Include bodies
            
            RateLimitHandler.handle_stackexchange_rate_limit(response)
            
            answers = []
            for item in response['items']:
                # Clean HTML from body
                soup = BeautifulSoup(item.get('body', ''), 'html.parser')
                text_body = soup.get_text()
                
                answers.append({
                    "body": text_body,
                    "score": item.get('score', 0),
                    "is_accepted": item.get('is_accepted', False),
                    "link": item.get('link')
                })
            
            return answers
        except Exception as e:
            logger.error(f"Error getting answers for question {question_id}: {str(e)}")
            return []
    
    def get_tag_wikis(self, tags: List[str]) -> Dict[str, str]:
        """
        Get tag wikis for specified tags.
        
        Args:
            tags: List of tags to get wikis for
            
        Returns:
            Dictionary mapping tags to their wiki excerpts
        """
        try:
            response = self.stack_api.fetch('tags/{tags}/wikis', tags=tags)
            RateLimitHandler.handle_stackexchange_rate_limit(response)
            
            tag_wikis = {}
            for item in response['items']:
                tag_name = item.get('tag_name')
                if tag_name:
                    # Clean HTML from excerpt
                    soup = BeautifulSoup(item.get('excerpt', ''), 'html.parser')
                    tag_wikis[tag_name] = soup.get_text()
            
            return tag_wikis
        except Exception as e:
            logger.error(f"Error getting tag wikis: {str(e)}")
            return {}

class GoogleKnowledgeGraphConnector:
    """Connector for Google Knowledge Graph API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Google Knowledge Graph connector."""
        self.api_key = api_key or GOOGLE_API_KEY
        if not self.api_key:
            logger.warning("Google Knowledge Graph API key not provided")
    
    def search_entities(self, query: str, types: Optional[List[str]] = None,
                       limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for entities in the Knowledge Graph.
        
        Args:
            query: Search query
            types: Optional list of entity types to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of entity dictionaries
        """
        if not self.api_key:
            logger.error("Cannot search Knowledge Graph without API key")
            return []
            
        try:
            service = build('kgsearch', 'v1', developerKey=self.api_key)
            
            params = {
                'query': query,
                'limit': limit,
                'indent': True
            }
            
            if types:
                params['types'] = types
                
            response = service.entities().search(**params).execute()
            
            entities = []
            for element in response.get('itemListElement', []):
                entity = element.get('result', {})
                entities.append({
                    "name": entity.get('name'),
                    "description": entity.get('description'),
                    "detailed_description": entity.get('detailedDescription', {}).get('articleBody'),
                    "url": entity.get('detailedDescription', {}).get('url'),
                    "types": entity.get('@type', [])
                })
                
            return entities
        except HttpError as e:
            logger.error(f"Google Knowledge Graph API error: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error searching Knowledge Graph: {str(e)}")
            return []

class ExternalKnowledgeIndexer:
    """Index external knowledge into the Qdrant database."""
    
    def __init__(self, github_connector: Optional[GitHubConnector] = None,
                stack_connector: Optional[StackExchangeConnector] = None,
                knowledge_graph_connector: Optional[GoogleKnowledgeGraphConnector] = None):
        """
        Initialize the indexer with API connectors.
        
        Args:
            github_connector: GitHub API connector
            stack_connector: Stack Exchange API connector
            knowledge_graph_connector: Google Knowledge Graph connector
        """
        self.github = github_connector or GitHubConnector()
        self.stack = stack_connector or StackExchangeConnector()
        self.knowledge_graph = knowledge_graph_connector or GoogleKnowledgeGraphConnector()
        self.qdrant_memory = QdrantMemory()
        
    def _preprocess_markdown(self, content: str) -> str:
        """
        Preprocess markdown content for better indexing.
        
        Args:
            content: Markdown content
            
        Returns:
            Processed plain text
        """
        # Convert markdown to HTML
        html = markdown.markdown(content)
        
        # Parse HTML and extract text
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    def index_github_repo_docs(self, repo_name: str) -> int:
        """
        Index documentation from a GitHub repository.
        
        Args:
            repo_name: Repository name in format "owner/repo"
            
        Returns:
            Number of documents indexed
        """
        docs = self.github.get_repository_documentation(repo_name)
        count = 0
        
        for doc in docs:
            # Process content based on file type
            if doc['type'] in ('md', 'markdown'):
                content = self._preprocess_markdown(doc['content'])
            else:
                content = doc['content']
                
            # Create metadata
            metadata = {
                "source": "github",
                "repo": doc['repo'],
                "path": doc['path'],
                "url": doc['url'],
                "type": doc['type'],
                "indexed_at": datetime.now().isoformat()
            }
            
            # Store in Qdrant
            file_path = f"external/github/{doc['repo']}/{doc['path']}"
            result = store_code_in_qdrant(file_path, content, metadata)
            
            if result.get('success'):
                count += 1
                logger.info(f"Indexed GitHub doc: {file_path}")
            else:
                logger.error(f"Failed to index GitHub doc: {file_path}")
                
        return count
        
    def index_stack_overflow_questions(self, query: str, tags: Optional[List[str]] = None,
                                      limit: int = 10) -> int:
        """
        Index Stack Overflow questions related to a query.
        
        Args:
            query: Search query
            tags: Optional list of tags to filter by
            limit: Maximum number of questions to index
            
        Returns:
            Number of questions indexed
        """
        questions = self.stack.search_questions(query, tagged=tags, limit=limit)
        count = 0
        
        for question in questions:
            # Get answers for each question
            answers = self.stack.get_question_answers(question['question_id'])
            
            # Create a comprehensive document with question and top answers
            content = f"# {question['title']}\n\n{question['body']}\n\n"
            
            # Add best answers (up to 3)
            for i, answer in enumerate(answers[:3]):
                content += f"\n## Answer {i+1}"
                if answer.get('is_accepted'):
                    content += " (Accepted)"
                content += f" - Score: {answer['score']}\n\n{answer['body']}\n"
                
            # Create metadata
            metadata = {
                "source": "stackoverflow",
                "question_id": question['question_id'],
                "title": question['title'],
                "tags": question['tags'],
                "score": question['score'],
                "answer_count": question['answer_count'],
                "is_answered": question['is_answered'],
                "link": question['link'],
                "indexed_at": datetime.now().isoformat()
            }
            
            # Store in Qdrant
            file_path = f"external/stackoverflow/question_{question['question_id']}"
            result = store_code_in_qdrant(file_path, content, metadata)
            
            if result.get('success'):
                count += 1
                logger.info(f"Indexed Stack Overflow question: {question['question_id']}")
            else:
                logger.error(f"Failed to index Stack Overflow question: {question['question_id']}")
                
        return count
        
    def index_knowledge_graph_entities(self, query: str, types: Optional[List[str]] = None,
                                      limit: int = 5) -> int:
        """
        Index knowledge graph entities related to a query.
        
        Args:
            query: Search query
            types: Optional list of entity types
            limit: Maximum number of entities to index
            
        Returns:
            Number of entities indexed
        """
        entities = self.knowledge_graph.search_entities(query, types=types, limit=limit)
        count = 0
        
        for entity in entities:
            # Create content from entity information
            content = f"# {entity.get('name', 'Unknown Entity')}\n\n"
            
            if entity.get('description'):
                content += f"{entity['description']}\n\n"
                
            if entity.get('detailed_description'):
                content += f"## Detailed Description\n\n{entity['detailed_description']}\n\n"
                
            if entity.get('url'):
                content += f"Source: {entity['url']}\n"
                
            # Create metadata
            metadata = {
                "source": "knowledge_graph",
                "name": entity.get('name'),
                "types": entity.get('types', []),
                "url": entity.get('url'),
                "indexed_at": datetime.now().isoformat()
            }
            
            # Create a filename-safe entity name
            safe_name = re.sub(r'[^\w\s-]', '', entity.get('name', 'unknown')).strip().lower()
            safe_name = re.sub(r'[-\s]+', '-', safe_name)
            
            # Store in Qdrant
            file_path = f"external/knowledge_graph/{safe_name}"
            result = store_code_in_qdrant(file_path, content, metadata)
            
            if result.get('success'):
                count += 1
                logger.info(f"Indexed Knowledge Graph entity: {entity.get('name')}")
            else:
                logger.error(f"Failed to index Knowledge Graph entity: {entity.get('name')}")
                
        return count

def index_related_content(topic: str, technology_tags: Optional[List[str]] = None,
                         github_repos: Optional[List[str]] = None,
                         limit_per_source: int = 10) -> Dict[str, int]:
    """
    Index related content from all external sources based on a topic.
    
    Args:
        topic: Main topic to search for
        technology_tags: Relevant technology tags for Stack Overflow
        github_repos: Specific GitHub repositories to index
        limit_per_source: Maximum items to index per source
        
    Returns:
        Dictionary with counts of indexed items by source
    """
    indexer = ExternalKnowledgeIndexer()
    counts = {}
    
    # Index relevant Stack Overflow questions
    if technology_tags:
        stackoverflow_count = indexer.index_stack_overflow_questions(
            topic, tags=technology_tags, limit=limit_per_source)
        counts['stackoverflow'] = stackoverflow_count
        
    # Index GitHub repository documentation
    github_count = 0
    if github_repos:
        for repo in github_repos:
            github_count += indexer.index_github_repo_docs(repo)
    counts['github'] = github_count
    
    # Index Knowledge Graph entities
    kg_count = indexer.index_knowledge_graph_entities(topic, limit=limit_per_source)
    counts['knowledge_graph'] = kg_count
    
    # Log summary
    total = sum(counts.values())
    logger.info(f"Indexed {total} items from external sources: {counts}")
    
    return counts

if __name__ == "__main__":
    # Example usage
    topic = "cloud orchestration virtual machine migration"
    tags = ["virtualization", "cloud", "vm-migration", "openstack"]
    repos = [
        "openstack/nova",
        "kubernetes/kubernetes",
        "moby/moby",
        "qemu/qemu"
    ]
    
    index_related_content(topic, technology_tags=tags, github_repos=repos)
