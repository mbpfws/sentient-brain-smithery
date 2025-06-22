"""
The Codebase Knowledge Memory Layer Agent.

Manages ingestion, indexing, and persistent storage of codebase-related 
information within SurrealDB. Provides real-time monitoring and updates.
"""
import asyncio
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import logging
import ast
import re
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from ..models.memory_layers import RelationshipType
from ..models.agent_models import AgentConfig, AgentMessage, AgentResponse
from ..services.groq_service import GroqLLMService
from ..services.memory_service import MemoryService

logger = logging.getLogger(__name__)


class CodeFileType(str):
    """Supported code file types."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    OTHER = "other"


class CodeChunk:
    """Represents a chunk of code with metadata."""
    def __init__(
        self,
        id: str,
        file_id: str,
        content: str,
        chunk_type: str,
        start_line: int,
        end_line: int,
        **kwargs
    ):
        self.id = id
        self.file_id = file_id
        self.content = content
        self.chunk_type = chunk_type
        self.start_line = start_line
        self.end_line = end_line
        self.metadata = kwargs


class CodeFile:
    """Represents a code file with metadata."""
    def __init__(
        self,
        id: str,
        file_path: str,
        content: str,
        file_type: str,
        **kwargs
    ):
        self.id = id
        self.file_path = file_path
        self.content = content
        self.file_type = file_type
        self.metadata = kwargs
        self.chunks: List[CodeChunk] = []


class CodebaseWatcher(FileSystemEventHandler):
    """File system watcher for real-time codebase monitoring."""
    
    def __init__(self, agent: 'CodebaseMemoryAgent'):
        self.agent = agent
        self.supported_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.cc', '.h', '.hpp',
            '.go', '.rs', '.rb', '.php', '.cs', '.swift', '.kt', '.scala', '.sh'
        }
    
    def on_modified(self, event):
        if not event.is_directory and self._is_supported_file(event.src_path):
            asyncio.create_task(self.agent.handle_file_change(event.src_path, "modified"))
    
    def on_created(self, event):
        if not event.is_directory and self._is_supported_file(event.src_path):
            asyncio.create_task(self.agent.handle_file_change(event.src_path, "created"))
    
    def on_deleted(self, event):
        if not event.is_directory and self._is_supported_file(event.src_path):
            asyncio.create_task(self.agent.handle_file_change(event.src_path, "deleted"))
    
    def _is_supported_file(self, file_path: str) -> bool:
        """Check if file type is supported for monitoring."""
        return Path(file_path).suffix.lower() in self.supported_extensions


class CodebaseMemoryAgent:
    """
    Manages the codebase knowledge memory layer with real-time monitoring,
    intelligent chunking, and rich relationship mapping.
    """
    
    def __init__(
        self,
        config: AgentConfig,
        llm_service: GroqLLMService,
        memory_service: MemoryService,
        codebase_path: str
    ):
        self.config = config
        self.llm = llm_service
        self.memory = memory_service
        self.codebase_path = Path(codebase_path)
        
        # File monitoring
        self.observer = Observer()
        self.watcher = CodebaseWatcher(self)
        self.monitoring = False
        
        # Caching and state
        self.file_hashes: Dict[str, str] = {}
        self.indexed_files: Set[str] = set()
        
        # Language parsers
        self.language_parsers = {
            CodeFileType.PYTHON: self._parse_python_file,
            CodeFileType.JAVASCRIPT: self._parse_javascript_file,
            CodeFileType.TYPESCRIPT: self._parse_typescript_file,
        }
    
    async def initialize(self) -> AgentResponse:
        """Initialize the codebase memory agent."""
        try:
            # Validate codebase path
            if not self.codebase_path.exists():
                raise ValueError(f"Codebase path does not exist: {self.codebase_path}")
            
            # Start file monitoring
            await self.start_monitoring()
            
            # Perform initial indexing
            await self.index_codebase()
            
            logger.info(f"Codebase memory agent initialized for: {self.codebase_path}")
            
            return AgentResponse(
                agent_id=self.config.agent_id,
                success=True,
                message="Codebase memory agent initialized successfully",
                data={
                    "codebase_path": str(self.codebase_path),
                    "indexed_files": len(self.indexed_files),
                    "monitoring": self.monitoring
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize codebase memory agent: {e}")
            return AgentResponse(
                agent_id=self.config.agent_id,
                success=False,
                error=str(e)
            )
    
    async def start_monitoring(self):
        """Start real-time file system monitoring."""
        try:
            self.observer.schedule(self.watcher, str(self.codebase_path), recursive=True)
            self.observer.start()
            self.monitoring = True
            logger.info("Started codebase monitoring")
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
    
    async def stop_monitoring(self):
        """Stop file system monitoring."""
        try:
            if self.monitoring:
                self.observer.stop()
                self.observer.join()
                self.monitoring = False
                logger.info("Stopped codebase monitoring")
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")
    
    # ================================
    # CODEBASE INDEXING
    # ================================
    
    async def index_codebase(self) -> Dict[str, Any]:
        """Perform complete codebase indexing."""
        try:
            stats = {
                "files_processed": 0,
                "chunks_created": 0,
                "relationships_created": 0,
                "errors": []
            }
            
            # Find all code files
            code_files = self._find_code_files()
            
            for file_path in code_files:
                try:
                    await self._index_file(file_path)
                    stats["files_processed"] += 1
                except Exception as e:
                    stats["errors"].append(f"{file_path}: {str(e)}")
                    logger.error(f"Failed to index file {file_path}: {e}")
            
            # Create cross-file relationships
            await self._create_cross_file_relationships()
            
            logger.info(f"Codebase indexing complete: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to index codebase: {e}")
            return {"error": str(e)}
    
    def _find_code_files(self) -> List[Path]:
        """Find all code files in the codebase."""
        code_files = []
        
        # File extensions to include
        extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.h', '.go', '.rs'}
        
        # Directories to exclude
        exclude_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'build', 'dist'}
        
        for file_path in self.codebase_path.rglob('*'):
            if (file_path.is_file() and 
                file_path.suffix in extensions and
                not any(exclude_dir in file_path.parts for exclude_dir in exclude_dirs)):
                code_files.append(file_path)
        
        return code_files
    
    async def _index_file(self, file_path: Path):
        """Index a single code file."""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Calculate content hash
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Skip if file hasn't changed
            relative_path = str(file_path.relative_to(self.codebase_path))
            if relative_path in self.file_hashes and self.file_hashes[relative_path] == content_hash:
                return
            
            # Determine file type
            file_type = self._determine_file_type(file_path)
            
            # Create code file record
            code_file = CodeFile(
                id=f"code_file:{content_hash[:16]}",
                file_path=relative_path,
                content=content,
                file_type=file_type,
                size=len(content),
                lines=content.count('\n') + 1,
                hash=content_hash,
                last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
                indexed_at=datetime.utcnow()
            )
            
            # Store file in memory
            await self._store_code_file(code_file)
            
            # Parse and chunk the file
            chunks = await self._parse_and_chunk_file(code_file)
            
            # Store chunks
            for chunk in chunks:
                await self._store_code_chunk(chunk)
            
            # Update tracking
            self.file_hashes[relative_path] = content_hash
            self.indexed_files.add(relative_path)
            
            logger.debug(f"Indexed file: {relative_path} ({len(chunks)} chunks)")
            
        except Exception as e:
            logger.error(f"Failed to index file {file_path}: {e}")
            raise
    
    def _determine_file_type(self, file_path: Path) -> str:
        """Determine the programming language/type of a file."""
        extension = file_path.suffix.lower()
        
        type_map = {
            '.py': CodeFileType.PYTHON,
            '.js': CodeFileType.JAVASCRIPT,
            '.jsx': CodeFileType.JAVASCRIPT,
            '.ts': CodeFileType.TYPESCRIPT,
            '.tsx': CodeFileType.TYPESCRIPT,
            '.java': CodeFileType.JAVA,
            '.cpp': CodeFileType.CPP,
            '.cc': CodeFileType.CPP,
            '.h': CodeFileType.CPP,
            '.hpp': CodeFileType.CPP,
            '.go': CodeFileType.GO,
            '.rs': CodeFileType.RUST,
        }
        
        return type_map.get(extension, CodeFileType.OTHER)
    
    # ================================
    # CODE PARSING AND CHUNKING
    # ================================
    
    async def _parse_and_chunk_file(self, code_file: CodeFile) -> List[CodeChunk]:
        """Parse and chunk a code file based on its type."""
        try:
            parser = self.language_parsers.get(code_file.file_type)
            
            if parser:
                return await parser(code_file)
            else:
                # Fallback to simple line-based chunking
                return await self._simple_chunk_file(code_file)
                
        except Exception as e:
            logger.error(f"Failed to parse file {code_file.file_path}: {e}")
            return await self._simple_chunk_file(code_file)
    
    async def _parse_python_file(self, code_file: CodeFile) -> List[CodeChunk]:
        """Parse Python file into semantic chunks."""
        chunks = []
        
        try:
            tree = ast.parse(code_file.content)
            
            for node in ast.walk(tree):
                chunk = None
                
                if isinstance(node, ast.FunctionDef):
                    chunk = self._create_chunk_from_node(
                        code_file, node, "function", node.name
                    )
                elif isinstance(node, ast.ClassDef):
                    chunk = self._create_chunk_from_node(
                        code_file, node, "class", node.name
                    )
                elif isinstance(node, ast.AsyncFunctionDef):
                    chunk = self._create_chunk_from_node(
                        code_file, node, "async_function", node.name
                    )
                
                if chunk:
                    chunks.append(chunk)
            
            # Add import statements as a chunk
            imports_chunk = self._extract_imports_chunk(code_file)
            if imports_chunk:
                chunks.append(imports_chunk)
                
        except SyntaxError as e:
            logger.warning(f"Syntax error in {code_file.file_path}: {e}")
            # Fall back to simple chunking
            return await self._simple_chunk_file(code_file)
        
        return chunks
    
    def _create_chunk_from_node(
        self, 
        code_file: CodeFile, 
        node: ast.AST, 
        chunk_type: str, 
        name: str
    ) -> CodeChunk:
        """Create a code chunk from an AST node."""
        lines = code_file.content.split('\n')
        start_line = node.lineno - 1
        end_line = getattr(node, 'end_lineno', node.lineno) - 1
        
        # Extract the actual content
        chunk_content = '\n'.join(lines[start_line:end_line + 1])
        
        chunk_id = f"chunk:{hashlib.sha256(chunk_content.encode()).hexdigest()[:16]}"
        
        return CodeChunk(
            id=chunk_id,
            file_id=code_file.id,
            content=chunk_content,
            chunk_type=chunk_type,
            start_line=start_line + 1,
            end_line=end_line + 1,
            name=name,
            file_path=code_file.file_path,
            language=code_file.file_type
        )
    
    def _extract_imports_chunk(self, code_file: CodeFile) -> Optional[CodeChunk]:
        """Extract import statements as a separate chunk."""
        lines = code_file.content.split('\n')
        import_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.startswith('import ') or 
                stripped.startswith('from ') or
                stripped.startswith('#') and 'import' in stripped):
                import_lines.append((i, line))
            elif stripped and not stripped.startswith('#'):
                # Stop at first non-import, non-comment line
                break
        
        if import_lines:
            content = '\n'.join([line for _, line in import_lines])
            start_line = import_lines[0][0] + 1
            end_line = import_lines[-1][0] + 1
            
            chunk_id = f"chunk:{hashlib.sha256(content.encode()).hexdigest()[:16]}"
            
            return CodeChunk(
                id=chunk_id,
                file_id=code_file.id,
                content=content,
                chunk_type="imports",
                start_line=start_line,
                end_line=end_line,
                name="imports",
                file_path=code_file.file_path,
                language=code_file.file_type
            )
        
        return None
    
    async def _parse_javascript_file(self, code_file: CodeFile) -> List[CodeChunk]:
        """Parse JavaScript/TypeScript file into chunks."""
        # Simple regex-based parsing for now
        # In production, would use proper AST parser like esprima
        
        chunks = []
        content = code_file.content
        lines = content.split('\n')
        
        # Find function declarations
        function_pattern = r'^\s*(export\s+)?(async\s+)?function\s+(\w+)'
        arrow_function_pattern = r'^\s*(export\s+)?const\s+(\w+)\s*=\s*(\([^)]*\))?\s*=>'
        class_pattern = r'^\s*(export\s+)?class\s+(\w+)'
        
        for i, line in enumerate(lines):
            # Function declarations
            func_match = re.match(function_pattern, line)
            if func_match:
                chunk = self._extract_js_block_chunk(
                    code_file, lines, i, "function", func_match.group(3)
                )
                if chunk:
                    chunks.append(chunk)
            
            # Arrow functions
            arrow_match = re.match(arrow_function_pattern, line)
            if arrow_match:
                chunk = self._extract_js_block_chunk(
                    code_file, lines, i, "arrow_function", arrow_match.group(2)
                )
                if chunk:
                    chunks.append(chunk)
            
            # Class declarations
            class_match = re.match(class_pattern, line)
            if class_match:
                chunk = self._extract_js_block_chunk(
                    code_file, lines, i, "class", class_match.group(2)
                )
                if chunk:
                    chunks.append(chunk)
        
        return chunks
    
    def _extract_js_block_chunk(
        self, 
        code_file: CodeFile, 
        lines: List[str], 
        start_idx: int, 
        chunk_type: str, 
        name: str
    ) -> Optional[CodeChunk]:
        """Extract a JavaScript block (function, class, etc.) as a chunk."""
        brace_count = 0
        end_idx = start_idx
        
        for i in range(start_idx, len(lines)):
            line = lines[i]
            brace_count += line.count('{') - line.count('}')
            
            if i > start_idx and brace_count == 0:
                end_idx = i
                break
        
        if end_idx > start_idx:
            content = '\n'.join(lines[start_idx:end_idx + 1])
            chunk_id = f"chunk:{hashlib.sha256(content.encode()).hexdigest()[:16]}"
            
            return CodeChunk(
                id=chunk_id,
                file_id=code_file.id,
                content=content,
                chunk_type=chunk_type,
                start_line=start_idx + 1,
                end_line=end_idx + 1,
                name=name,
                file_path=code_file.file_path,
                language=code_file.file_type
            )
        
        return None
    
    async def _parse_typescript_file(self, code_file: CodeFile) -> List[CodeChunk]:
        """Parse TypeScript file (similar to JavaScript for now)."""
        return await self._parse_javascript_file(code_file)
    
    async def _simple_chunk_file(self, code_file: CodeFile) -> List[CodeChunk]:
        """Simple line-based chunking for unsupported file types."""
        chunks = []
        lines = code_file.content.split('\n')
        chunk_size = 50  # Lines per chunk
        
        for i in range(0, len(lines), chunk_size):
            end_idx = min(i + chunk_size, len(lines))
            content = '\n'.join(lines[i:end_idx])
            
            chunk_id = f"chunk:{hashlib.sha256(content.encode()).hexdigest()[:16]}"
            
            chunk = CodeChunk(
                id=chunk_id,
                file_id=code_file.id,
                content=content,
                chunk_type="block",
                start_line=i + 1,
                end_line=end_idx,
                name=f"block_{i//chunk_size + 1}",
                file_path=code_file.file_path,
                language=code_file.file_type
            )
            
            chunks.append(chunk)
        
        return chunks
    
    # ================================
    # MEMORY STORAGE
    # ================================
    
    async def _store_code_file(self, code_file: CodeFile):
        """Store code file in memory system."""
        try:
            # Generate embedding for file content
            file_summary = f"File: {code_file.file_path}\nType: {code_file.file_type}\nSize: {code_file.metadata.get('size', 0)} bytes"
            embedding = self.memory.generate_embedding(file_summary)
            
            # Store in database
            file_data = {
                "id": code_file.id,
                "file_path": code_file.file_path,
                "content": code_file.content,
                "file_type": code_file.file_type,
                "embedding": embedding,
                **code_file.metadata
            }
            
            await self.memory.db.create_record("code_files", file_data)
            
        except Exception as e:
            logger.error(f"Failed to store code file {code_file.file_path}: {e}")
            raise
    
    async def _store_code_chunk(self, chunk: CodeChunk):
        """Store code chunk in memory system."""
        try:
            # Generate embedding for chunk content
            embedding = self.memory.generate_embedding(chunk.content)
            
            # Store in database
            chunk_data = {
                "id": chunk.id,
                "file_id": chunk.file_id,
                "content": chunk.content,
                "chunk_type": chunk.chunk_type,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "embedding": embedding,
                **chunk.metadata
            }
            
            await self.memory.db.create_record("code_chunks", chunk_data)
            
            # Create relationship to parent file
            await self.memory.create_relationship(
                chunk.file_id,
                chunk.id,
                RelationshipType.HAS_CHUNK,
                created_by_agent=self.config.agent_id
            )
            
        except Exception as e:
            logger.error(f"Failed to store code chunk {chunk.id}: {e}")
            raise
    
    # ================================
    # RELATIONSHIP CREATION
    # ================================
    
    async def _create_cross_file_relationships(self):
        """Create relationships between code files and chunks."""
        try:
            # Find import relationships
            await self._create_import_relationships()
            
            # Find function call relationships
            await self._create_call_relationships()
            
            # Find inheritance relationships
            await self._create_inheritance_relationships()
            
            logger.info("Created cross-file relationships")
            
        except Exception as e:
            logger.error(f"Failed to create cross-file relationships: {e}")
    
    async def _create_import_relationships(self):
        """Create relationships based on import statements."""
        try:
            # Get all import chunks
            import_query = "SELECT * FROM code_chunks WHERE chunk_type = 'imports'"
            import_chunks = await self.memory.db.query_records(import_query)
            
            for chunk in import_chunks:
                # Parse import statements to find dependencies
                imports = self._parse_import_statements(chunk["content"])
                
                for imported_module in imports:
                    # Try to find corresponding file
                    target_file = await self._find_file_by_module_name(imported_module)
                    
                    if target_file:
                        await self.memory.create_relationship(
                            chunk["id"],
                            target_file["id"],
                            RelationshipType.IMPORTS,
                            created_by_agent=self.config.agent_id
                        )
            
        except Exception as e:
            logger.error(f"Failed to create import relationships: {e}")
    
    async def _create_call_relationships(self):
        """Create relationships based on function calls."""
        # This would analyze function calls across files
        # Implementation would depend on language-specific analysis
        pass
    
    async def _create_inheritance_relationships(self):
        """Create relationships based on class inheritance."""
        # This would analyze class inheritance patterns
        # Implementation would depend on language-specific analysis
        pass
    
    def _parse_import_statements(self, content: str) -> List[str]:
        """Parse import statements to extract module names."""
        imports = []
        
        for line in content.split('\n'):
            line = line.strip()
            
            if line.startswith('import '):
                # Handle: import module
                module = line[7:].split(' as ')[0].strip()
                imports.append(module)
            elif line.startswith('from '):
                # Handle: from module import ...
                parts = line.split(' import ')
                if len(parts) > 1:
                    module = parts[0][5:].strip()
                    imports.append(module)
        
        return imports
    
    async def _find_file_by_module_name(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Find a code file by module name."""
        try:
            # Simple heuristic: convert module.name to module/name.py or module_name.py
            possible_paths = [
                f"{module_name.replace('.', '/')}.py",
                f"{module_name.replace('.', '_')}.py",
                f"{module_name}.py"
            ]
            
            for path in possible_paths:
                query = f"SELECT * FROM code_files WHERE file_path LIKE '%{path}' LIMIT 1"
                results = await self.memory.db.query_records(query)
                if results:
                    return results[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to find file for module {module_name}: {e}")
            return None
    
    # ================================
    # REAL-TIME MONITORING
    # ================================
    
    async def handle_file_change(self, file_path: str, change_type: str):
        """Handle real-time file changes."""
        try:
            file_path_obj = Path(file_path)
            
            if change_type == "deleted":
                await self._handle_file_deletion(file_path_obj)
            else:
                await self._handle_file_modification(file_path_obj)
            
            logger.info(f"Handled file {change_type}: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to handle file change {file_path}: {e}")
    
    async def _handle_file_deletion(self, file_path: Path):
        """Handle file deletion."""
        try:
            relative_path = str(file_path.relative_to(self.codebase_path))
            
            # Remove from tracking
            if relative_path in self.file_hashes:
                del self.file_hashes[relative_path]
            if relative_path in self.indexed_files:
                self.indexed_files.remove(relative_path)
            
            # Remove from database
            query = f"DELETE FROM code_files WHERE file_path = '{relative_path}'"
            await self.memory.db.query_records(query)
            
        except Exception as e:
            logger.error(f"Failed to handle file deletion {file_path}: {e}")
    
    async def _handle_file_modification(self, file_path: Path):
        """Handle file modification or creation."""
        try:
            # Re-index the modified file
            await self._index_file(file_path)
            
        except Exception as e:
            logger.error(f"Failed to handle file modification {file_path}: {e}")
    
    # ================================
    # QUERY AND ANALYSIS
    # ================================
    
    async def search_code(
        self, 
        query: str, 
        file_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search code using semantic similarity."""
        try:
            # Generate query embedding
            query_embedding = self.memory.generate_embedding(query)
            
            # Build search conditions
            conditions = []
            if file_types:
                file_type_conditions = [f"file_type = '{ft}'" for ft in file_types]
                conditions.append(f"({' OR '.join(file_type_conditions)})")
            
            # Perform vector search
            results = await self.memory.db.vector_search(
                "code_chunks",
                query_embedding,
                limit=limit,
                threshold=0.7,
                conditions=conditions
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search code: {e}")
            return []
    
    async def analyze_code_dependencies(self, file_path: str) -> Dict[str, Any]:
        """Analyze dependencies for a specific file."""
        try:
            # Get file record
            query = f"SELECT * FROM code_files WHERE file_path = '{file_path}' LIMIT 1"
            file_results = await self.memory.db.query_records(query)
            
            if not file_results:
                return {"error": "File not found"}
            
            file_id = file_results[0]["id"]
            
            # Get relationships
            relationships = await self.memory.get_relationships(
                file_id,
                relationship_types=[RelationshipType.IMPORTS, RelationshipType.CALLS]
            )
            
            return {
                "file_path": file_path,
                "dependencies": len(relationships),
                "relationships": [r.dict() for r in relationships]
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze dependencies for {file_path}: {e}")
            return {"error": str(e)}
    
    async def get_codebase_statistics(self) -> Dict[str, Any]:
        """Get comprehensive codebase statistics."""
        try:
            stats = {}
            
            # File counts by type
            file_type_query = """
                SELECT file_type, count() as count 
                FROM code_files 
                GROUP BY file_type
            """
            file_type_results = await self.memory.db.query_records(file_type_query)
            stats["files_by_type"] = {r["file_type"]: r["count"] for r in file_type_results}
            
            # Chunk counts by type
            chunk_type_query = """
                SELECT chunk_type, count() as count 
                FROM code_chunks 
                GROUP BY chunk_type
            """
            chunk_type_results = await self.memory.db.query_records(chunk_type_query)
            stats["chunks_by_type"] = {r["chunk_type"]: r["count"] for r in chunk_type_results}
            
            # Total counts
            stats["total_files"] = len(self.indexed_files)
            stats["monitoring_active"] = self.monitoring
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get codebase statistics: {e}")
            return {"error": str(e)} 