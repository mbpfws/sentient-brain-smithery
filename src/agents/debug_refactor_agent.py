"""
The Debug and Refactor Agent.

Specializes in identifying inefficiencies, bugs, and areas for code improvement.
Collaborates with other agents and produces comprehensive improvement plans.
"""
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import re
import ast
from enum import Enum

from ..models.memory_layers import Task, TaskPriority, RelationshipType
from ..models.agent_models import AgentConfig, AgentMessage, AgentResponse
from ..services.groq_service import GroqLLMService
from ..services.memory_service import MemoryService

logger = logging.getLogger(__name__)


class IssueType(str, Enum):
    """Types of code issues that can be identified."""
    BUG = "bug"
    PERFORMANCE = "performance"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"
    CODE_SMELL = "code_smell"
    ANTI_PATTERN = "anti_pattern"
    TECHNICAL_DEBT = "technical_debt"
    DOCUMENTATION = "documentation"
    TESTING = "testing"


class IssueSeverity(str, Enum):
    """Severity levels for identified issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class CodeIssue:
    """Represents a code issue found during analysis."""
    def __init__(
        self,
        id: str,
        issue_type: IssueType,
        severity: IssueSeverity,
        title: str,
        description: str,
        file_path: str,
        line_number: Optional[int] = None,
        suggestion: Optional[str] = None,
        **kwargs
    ):
        self.id = id
        self.issue_type = issue_type
        self.severity = severity
        self.title = title
        self.description = description
        self.file_path = file_path
        self.line_number = line_number
        self.suggestion = suggestion
        self.metadata = kwargs


class RefactoringPlan:
    """Represents a comprehensive refactoring plan."""
    def __init__(
        self,
        id: str,
        title: str,
        description: str,
        issues_addressed: List[str],
        estimated_effort: str,
        priority: TaskPriority,
        **kwargs
    ):
        self.id = id
        self.title = title
        self.description = description
        self.issues_addressed = issues_addressed
        self.estimated_effort = estimated_effort
        self.priority = priority
        self.metadata = kwargs


class DebugRefactorAgent:
    """
    Debug and Refactor Agent specializes in code quality improvement.
    
    Key responsibilities:
    - Identify bugs, inefficiencies, and code smells
    - Analyze code execution patterns
    - Generate comprehensive improvement plans
    - Collaborate with Architect Agent for design insights
    - Monitor continuous code quality
    """
    
    def __init__(
        self,
        config: AgentConfig,
        llm_service: GroqLLMService,
        memory_service: MemoryService
    ):
        self.config = config
        self.llm = llm_service
        self.memory = memory_service
        
        # Analysis patterns and rules
        self.analysis_patterns = self._load_analysis_patterns()
        self.tech_stack_rules = {}
        
        # Issue tracking
        self.identified_issues: List[CodeIssue] = []
        self.refactoring_plans: List[RefactoringPlan] = []
    
    async def initialize(self, project_context: Dict[str, Any]) -> AgentResponse:
        """Initialize the debug and refactor agent."""
        try:
            # Learn about the project's tech stack
            await self._learn_tech_stack(project_context)
            
            # Load tech-stack specific rules
            await self._load_tech_stack_rules()
            
            logger.info("Debug and refactor agent initialized")
            
            return AgentResponse(
                agent_id=self.config.agent_id,
                success=True,
                message="Debug and Refactor Agent initialized successfully",
                data={
                    "tech_stack_rules_loaded": len(self.tech_stack_rules),
                    "analysis_patterns": len(self.analysis_patterns)
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize debug refactor agent: {e}")
            return AgentResponse(
                agent_id=self.config.agent_id,
                success=False,
                error=str(e)
            )
    
    # ================================
    # CODE ANALYSIS
    # ================================
    
    async def analyze_codebase(self, scope: str = "full") -> AgentResponse:
        """Perform comprehensive codebase analysis."""
        try:
            analysis_results = {
                "issues_found": 0,
                "files_analyzed": 0,
                "issues_by_type": {},
                "issues_by_severity": {},
                "refactoring_plans": []
            }
            
            # Get code files to analyze
            if scope == "full":
                code_files = await self._get_all_code_files()
            else:
                code_files = await self._get_recent_code_files()
            
            # Analyze each file
            for file_data in code_files:
                file_issues = await self._analyze_code_file(file_data)
                self.identified_issues.extend(file_issues)
                analysis_results["files_analyzed"] += 1
            
            # Categorize issues
            analysis_results["issues_found"] = len(self.identified_issues)
            analysis_results["issues_by_type"] = self._categorize_issues_by_type()
            analysis_results["issues_by_severity"] = self._categorize_issues_by_severity()
            
            # Generate refactoring plans
            refactoring_plans = await self._generate_refactoring_plans()
            analysis_results["refactoring_plans"] = len(refactoring_plans)
            
            # Store results in memory
            await self._store_analysis_results(analysis_results)
            
            logger.info(f"Codebase analysis complete: {analysis_results['issues_found']} issues found")
            
            return AgentResponse(
                agent_id=self.config.agent_id,
                success=True,
                message=f"Analysis complete: {analysis_results['issues_found']} issues found",
                data=analysis_results
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze codebase: {e}")
            return AgentResponse(
                agent_id=self.config.agent_id,
                success=False,
                error=str(e)
            )
    
    async def _analyze_code_file(self, file_data: Dict[str, Any]) -> List[CodeIssue]:
        """Analyze a single code file for issues."""
        issues = []
        
        try:
            file_path = file_data["file_path"]
            content = file_data["content"]
            file_type = file_data["file_type"]
            
            # Static analysis
            static_issues = await self._static_analysis(file_path, content, file_type)
            issues.extend(static_issues)
            
            # Pattern-based analysis
            pattern_issues = await self._pattern_analysis(file_path, content, file_type)
            issues.extend(pattern_issues)
            
            # LLM-powered analysis
            llm_issues = await self._llm_analysis(file_path, content, file_type)
            issues.extend(llm_issues)
            
            # Tech-stack specific analysis
            tech_issues = await self._tech_stack_analysis(file_path, content, file_type)
            issues.extend(tech_issues)
            
            logger.debug(f"Found {len(issues)} issues in {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to analyze file {file_data.get('file_path', 'unknown')}: {e}")
        
        return issues
    
    async def _static_analysis(self, file_path: str, content: str, file_type: str) -> List[CodeIssue]:
        """Perform static code analysis."""
        issues = []
        
        try:
            if file_type == "python":
                issues.extend(await self._analyze_python_static(file_path, content))
            elif file_type in ["javascript", "typescript"]:
                issues.extend(await self._analyze_js_static(file_path, content))
            
        except Exception as e:
            logger.error(f"Static analysis failed for {file_path}: {e}")
        
        return issues
    
    async def _analyze_python_static(self, file_path: str, content: str) -> List[CodeIssue]:
        """Static analysis for Python files."""
        issues = []
        
        try:
            # Parse AST
            tree = ast.parse(content)
            
            # Check for common issues
            for node in ast.walk(tree):
                # Unused imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if not self._is_name_used(alias.name, content):
                            issues.append(CodeIssue(
                                id=f"unused_import_{alias.name}_{node.lineno}",
                                issue_type=IssueType.CODE_SMELL,
                                severity=IssueSeverity.LOW,
                                title="Unused import",
                                description=f"Import '{alias.name}' is not used",
                                file_path=file_path,
                                line_number=node.lineno,
                                suggestion=f"Remove unused import: {alias.name}"
                            ))
                
                # Long functions
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_lines = node.end_lineno - node.lineno + 1
                    if func_lines > 50:
                        issues.append(CodeIssue(
                            id=f"long_function_{node.name}_{node.lineno}",
                            issue_type=IssueType.MAINTAINABILITY,
                            severity=IssueSeverity.MEDIUM,
                            title="Long function",
                            description=f"Function '{node.name}' is {func_lines} lines long",
                            file_path=file_path,
                            line_number=node.lineno,
                            suggestion="Consider breaking this function into smaller functions"
                        ))
                
                # Too many parameters
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    param_count = len(node.args.args)
                    if param_count > 7:
                        issues.append(CodeIssue(
                            id=f"too_many_params_{node.name}_{node.lineno}",
                            issue_type=IssueType.MAINTAINABILITY,
                            severity=IssueSeverity.MEDIUM,
                            title="Too many parameters",
                            description=f"Function '{node.name}' has {param_count} parameters",
                            file_path=file_path,
                            line_number=node.lineno,
                            suggestion="Consider using a configuration object or breaking the function down"
                        ))
                
                # Deep nesting
                if isinstance(node, (ast.If, ast.For, ast.While)):
                    nesting_level = self._calculate_nesting_level(node, tree)
                    if nesting_level > 4:
                        issues.append(CodeIssue(
                            id=f"deep_nesting_{node.lineno}",
                            issue_type=IssueType.MAINTAINABILITY,
                            severity=IssueSeverity.MEDIUM,
                            title="Deep nesting",
                            description=f"Code nesting level is {nesting_level}",
                            file_path=file_path,
                            line_number=node.lineno,
                            suggestion="Consider extracting nested logic into separate functions"
                        ))
            
        except SyntaxError as e:
            issues.append(CodeIssue(
                id=f"syntax_error_{e.lineno}",
                issue_type=IssueType.BUG,
                severity=IssueSeverity.CRITICAL,
                title="Syntax error",
                description=str(e),
                file_path=file_path,
                line_number=e.lineno,
                suggestion="Fix syntax error"
            ))
        
        return issues
    
    async def _analyze_js_static(self, file_path: str, content: str) -> List[CodeIssue]:
        """Static analysis for JavaScript/TypeScript files."""
        issues = []
        
        # Simple regex-based checks (in production, would use proper AST parser)
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Console.log statements
            if 'console.log' in line and not line.strip().startswith('//'):
                issues.append(CodeIssue(
                    id=f"console_log_{i}",
                    issue_type=IssueType.CODE_SMELL,
                    severity=IssueSeverity.LOW,
                    title="Console.log statement",
                    description="Console.log statement found",
                    file_path=file_path,
                    line_number=i,
                    suggestion="Remove console.log or replace with proper logging"
                ))
            
            # Var usage (prefer let/const)
            if re.match(r'^\s*var\s+', line):
                issues.append(CodeIssue(
                    id=f"var_usage_{i}",
                    issue_type=IssueType.CODE_SMELL,
                    severity=IssueSeverity.LOW,
                    title="Use of var",
                    description="Using 'var' instead of 'let' or 'const'",
                    file_path=file_path,
                    line_number=i,
                    suggestion="Replace 'var' with 'let' or 'const'"
                ))
            
            # == usage (prefer ===)
            if '==' in line and '===' not in line and '!=' in line and '!==' not in line:
                issues.append(CodeIssue(
                    id=f"loose_equality_{i}",
                    issue_type=IssueType.BUG,
                    severity=IssueSeverity.MEDIUM,
                    title="Loose equality comparison",
                    description="Using == or != instead of === or !==",
                    file_path=file_path,
                    line_number=i,
                    suggestion="Use strict equality comparison (=== or !==)"
                ))
        
        return issues
    
    async def _pattern_analysis(self, file_path: str, content: str, file_type: str) -> List[CodeIssue]:
        """Pattern-based analysis using predefined patterns."""
        issues = []
        
        try:
            patterns = self.analysis_patterns.get(file_type, [])
            
            for pattern in patterns:
                matches = re.finditer(pattern["regex"], content, re.MULTILINE)
                
                for match in matches:
                    line_number = content[:match.start()].count('\n') + 1
                    
                    issues.append(CodeIssue(
                        id=f"pattern_{pattern['name']}_{line_number}",
                        issue_type=IssueType(pattern["issue_type"]),
                        severity=IssueSeverity(pattern["severity"]),
                        title=pattern["title"],
                        description=pattern["description"],
                        file_path=file_path,
                        line_number=line_number,
                        suggestion=pattern.get("suggestion", "")
                    ))
            
        except Exception as e:
            logger.error(f"Pattern analysis failed for {file_path}: {e}")
        
        return issues
    
    async def _llm_analysis(self, file_path: str, content: str, file_type: str) -> List[CodeIssue]:
        """LLM-powered code analysis."""
        issues = []
        
        try:
            # Limit content size for LLM analysis
            if len(content) > 5000:
                content = content[:5000] + "\n... (truncated)"
            
            analysis_prompt = f"""
            Analyze the following {file_type} code for potential issues:
            
            File: {file_path}
            
            ```{file_type}
            {content}
            ```
            
            Look for:
            1. Potential bugs or logical errors
            2. Security vulnerabilities
            3. Performance issues
            4. Code smells and anti-patterns
            5. Maintainability concerns
            
            For each issue found, provide:
            - Type of issue
            - Severity (critical, high, medium, low)
            - Description
            - Line number (if applicable)
            - Suggested fix
            
            Format as JSON array of issues.
            """
            
            response = await self.llm.generate_structured_response(
                prompt=analysis_prompt,
                response_format={
                    "issues": [
                        {
                            "type": "string",
                            "severity": "string", 
                            "title": "string",
                            "description": "string",
                            "line_number": "number",
                            "suggestion": "string"
                        }
                    ]
                }
            )
            
            # Convert LLM response to CodeIssue objects
            for issue_data in response.get("issues", []):
                try:
                    issue = CodeIssue(
                        id=f"llm_{issue_data.get('type', 'unknown')}_{issue_data.get('line_number', 0)}",
                        issue_type=IssueType(issue_data.get("type", "code_smell")),
                        severity=IssueSeverity(issue_data.get("severity", "medium")),
                        title=issue_data.get("title", "LLM-identified issue"),
                        description=issue_data.get("description", ""),
                        file_path=file_path,
                        line_number=issue_data.get("line_number"),
                        suggestion=issue_data.get("suggestion", "")
                    )
                    issues.append(issue)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Invalid issue data from LLM: {e}")
            
        except Exception as e:
            logger.error(f"LLM analysis failed for {file_path}: {e}")
        
        return issues
    
    async def _tech_stack_analysis(self, file_path: str, content: str, file_type: str) -> List[CodeIssue]:
        """Tech-stack specific analysis."""
        issues = []
        
        try:
            # Get tech-stack specific rules
            rules = self.tech_stack_rules.get(file_type, [])
            
            for rule in rules:
                if rule["type"] == "regex":
                    matches = re.finditer(rule["pattern"], content, re.MULTILINE)
                    
                    for match in matches:
                        line_number = content[:match.start()].count('\n') + 1
                        
                        issues.append(CodeIssue(
                            id=f"tech_{rule['name']}_{line_number}",
                            issue_type=IssueType(rule["issue_type"]),
                            severity=IssueSeverity(rule["severity"]),
                            title=rule["title"],
                            description=rule["description"],
                            file_path=file_path,
                            line_number=line_number,
                            suggestion=rule.get("suggestion", "")
                        ))
            
        except Exception as e:
            logger.error(f"Tech-stack analysis failed for {file_path}: {e}")
        
        return issues
    
    # ================================
    # REFACTORING PLAN GENERATION
    # ================================
    
    async def _generate_refactoring_plans(self) -> List[RefactoringPlan]:
        """Generate comprehensive refactoring plans based on identified issues."""
        plans = []
        
        try:
            # Group issues by file and type
            issues_by_file = self._group_issues_by_file()
            issues_by_type = self._group_issues_by_type()
            
            # Generate file-specific refactoring plans
            for file_path, file_issues in issues_by_file.items():
                if len(file_issues) >= 3:  # Only create plan if multiple issues
                    plan = await self._create_file_refactoring_plan(file_path, file_issues)
                    if plan:
                        plans.append(plan)
            
            # Generate type-specific refactoring plans
            for issue_type, type_issues in issues_by_type.items():
                if len(type_issues) >= 5:  # Create plan for widespread issues
                    plan = await self._create_type_refactoring_plan(issue_type, type_issues)
                    if plan:
                        plans.append(plan)
            
            # Generate architectural refactoring plans
            arch_plan = await self._create_architectural_refactoring_plan()
            if arch_plan:
                plans.append(arch_plan)
            
            self.refactoring_plans = plans
            
        except Exception as e:
            logger.error(f"Failed to generate refactoring plans: {e}")
        
        return plans
    
    async def _create_file_refactoring_plan(
        self, 
        file_path: str, 
        issues: List[CodeIssue]
    ) -> Optional[RefactoringPlan]:
        """Create a refactoring plan for a specific file."""
        try:
            high_severity_count = len([i for i in issues if i.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH]])
            
            plan = RefactoringPlan(
                id=f"file_refactor_{file_path.replace('/', '_')}",
                title=f"Refactor {file_path}",
                description=f"Address {len(issues)} issues in {file_path}",
                issues_addressed=[issue.id for issue in issues],
                estimated_effort="2-4 hours" if high_severity_count > 2 else "1-2 hours",
                priority=TaskPriority.HIGH if high_severity_count > 2 else TaskPriority.MEDIUM,
                file_path=file_path,
                issue_count=len(issues),
                high_severity_count=high_severity_count
            )
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to create file refactoring plan for {file_path}: {e}")
            return None
    
    async def _create_type_refactoring_plan(
        self, 
        issue_type: IssueType, 
        issues: List[CodeIssue]
    ) -> Optional[RefactoringPlan]:
        """Create a refactoring plan for a specific issue type."""
        try:
            affected_files = set(issue.file_path for issue in issues)
            
            plan = RefactoringPlan(
                id=f"type_refactor_{issue_type.value}",
                title=f"Address {issue_type.value} issues",
                description=f"Fix {len(issues)} {issue_type.value} issues across {len(affected_files)} files",
                issues_addressed=[issue.id for issue in issues],
                estimated_effort=f"{len(affected_files)} hours",
                priority=TaskPriority.HIGH if issue_type in [IssueType.BUG, IssueType.SECURITY] else TaskPriority.MEDIUM,
                issue_type=issue_type.value,
                affected_files=list(affected_files),
                issue_count=len(issues)
            )
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to create type refactoring plan for {issue_type}: {e}")
            return None
    
    async def _create_architectural_refactoring_plan(self) -> Optional[RefactoringPlan]:
        """Create an architectural refactoring plan based on overall analysis."""
        try:
            # Analyze for architectural issues
            arch_issues = [
                issue for issue in self.identified_issues 
                if issue.issue_type in [IssueType.ANTI_PATTERN, IssueType.TECHNICAL_DEBT]
            ]
            
            if len(arch_issues) < 3:
                return None
            
            # Generate architectural recommendations using LLM
            arch_prompt = f"""
            Based on the following architectural issues found in the codebase:
            
            {[f"- {issue.title}: {issue.description}" for issue in arch_issues[:10]]}
            
            Generate a comprehensive architectural refactoring plan that addresses:
            1. Overall code organization
            2. Design pattern improvements
            3. Technical debt reduction
            4. Long-term maintainability
            
            Provide specific, actionable recommendations.
            """
            
            response = await self.llm.generate_structured_response(
                prompt=arch_prompt,
                response_format={
                    "title": "string",
                    "description": "string",
                    "recommendations": "array",
                    "estimated_effort": "string",
                    "priority": "string"
                }
            )
            
            plan = RefactoringPlan(
                id="architectural_refactor",
                title=response.get("title", "Architectural Refactoring"),
                description=response.get("description", "Comprehensive architectural improvements"),
                issues_addressed=[issue.id for issue in arch_issues],
                estimated_effort=response.get("estimated_effort", "1-2 weeks"),
                priority=TaskPriority.MEDIUM,
                recommendations=response.get("recommendations", []),
                issue_count=len(arch_issues)
            )
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to create architectural refactoring plan: {e}")
            return None
    
    # ================================
    # MEMORY INTEGRATION
    # ================================
    
    async def _store_analysis_results(self, results: Dict[str, Any]):
        """Store analysis results in memory."""
        try:
            # Store individual issues
            for issue in self.identified_issues:
                await self._store_code_issue(issue)
            
            # Store refactoring plans as tasks
            for plan in self.refactoring_plans:
                await self._store_refactoring_plan(plan)
            
            # Store overall analysis summary
            await self._store_analysis_summary(results)
            
        except Exception as e:
            logger.error(f"Failed to store analysis results: {e}")
    
    async def _store_code_issue(self, issue: CodeIssue):
        """Store a code issue in memory."""
        try:
            issue_data = {
                "id": issue.id,
                "issue_type": issue.issue_type.value,
                "severity": issue.severity.value,
                "title": issue.title,
                "description": issue.description,
                "file_path": issue.file_path,
                "line_number": issue.line_number,
                "suggestion": issue.suggestion,
                "discovered_at": datetime.utcnow(),
                "discovered_by_agent": self.config.agent_id,
                **issue.metadata
            }
            
            await self.memory.db.create_record("code_issues", issue_data)
            
        except Exception as e:
            logger.error(f"Failed to store code issue {issue.id}: {e}")
    
    async def _store_refactoring_plan(self, plan: RefactoringPlan):
        """Store a refactoring plan as a task."""
        try:
            task = Task(
                title=plan.title,
                description=plan.description,
                priority=plan.priority,
                domain_tags=["refactoring", "code_quality"],
                tech_stack_tags=[],
                estimated_hours=self._parse_effort_to_hours(plan.estimated_effort),
                created_by_agent=self.config.agent_id
            )
            
            task_id = await self.memory.create_task(task)
            
            # Link task to issues
            for issue_id in plan.issues_addressed:
                await self.memory.create_relationship(
                    task_id,
                    issue_id,
                    RelationshipType.ADDRESSES,
                    created_by_agent=self.config.agent_id
                )
            
        except Exception as e:
            logger.error(f"Failed to store refactoring plan {plan.id}: {e}")
    
    async def _store_analysis_summary(self, results: Dict[str, Any]):
        """Store analysis summary."""
        try:
            summary_data = {
                "id": f"analysis_{datetime.utcnow().isoformat()}",
                "analysis_date": datetime.utcnow(),
                "files_analyzed": results["files_analyzed"],
                "issues_found": results["issues_found"],
                "issues_by_type": results["issues_by_type"],
                "issues_by_severity": results["issues_by_severity"],
                "refactoring_plans_created": results["refactoring_plans"],
                "analyzed_by_agent": self.config.agent_id
            }
            
            await self.memory.db.create_record("analysis_summaries", summary_data)
            
        except Exception as e:
            logger.error(f"Failed to store analysis summary: {e}")
    
    # ================================
    # UTILITY METHODS
    # ================================
    
    def _load_analysis_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load predefined analysis patterns."""
        return {
            "python": [
                {
                    "name": "sql_injection",
                    "regex": r"execute\(['\"].*%.*['\"]",
                    "issue_type": "security",
                    "severity": "high",
                    "title": "Potential SQL injection",
                    "description": "String formatting in SQL query may lead to injection",
                    "suggestion": "Use parameterized queries"
                },
                {
                    "name": "hardcoded_password",
                    "regex": r"password\s*=\s*['\"][^'\"]+['\"]",
                    "issue_type": "security",
                    "severity": "critical",
                    "title": "Hardcoded password",
                    "description": "Password appears to be hardcoded",
                    "suggestion": "Use environment variables or secure configuration"
                }
            ],
            "javascript": [
                {
                    "name": "eval_usage",
                    "regex": r"\beval\s*\(",
                    "issue_type": "security",
                    "severity": "high",
                    "title": "Use of eval()",
                    "description": "eval() can be dangerous and should be avoided",
                    "suggestion": "Use safer alternatives like JSON.parse() or Function constructor"
                }
            ]
        }
    
    async def _learn_tech_stack(self, project_context: Dict[str, Any]):
        """Learn about the project's technology stack."""
        try:
            # Extract tech stack from project context
            tech_stack = project_context.get("tech_stack", [])
            
            # Query memory for additional tech stack information
            tech_query = "SELECT DISTINCT tech_stack FROM grand_plans WHERE tech_stack IS NOT NULL"
            tech_results = await self.memory.db.query_records(tech_query)
            
            for result in tech_results:
                tech_stack.extend(result.get("tech_stack", []))
            
            # Store learned tech stack
            self.tech_stack = list(set(tech_stack))
            
        except Exception as e:
            logger.error(f"Failed to learn tech stack: {e}")
            self.tech_stack = []
    
    async def _load_tech_stack_rules(self):
        """Load technology-specific analysis rules."""
        # This would load rules based on detected technologies
        # For now, using basic rules
        self.tech_stack_rules = {
            "python": [
                {
                    "name": "django_debug",
                    "type": "regex",
                    "pattern": r"DEBUG\s*=\s*True",
                    "issue_type": "security",
                    "severity": "high",
                    "title": "Debug mode enabled",
                    "description": "Django DEBUG should be False in production",
                    "suggestion": "Set DEBUG = False for production"
                }
            ]
        }
    
    def _is_name_used(self, name: str, content: str) -> bool:
        """Check if a name is used in the content."""
        # Simple check - in production would use proper AST analysis
        lines = content.split('\n')
        for line in lines:
            if name in line and not line.strip().startswith('import') and not line.strip().startswith('from'):
                return True
        return False
    
    def _calculate_nesting_level(self, node: ast.AST, tree: ast.AST) -> int:
        """Calculate nesting level of a node."""
        # Simplified nesting calculation
        level = 0
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                if child == node:
                    if isinstance(parent, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                        level += 1
        return level
    
    def _categorize_issues_by_type(self) -> Dict[str, int]:
        """Categorize issues by type."""
        categories = {}
        for issue in self.identified_issues:
            issue_type = issue.issue_type.value
            categories[issue_type] = categories.get(issue_type, 0) + 1
        return categories
    
    def _categorize_issues_by_severity(self) -> Dict[str, int]:
        """Categorize issues by severity."""
        categories = {}
        for issue in self.identified_issues:
            severity = issue.severity.value
            categories[severity] = categories.get(severity, 0) + 1
        return categories
    
    def _group_issues_by_file(self) -> Dict[str, List[CodeIssue]]:
        """Group issues by file path."""
        groups = {}
        for issue in self.identified_issues:
            file_path = issue.file_path
            if file_path not in groups:
                groups[file_path] = []
            groups[file_path].append(issue)
        return groups
    
    def _group_issues_by_type(self) -> Dict[IssueType, List[CodeIssue]]:
        """Group issues by type."""
        groups = {}
        for issue in self.identified_issues:
            issue_type = issue.issue_type
            if issue_type not in groups:
                groups[issue_type] = []
            groups[issue_type].append(issue)
        return groups
    
    def _parse_effort_to_hours(self, effort_str: str) -> float:
        """Parse effort string to hours."""
        # Simple parsing - could be enhanced
        if "hour" in effort_str:
            numbers = re.findall(r'\d+', effort_str)
            if numbers:
                return float(numbers[0])
        elif "day" in effort_str:
            numbers = re.findall(r'\d+', effort_str)
            if numbers:
                return float(numbers[0]) * 8
        elif "week" in effort_str:
            numbers = re.findall(r'\d+', effort_str)
            if numbers:
                return float(numbers[0]) * 40
        return 4.0  # Default
    
    async def _get_all_code_files(self) -> List[Dict[str, Any]]:
        """Get all code files from memory."""
        try:
            query = "SELECT * FROM code_files"
            return await self.memory.db.query_records(query)
        except Exception as e:
            logger.error(f"Failed to get code files: {e}")
            return []
    
    async def _get_recent_code_files(self) -> List[Dict[str, Any]]:
        """Get recently modified code files."""
        try:
            # Get files modified in last 7 days
            query = """
                SELECT * FROM code_files 
                WHERE last_modified > time::now() - 7d
                ORDER BY last_modified DESC
            """
            return await self.memory.db.query_records(query)
        except Exception as e:
            logger.error(f"Failed to get recent code files: {e}")
            return [] 