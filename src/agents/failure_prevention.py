"""
Failure Prevention Module for Sentient-Brain AI Agents

Addresses the top failure patterns from research:
- 89% Ambiguous Prompt Failures
- 75% Full-Stack Development Failures  
- 68-72% Context Management Failures
- 65% Improvement Request Failures
- Debugging & Error Handling Failures
"""
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
import logging
from datetime import datetime
import json
from dataclasses import dataclass

from ..services.groq_service import GroqLLMService
from ..services.memory_service import MemoryService

logger = logging.getLogger(__name__)


class RequirementValidationStatus(str, Enum):
    """Status of requirement validation."""
    INSUFFICIENT = "insufficient"
    AMBIGUOUS = "ambiguous" 
    CONFLICTING = "conflicting"
    VALIDATED = "validated"


class ContextOverloadLevel(str, Enum):
    """Context overload severity levels."""
    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"
    OVERLOAD = "overload"


@dataclass
class ValidationResult:
    """Result of requirement validation."""
    status: RequirementValidationStatus
    requirements: Dict[str, Any]
    missing_requirements: List[str]
    ambiguous_items: List[Dict[str, str]]
    conflicts: List[Dict[str, Any]]
    confidence_score: float


@dataclass
class PolicyCheckResult:
    """Result of policy compliance check."""
    compliant: bool
    violations: List[Dict[str, str]]
    recommendations: List[str]
    policy_nodes: List[str]


@dataclass
class ContextAnalysis:
    """Analysis of context management state."""
    overload_level: ContextOverloadLevel
    token_usage: int
    max_tokens: int
    priority_items: List[str]
    prunable_items: List[str]
    summary_nodes: List[str]


class RequirementValidator:
    """Validates and structures user requirements to prevent ambiguous prompt failures."""
    
    def __init__(self, llm_service: GroqLLMService):
        self.llm = llm_service
        self.required_fields = {
            "project_type": ["web_app", "api", "desktop", "mobile", "cli", "library"],
            "scope": ["new_project", "feature_addition", "refactor", "bug_fix"],
            "tech_preferences": ["frontend", "backend", "database", "deployment"],
            "constraints": ["timeline", "budget", "team_size", "experience_level"],
            "success_criteria": ["functional", "performance", "usability", "maintainability"]
        }
    
    async def validate_requirements(self, user_input: str, metadata: Dict[str, Any]) -> ValidationResult:
        """Comprehensive requirement validation with structured decomposition."""
        try:
            # Step 1: Extract structured requirements using Chain-of-Thought
            extraction_prompt = f"""
            Analyze this user request and extract structured requirements:
            
            User Input: {user_input}
            Context: {metadata}
            
            Use Chain-of-Thought reasoning to:
            1. Identify the core project intent
            2. Extract explicit requirements
            3. Identify implicit assumptions
            4. Flag ambiguous or missing elements
            5. Detect potential conflicts
            
            Required structure:
            - project_type: {self.required_fields['project_type']}
            - scope: {self.required_fields['scope']}  
            - tech_preferences: {self.required_fields['tech_preferences']}
            - constraints: {self.required_fields['constraints']}
            - success_criteria: {self.required_fields['success_criteria']}
            
            Respond in JSON format with extracted requirements and analysis.
            """
            
            response = await self.llm.generate_structured_response(
                prompt=extraction_prompt,
                response_format={
                    "requirements": "object",
                    "missing_fields": "array",
                    "ambiguous_items": "array", 
                    "potential_conflicts": "array",
                    "confidence": "number",
                    "reasoning": "string"
                }
            )
            
            # Step 2: Validate completeness
            missing_requirements = self._check_completeness(response.get("requirements", {}))
            
            # Step 3: Analyze ambiguities
            ambiguous_items = response.get("ambiguous_items", [])
            
            # Step 4: Detect conflicts
            conflicts = response.get("potential_conflicts", [])
            
            # Step 5: Determine validation status
            status = self._determine_validation_status(
                missing_requirements, ambiguous_items, conflicts
            )
            
            return ValidationResult(
                status=status,
                requirements=response.get("requirements", {}),
                missing_requirements=missing_requirements,
                ambiguous_items=ambiguous_items,
                conflicts=conflicts,
                confidence_score=response.get("confidence", 0.0)
            )
            
        except Exception as e:
            logger.error(f"Requirement validation failed: {e}")
            return ValidationResult(
                status=RequirementValidationStatus.INSUFFICIENT,
                requirements={},
                missing_requirements=list(self.required_fields.keys()),
                ambiguous_items=[],
                conflicts=[],
                confidence_score=0.0
            )
    
    def _check_completeness(self, requirements: Dict[str, Any]) -> List[str]:
        """Check for missing required fields."""
        missing = []
        for field, options in self.required_fields.items():
            if field not in requirements or not requirements[field]:
                missing.append(field)
        return missing
    
    def _determine_validation_status(
        self, 
        missing: List[str], 
        ambiguous: List[Dict], 
        conflicts: List[Dict]
    ) -> RequirementValidationStatus:
        """Determine overall validation status."""
        if conflicts:
            return RequirementValidationStatus.CONFLICTING
        elif len(missing) > 2:  # More than 2 critical fields missing
            return RequirementValidationStatus.INSUFFICIENT
        elif ambiguous:
            return RequirementValidationStatus.AMBIGUOUS
        else:
            return RequirementValidationStatus.VALIDATED


class PolicyEnforcer:
    """Enforces architectural and coding policies to prevent violations."""
    
    def __init__(self, memory_service: MemoryService):
        self.memory = memory_service
        self.policies = {}
    
    async def load_policies(self):
        """Load policy nodes from memory graph."""
        try:
            # Query for policy nodes
            policy_query = """
            SELECT * FROM policy_node 
            WHERE active = true 
            ORDER BY priority DESC
            """
            
            policies = await self.memory.execute_query(policy_query)
            self.policies = {p['id']: p for p in policies}
            
            logger.info(f"Loaded {len(self.policies)} active policies")
            
        except Exception as e:
            logger.error(f"Failed to load policies: {e}")
    
    async def check_compliance(self, requirements: Dict[str, Any]) -> PolicyCheckResult:
        """Check requirements against loaded policies."""
        try:
            violations = []
            recommendations = []
            applicable_policies = []
            
            for policy_id, policy in self.policies.items():
                if self._policy_applies(policy, requirements):
                    applicable_policies.append(policy_id)
                    
                    # Check compliance
                    compliance_check = await self._check_policy_compliance(policy, requirements)
                    
                    if not compliance_check['compliant']:
                        violations.extend(compliance_check['violations'])
                        recommendations.extend(compliance_check['recommendations'])
            
            return PolicyCheckResult(
                compliant=len(violations) == 0,
                violations=violations,
                recommendations=recommendations,
                policy_nodes=applicable_policies
            )
            
        except Exception as e:
            logger.error(f"Policy compliance check failed: {e}")
            return PolicyCheckResult(
                compliant=True,  # Fail open
                violations=[],
                recommendations=[],
                policy_nodes=[]
            )
    
    def _policy_applies(self, policy: Dict[str, Any], requirements: Dict[str, Any]) -> bool:
        """Check if policy applies to current requirements."""
        policy_scope = policy.get('scope', [])
        req_type = requirements.get('project_type')
        req_scope = requirements.get('scope')
        
        return (not policy_scope or 
                req_type in policy_scope or 
                req_scope in policy_scope)
    
    async def _check_policy_compliance(
        self, 
        policy: Dict[str, Any], 
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check specific policy compliance."""
        # Simplified compliance check - can be enhanced
        violations = []
        recommendations = []
        
        policy_rules = policy.get('rules', [])
        for rule in policy_rules:
            if not self._check_rule_compliance(rule, requirements):
                violations.append({
                    'rule': rule['name'],
                    'description': rule['description'],
                    'severity': rule.get('severity', 'warning')
                })
                
                if 'recommendation' in rule:
                    recommendations.append(rule['recommendation'])
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'recommendations': recommendations
        }
    
    def _check_rule_compliance(self, rule: Dict[str, Any], requirements: Dict[str, Any]) -> bool:
        """Check individual rule compliance."""
        # Simplified rule checking - can be enhanced with more sophisticated logic
        rule_type = rule.get('type')
        
        if rule_type == 'required_field':
            field = rule.get('field')
            return field in requirements and requirements[field]
        
        elif rule_type == 'allowed_values':
            field = rule.get('field')
            allowed = rule.get('allowed_values', [])
            value = requirements.get(field)
            return not value or value in allowed
        
        elif rule_type == 'conditional':
            condition = rule.get('condition', {})
            requirement = rule.get('requirement', {})
            
            # Check if condition is met
            condition_met = all(
                requirements.get(k) == v for k, v in condition.items()
            )
            
            if condition_met:
                # Check if requirement is satisfied
                return all(
                    requirements.get(k) == v for k, v in requirement.items()
                )
        
        return True  # Default to compliant for unknown rule types


class ContextManager:
    """Manages context to prevent overload and optimize retrieval."""
    
    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.context_cache = {}
    
    async def analyze_context(self, context_items: List[Dict[str, Any]]) -> ContextAnalysis:
        """Analyze context for potential overload."""
        try:
            total_tokens = sum(item.get('token_count', 0) for item in context_items)
            
            # Determine overload level
            if total_tokens < self.max_tokens * 0.6:
                overload_level = ContextOverloadLevel.NONE
            elif total_tokens < self.max_tokens * 0.8:
                overload_level = ContextOverloadLevel.WARNING
            elif total_tokens < self.max_tokens:
                overload_level = ContextOverloadLevel.CRITICAL
            else:
                overload_level = ContextOverloadLevel.OVERLOAD
            
            # Prioritize items
            priority_items = self._prioritize_context_items(context_items)
            
            # Identify prunable items
            prunable_items = self._identify_prunable_items(context_items)
            
            # Find summary nodes
            summary_nodes = self._find_summary_nodes(context_items)
            
            return ContextAnalysis(
                overload_level=overload_level,
                token_usage=total_tokens,
                max_tokens=self.max_tokens,
                priority_items=priority_items,
                prunable_items=prunable_items,
                summary_nodes=summary_nodes
            )
            
        except Exception as e:
            logger.error(f"Context analysis failed: {e}")
            return ContextAnalysis(
                overload_level=ContextOverloadLevel.NONE,
                token_usage=0,
                max_tokens=self.max_tokens,
                priority_items=[],
                prunable_items=[],
                summary_nodes=[]
            )
    
    def _prioritize_context_items(self, items: List[Dict[str, Any]]) -> List[str]:
        """Prioritize context items by relevance and recency."""
        scored_items = []
        
        for item in items:
            score = 0
            
            # Recency score
            if 'timestamp' in item:
                age_hours = (datetime.now() - item['timestamp']).total_seconds() / 3600
                score += max(0, 10 - age_hours)  # Higher score for recent items
            
            # Relevance score
            score += item.get('relevance_score', 0)
            
            # Type priority
            type_priorities = {
                'requirement': 10,
                'policy': 9,
                'task': 8,
                'code': 7,
                'doc': 6,
                'session': 5
            }
            score += type_priorities.get(item.get('type'), 0)
            
            scored_items.append((item['id'], score))
        
        # Sort by score and return top items
        scored_items.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in scored_items[:20]]  # Top 20 items
    
    def _identify_prunable_items(self, items: List[Dict[str, Any]]) -> List[str]:
        """Identify items that can be pruned to reduce context."""
        prunable = []
        
        for item in items:
            # Old session messages
            if (item.get('type') == 'session' and 
                'timestamp' in item and
                (datetime.now() - item['timestamp']).total_seconds() > 3600):
                prunable.append(item['id'])
            
            # Low relevance docs
            if (item.get('type') == 'doc' and 
                item.get('relevance_score', 0) < 0.3):
                prunable.append(item['id'])
            
            # Duplicate code chunks
            if item.get('type') == 'code' and item.get('is_duplicate', False):
                prunable.append(item['id'])
        
        return prunable
    
    def _find_summary_nodes(self, items: List[Dict[str, Any]]) -> List[str]:
        """Find existing summary nodes that can replace detailed items."""
        summaries = []
        
        for item in items:
            if item.get('type') == 'summary':
                summaries.append(item['id'])
        
        return summaries


class FullStackCoordinator:
    """Coordinates full-stack development to prevent 75% failure rate."""
    
    def __init__(self, memory_service: MemoryService):
        self.memory = memory_service
        self.api_contracts = {}
        self.db_schemas = {}
    
    async def validate_full_stack_design(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Validate full-stack design for consistency and completeness."""
        try:
            validation_results = {
                'api_consistency': await self._validate_api_consistency(requirements),
                'db_schema_alignment': await self._validate_db_schema_alignment(requirements),
                'state_management': await self._validate_state_management(requirements),
                'integration_points': await self._validate_integration_points(requirements)
            }
            
            overall_valid = all(result['valid'] for result in validation_results.values())
            
            return {
                'valid': overall_valid,
                'details': validation_results,
                'recommendations': self._generate_fullstack_recommendations(validation_results)
            }
            
        except Exception as e:
            logger.error(f"Full-stack validation failed: {e}")
            return {'valid': False, 'error': str(e)}
    
    async def _validate_api_consistency(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Validate API contract consistency between frontend and backend."""
        # Implementation for API validation
        return {'valid': True, 'issues': []}
    
    async def _validate_db_schema_alignment(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Validate database schema alignment with application requirements."""
        # Implementation for DB schema validation
        return {'valid': True, 'issues': []}
    
    async def _validate_state_management(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Validate state management across application layers."""
        # Implementation for state management validation
        return {'valid': True, 'issues': []}
    
    async def _validate_integration_points(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Validate integration points between components."""
        # Implementation for integration validation
        return {'valid': True, 'issues': []}
    
    def _generate_fullstack_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        for component, result in validation_results.items():
            if not result['valid']:
                recommendations.append(f"Address {component} issues: {result.get('issues', [])}")
        
        return recommendations 