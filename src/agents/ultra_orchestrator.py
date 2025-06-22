"""
Ultra Orchestrator Agent - Enhanced with Failure Prevention

Master coordinator that receives user prompts, classifies intent, and delegates to specialized agents.
Enhanced with comprehensive failure prevention mechanisms to address:
- 89% Ambiguous Prompt Failures
- 75% Full-Stack Development Failures  
- 68-72% Context Management Failures
- 65% Improvement Request Failures

Features:
- Structured requirement validation
- Interactive ambiguity resolution
- Policy enforcement
- Context optimization
- Multi-agent coordination with fallbacks
"""
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import logging
from datetime import datetime
import json

from ..models.agent_models import AgentConfig, AgentMessage, AgentResponse
from ..services.groq_service import GroqLLMService
from ..services.memory_service import MemoryService
from .failure_prevention import (
    RequirementValidator, PolicyEnforcer, ContextManager, FullStackCoordinator,
    RequirementValidationStatus, ContextOverloadLevel
)
from .architect_agent import ArchitectAgent
from .codebase_agent import CodebaseAgent

logger = logging.getLogger(__name__)


class IntentClassification(str, Enum):
    """User intent classifications."""
    NEW_PROJECT = "new_project"
    CODE_EDIT = "code_edit"
    BUG_FIX = "bug_fix"
    FEATURE_ADD = "feature_add"
    REFACTOR = "refactor"
    DOC_QUERY = "doc_query"
    IMPROVEMENT = "improvement"
    AMBIGUOUS = "ambiguous"


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    INITIALIZING = "initializing"
    VALIDATING = "validating"
    EXECUTING = "executing"
    COORDINATING = "coordinating"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_CLARIFICATION = "requires_clarification"


class UltraOrchestrator:
    """
    Enhanced Ultra Orchestrator with comprehensive failure prevention.
    
    Workflow:
    1. Intent Classification & Validation
    2. Requirement Validation & Ambiguity Resolution
    3. Policy Compliance Check
    4. Context Optimization
    5. Agent Delegation & Coordination
    6. Continuous Monitoring & Fallback Handling
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
        
        # Failure prevention components
        self.requirement_validator = RequirementValidator(llm_service)
        self.policy_enforcer = PolicyEnforcer(memory_service)
        self.context_manager = ContextManager()
        self.fullstack_coordinator = FullStackCoordinator(memory_service)
        
        # Agent registry
        self.agents = {}
        self.active_workflows = {}
        
        # Workflow state
        self.current_workflow = None
        self.workflow_history = []
    
    async def initialize(self) -> AgentResponse:
        """Initialize the Ultra Orchestrator with failure prevention systems."""
        try:
            # Load policies
            await self.policy_enforcer.load_policies()
            
            # Initialize specialized agents
            await self._initialize_agents()
            
            logger.info("Ultra Orchestrator initialized with failure prevention")
            
            return AgentResponse(
                agent_id=self.config.agent_id,
                success=True,
                message="Ultra Orchestrator ready with enhanced failure prevention",
                data={
                    "failure_prevention_active": True,
                    "available_agents": list(self.agents.keys()),
                    "policies_loaded": len(self.policy_enforcer.policies)
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize Ultra Orchestrator: {e}")
            return AgentResponse(
                agent_id=self.config.agent_id,
                success=False,
                error=str(e)
            )
    
    async def process_user_request(self, message: AgentMessage) -> AgentResponse:
        """
        Process user request with comprehensive failure prevention.
        
        Enhanced workflow:
        1. Intent classification with ambiguity detection
        2. Requirement validation and interactive clarification
        3. Policy compliance verification
        4. Context optimization
        5. Agent delegation with coordination
        6. Continuous monitoring and fallback handling
        """
        try:
            workflow_id = f"workflow_{datetime.now().isoformat()}"
            self.current_workflow = {
                "id": workflow_id,
                "status": WorkflowStatus.INITIALIZING,
                "message": message,
                "start_time": datetime.now(),
                "steps": []
            }
            
            # Step 1: Intent Classification with Ambiguity Detection
            intent_result = await self._classify_intent_with_validation(message)
            self._log_workflow_step("intent_classification", intent_result)
            
            if intent_result["intent"] == IntentClassification.AMBIGUOUS:
                return await self._handle_ambiguous_intent(intent_result, message)
            
            # Step 2: Requirement Validation
            self.current_workflow["status"] = WorkflowStatus.VALIDATING
            validation_result = await self.requirement_validator.validate_requirements(
                message.content, message.metadata
            )
            self._log_workflow_step("requirement_validation", validation_result.__dict__)
            
            if validation_result.status != RequirementValidationStatus.VALIDATED:
                return await self._handle_validation_failure(validation_result, message)
            
            # Step 3: Policy Compliance Check
            policy_check = await self.policy_enforcer.check_compliance(validation_result.requirements)
            self._log_workflow_step("policy_check", policy_check.__dict__)
            
            if not policy_check.compliant:
                return await self._handle_policy_violations(policy_check, message)
            
            # Step 4: Context Optimization
            context_analysis = await self._optimize_context(message, validation_result.requirements)
            self._log_workflow_step("context_optimization", context_analysis.__dict__)
            
            if context_analysis.overload_level == ContextOverloadLevel.OVERLOAD:
                return await self._handle_context_overload(context_analysis, message)
            
            # Step 5: Agent Delegation & Coordination
            self.current_workflow["status"] = WorkflowStatus.EXECUTING
            execution_result = await self._execute_with_coordination(
                intent_result["intent"], 
                validation_result, 
                policy_check,
                context_analysis
            )
            
            self.current_workflow["status"] = WorkflowStatus.COMPLETED
            self._log_workflow_step("execution_completed", execution_result.__dict__)
            
            # Store workflow for learning
            await self._store_workflow_outcome()
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Ultra Orchestrator workflow failed: {e}")
            
            if self.current_workflow:
                self.current_workflow["status"] = WorkflowStatus.FAILED
                self._log_workflow_step("error", {"error": str(e)})
            
            # Attempt fallback recovery
            return await self._attempt_fallback_recovery(message, str(e))
    
    async def _classify_intent_with_validation(self, message: AgentMessage) -> Dict[str, Any]:
        """Enhanced intent classification with ambiguity detection."""
        try:
            classification_prompt = f"""
            Analyze this user request and classify the intent with confidence scoring:
            
            User Message: {message.content}
            Context: {message.metadata}
            
            Classification Options:
            - new_project: Starting a completely new software project
            - code_edit: Modifying existing code
            - bug_fix: Fixing identified bugs or issues
            - feature_add: Adding new features to existing project
            - refactor: Improving code structure without changing functionality
            - doc_query: Asking about documentation or how-to questions
            - improvement: General improvement requests
            - ambiguous: Request is unclear or lacks sufficient detail
            
            Provide:
            1. Primary intent classification
            2. Confidence score (0-1)
            3. Secondary intents if applicable
            4. Ambiguity indicators
            5. Required clarifications if ambiguous
            
            Respond in JSON format.
            """
            
            response = await self.llm.generate_structured_response(
                prompt=classification_prompt,
                response_format={
                    "primary_intent": "string",
                    "confidence": "number",
                    "secondary_intents": "array",
                    "ambiguity_indicators": "array",
                    "required_clarifications": "array",
                    "reasoning": "string"
                }
            )
            
            # Determine if ambiguous based on confidence and indicators
            is_ambiguous = (
                response.get("confidence", 0) < 0.7 or 
                len(response.get("ambiguity_indicators", [])) > 2 or
                response.get("primary_intent") == "ambiguous"
            )
            
            intent = IntentClassification.AMBIGUOUS if is_ambiguous else IntentClassification(
                response.get("primary_intent", "ambiguous")
            )
            
            return {
                "intent": intent,
                "confidence": response.get("confidence", 0),
                "secondary_intents": response.get("secondary_intents", []),
                "ambiguity_indicators": response.get("ambiguity_indicators", []),
                "required_clarifications": response.get("required_clarifications", []),
                "reasoning": response.get("reasoning", "")
            }
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return {
                "intent": IntentClassification.AMBIGUOUS,
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def _handle_ambiguous_intent(
        self, 
        intent_result: Dict[str, Any], 
        message: AgentMessage
    ) -> AgentResponse:
        """Handle ambiguous intent with structured clarification."""
        
        clarification_questions = await self._generate_intent_clarification_questions(
            intent_result["ambiguity_indicators"],
            intent_result["required_clarifications"]
        )
        
        self.current_workflow["status"] = WorkflowStatus.REQUIRES_CLARIFICATION
        
        return AgentResponse(
            agent_id=self.config.agent_id,
            success=True,
            message="I need to better understand what you'd like to accomplish. Let me ask some clarifying questions:",
            data={
                "status": "intent_clarification_needed",
                "ambiguity_indicators": intent_result["ambiguity_indicators"],
                "clarification_questions": clarification_questions,
                "suggested_intents": intent_result.get("secondary_intents", []),
                "workflow_id": self.current_workflow["id"],
                "next_step": "intent_clarification"
            }
        )
    
    async def _optimize_context(
        self, 
        message: AgentMessage, 
        requirements: Dict[str, Any]
    ) -> 'ContextAnalysis':
        """Optimize context to prevent overload failures."""
        try:
            # Gather relevant context items
            context_items = await self._gather_context_items(message, requirements)
            
            # Analyze context for potential overload
            context_analysis = await self.context_manager.analyze_context(context_items)
            
            # If overload detected, optimize context
            if context_analysis.overload_level in [ContextOverloadLevel.CRITICAL, ContextOverloadLevel.OVERLOAD]:
                optimized_context = await self._prune_and_summarize_context(
                    context_items, context_analysis
                )
                
                # Re-analyze optimized context
                context_analysis = await self.context_manager.analyze_context(optimized_context)
            
            return context_analysis
            
        except Exception as e:
            logger.error(f"Context optimization failed: {e}")
            return ContextAnalysis(
                overload_level=ContextOverloadLevel.NONE,
                token_usage=0,
                max_tokens=8000,
                priority_items=[],
                prunable_items=[],
                summary_nodes=[]
            )
    
    async def _execute_with_coordination(
        self,
        intent: IntentClassification,
        validation_result: 'ValidationResult',
        policy_check: 'PolicyCheckResult',
        context_analysis: 'ContextAnalysis'
    ) -> AgentResponse:
        """Execute request with multi-agent coordination and fallbacks."""
        try:
            self.current_workflow["status"] = WorkflowStatus.COORDINATING
            
            # Determine execution strategy based on intent
            if intent == IntentClassification.NEW_PROJECT:
                return await self._execute_new_project_workflow(
                    validation_result, policy_check, context_analysis
                )
            
            elif intent in [IntentClassification.CODE_EDIT, IntentClassification.BUG_FIX]:
                return await self._execute_code_modification_workflow(
                    intent, validation_result, policy_check, context_analysis
                )
            
            elif intent == IntentClassification.FEATURE_ADD:
                # Check if full-stack coordination needed
                if self._requires_fullstack_coordination(validation_result.requirements):
                    return await self._execute_fullstack_workflow(
                        validation_result, policy_check, context_analysis
                    )
                else:
                    return await self._execute_feature_addition_workflow(
                        validation_result, policy_check, context_analysis
                    )
            
            elif intent == IntentClassification.IMPROVEMENT:
                return await self._execute_improvement_workflow(
                    validation_result, policy_check, context_analysis
                )
            
            else:
                return await self._execute_generic_workflow(
                    intent, validation_result, policy_check, context_analysis
                )
                
        except Exception as e:
            logger.error(f"Execution coordination failed: {e}")
            return await self._attempt_fallback_recovery(
                self.current_workflow["message"], str(e)
            )
    
    async def _execute_fullstack_workflow(
        self,
        validation_result: 'ValidationResult',
        policy_check: 'PolicyCheckResult', 
        context_analysis: 'ContextAnalysis'
    ) -> AgentResponse:
        """Execute full-stack development workflow with enhanced coordination."""
        try:
            # Step 1: Validate full-stack design consistency
            fullstack_validation = await self.fullstack_coordinator.validate_full_stack_design(
                validation_result.requirements
            )
            
            if not fullstack_validation['valid']:
                return AgentResponse(
                    agent_id=self.config.agent_id,
                    success=True,
                    message="I detected some full-stack design issues that need to be addressed:",
                    data={
                        "status": "fullstack_validation_failed",
                        "issues": fullstack_validation['details'],
                        "recommendations": fullstack_validation['recommendations'],
                        "next_step": "design_refinement"
                    }
                )
            
            # Step 2: Coordinate specialized agents
            coordination_plan = await self._create_fullstack_coordination_plan(
                validation_result.requirements
            )
            
            # Step 3: Execute coordinated workflow
            results = await self._execute_coordinated_agents(coordination_plan)
            
            return AgentResponse(
                agent_id=self.config.agent_id,
                success=True,
                message="Full-stack development workflow completed successfully",
                data={
                    "status": "fullstack_completed",
                    "results": results,
                    "coordination_plan": coordination_plan
                }
            )
            
        except Exception as e:
            logger.error(f"Full-stack workflow failed: {e}")
            raise
    
    # Additional workflow methods...
    
    def _log_workflow_step(self, step_name: str, data: Any):
        """Log workflow step for monitoring and debugging."""
        if self.current_workflow:
            self.current_workflow["steps"].append({
                "step": step_name,
                "timestamp": datetime.now(),
                "data": data
            })
    
    async def _store_workflow_outcome(self):
        """Store workflow outcome for learning and improvement."""
        try:
            if self.current_workflow:
                await self.memory.store_workflow_outcome(self.current_workflow)
                self.workflow_history.append(self.current_workflow)
        except Exception as e:
            logger.error(f"Failed to store workflow outcome: {e}")
    
    # Placeholder methods for additional functionality
    async def _initialize_agents(self): pass
    async def _handle_validation_failure(self, validation_result, message): pass
    async def _handle_policy_violations(self, policy_check, message): pass
    async def _handle_context_overload(self, context_analysis, message): pass
    async def _attempt_fallback_recovery(self, message, error): pass
    async def _generate_intent_clarification_questions(self, indicators, clarifications): pass
    async def _gather_context_items(self, message, requirements): pass
    async def _prune_and_summarize_context(self, items, analysis): pass
    async def _execute_new_project_workflow(self, validation, policy, context): pass
    async def _execute_code_modification_workflow(self, intent, validation, policy, context): pass
    async def _execute_feature_addition_workflow(self, validation, policy, context): pass
    async def _execute_improvement_workflow(self, validation, policy, context): pass
    async def _execute_generic_workflow(self, intent, validation, policy, context): pass
    def _requires_fullstack_coordination(self, requirements): return False
    async def _create_fullstack_coordination_plan(self, requirements): return {}
    async def _execute_coordinated_agents(self, plan): return {} 