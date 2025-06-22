"""
The Architect Agent - Design & Planning Specialist.

Responsible for conceptualizing, designing, and detailing project architecture.
Enhanced with failure prevention mechanisms to address:
- 89% Ambiguous Prompt Failures
- 75% Full-Stack Development Failures
- 68-72% Context Management Failures

Operates in two modes based on user proficiency:
1. Guided Enhancement (novice/intermediate users)
2. Standardized Optimization (experienced users)
"""
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import logging
from datetime import datetime
import json

from ..models.memory_layers import GrandPlan, Task, Milestone, PlanStatus, TaskPriority
from ..models.agent_models import AgentConfig, AgentMessage, AgentResponse
from ..services.groq_service import GroqLLMService
from ..services.memory_service import MemoryService

logger = logging.getLogger(__name__)


class ArchitectMode(str, Enum):
    """Operating modes for the Architect Agent."""
    GUIDED_ENHANCEMENT = "guided_enhancement"
    STANDARDIZED_OPTIMIZATION = "standardized_optimization"


class UserProficiency(str, Enum):
    """User technical proficiency levels."""
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    EXPERIENCED = "experienced"
    EXPERT = "expert"


class RequirementValidationStatus(str, Enum):
    """Status of requirement validation."""
    INSUFFICIENT = "insufficient"
    AMBIGUOUS = "ambiguous"
    CONFLICTING = "conflicting"
    VALIDATED = "validated"


class ArchitectAgent:
    """
    The Architect Agent specializes in design and planning.
    Enhanced with failure prevention mechanisms:
    - Structured requirement validation
    - Interactive ambiguity resolution
    - Policy enforcement nodes
    - Full-stack coordination patterns
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
        self.current_mode = None
        self.user_proficiency = None
        
        # Failure prevention components
        self.requirement_validator = RequirementValidator(llm_service)
        self.policy_enforcer = PolicyEnforcer(memory_service)
        self.context_manager = ArchitectContextManager()
        
        # Mode-specific prompts
        self.mode_prompts = {
            ArchitectMode.GUIDED_ENHANCEMENT: self._get_guided_enhancement_prompt(),
            ArchitectMode.STANDARDIZED_OPTIMIZATION: self._get_standardized_optimization_prompt()
        }
    
    async def initialize(self, user_context: Dict[str, Any]) -> AgentResponse:
        """Initialize the architect agent and determine operating mode."""
        try:
            # Load architectural policies from memory
            await self.policy_enforcer.load_policies()
            
            # Assess user proficiency from initial context
            self.user_proficiency = await self._assess_user_proficiency(user_context)
            
            # Determine operating mode
            if self.user_proficiency in [UserProficiency.NOVICE, UserProficiency.INTERMEDIATE]:
                self.current_mode = ArchitectMode.GUIDED_ENHANCEMENT
            else:
                self.current_mode = ArchitectMode.STANDARDIZED_OPTIMIZATION
            
            logger.info(f"Architect agent initialized in {self.current_mode} mode for {self.user_proficiency} user")
            
            return AgentResponse(
                agent_id=self.config.agent_id,
                success=True,
                message=f"Architect Agent ready in {self.current_mode} mode with failure prevention enabled",
                data={
                    "mode": self.current_mode,
                    "user_proficiency": self.user_proficiency,
                    "failure_prevention_active": True
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize architect agent: {e}")
            return AgentResponse(
                agent_id=self.config.agent_id,
                success=False,
                error=str(e)
            )
    
    async def process_design_request(self, message: AgentMessage) -> AgentResponse:
        """Process a design request with comprehensive failure prevention."""
        try:
            # Step 1: Validate requirements to prevent ambiguous prompt failures
            validation_result = await self.requirement_validator.validate_requirements(
                message.content, 
                message.metadata
            )
            
            if validation_result.status != RequirementValidationStatus.VALIDATED:
                return await self._handle_requirement_issues(validation_result, message)
            
            # Step 2: Check policy compliance
            policy_check = await self.policy_enforcer.check_compliance(validation_result.requirements)
            if not policy_check.compliant:
                return await self._handle_policy_violations(policy_check, message)
            
            # Step 3: Process with appropriate mode
            if self.current_mode == ArchitectMode.GUIDED_ENHANCEMENT:
                return await self._guided_enhancement_flow(message, validation_result)
            else:
                return await self._standardized_optimization_flow(message, validation_result)
                
        except Exception as e:
            logger.error(f"Failed to process design request: {e}")
            return AgentResponse(
                agent_id=self.config.agent_id,
                success=False,
                error=str(e)
            )
    
    async def _handle_requirement_issues(
        self, 
        validation_result: 'ValidationResult', 
        message: AgentMessage
    ) -> AgentResponse:
        """Handle requirement validation issues with interactive clarification."""
        
        if validation_result.status == RequirementValidationStatus.INSUFFICIENT:
            clarification_questions = await self._generate_requirement_questions(
                validation_result.missing_requirements
            )
            
            return AgentResponse(
                agent_id=self.config.agent_id,
                success=True,
                message="I need more information to create a robust design. Let me ask some specific questions:",
                data={
                    "status": "clarification_needed",
                    "questions": clarification_questions,
                    "missing_requirements": validation_result.missing_requirements,
                    "next_step": "requirement_clarification"
                }
            )
        
        elif validation_result.status == RequirementValidationStatus.AMBIGUOUS:
            disambiguation_options = await self._generate_disambiguation_options(
                validation_result.ambiguous_items
            )
            
            return AgentResponse(
                agent_id=self.config.agent_id,
                success=True,
                message="I found some ambiguities in your requirements. Please help me clarify:",
                data={
                    "status": "disambiguation_needed",
                    "ambiguities": validation_result.ambiguous_items,
                    "options": disambiguation_options,
                    "next_step": "disambiguation"
                }
            )
        
        elif validation_result.status == RequirementValidationStatus.CONFLICTING:
            conflict_resolution = await self._generate_conflict_resolution(
                validation_result.conflicts
            )
            
            return AgentResponse(
                agent_id=self.config.agent_id,
                success=True,
                message="I detected conflicting requirements. Let's resolve these:",
                data={
                    "status": "conflict_resolution_needed",
                    "conflicts": validation_result.conflicts,
                    "resolution_options": conflict_resolution,
                    "next_step": "conflict_resolution"
                }
            )
        
        return AgentResponse(
            agent_id=self.config.agent_id,
            success=False,
            error="Unknown requirement validation status"
        )

    async def _handle_policy_violations(
        self, 
        policy_check: 'PolicyCheckResult', 
        message: AgentMessage
    ) -> AgentResponse:
        """Handle policy compliance violations."""
        
        return AgentResponse(
            agent_id=self.config.agent_id,
            success=True,
            message="I found some policy compliance issues that need to be addressed:",
            data={
                "status": "policy_violation",
                "violations": policy_check.violations,
                "recommendations": policy_check.recommendations,
                "next_step": "policy_compliance"
            }
        )
    
    # ================================
    # USER PROFICIENCY ASSESSMENT
    # ================================
    
    async def _assess_user_proficiency(self, user_context: Dict[str, Any]) -> UserProficiency:
        """Assess user's technical proficiency from context."""
        try:
            assessment_prompt = f"""
            Analyze the following user context and determine their technical proficiency level:
            
            Context: {user_context}
            
            Consider:
            - Technical terminology used
            - Complexity of concepts mentioned
            - Specificity of requirements
            - Experience indicators
            
            Respond with one of: novice, intermediate, experienced, expert
            
            Provide brief reasoning for your assessment.
            """
            
            response = await self.llm.generate_structured_response(
                prompt=assessment_prompt,
                response_format={
                    "proficiency": "string",
                    "reasoning": "string",
                    "confidence": "number"
                }
            )
            
            proficiency_map = {
                "novice": UserProficiency.NOVICE,
                "intermediate": UserProficiency.INTERMEDIATE,
                "experienced": UserProficiency.EXPERIENCED,
                "expert": UserProficiency.EXPERT
            }
            
            return proficiency_map.get(response.get("proficiency", "intermediate"), UserProficiency.INTERMEDIATE)
            
        except Exception as e:
            logger.error(f"Failed to assess user proficiency: {e}")
            return UserProficiency.INTERMEDIATE  # Default to intermediate
    
    # ================================
    # GUIDED ENHANCEMENT MODE
    # ================================
    
    async def _guided_enhancement_flow(self, message: AgentMessage, validation_result: 'ValidationResult') -> AgentResponse:
        """
        Mode 1: Guided Enhancement for novice/intermediate users.
        Collaborative, turn-based interaction with structured guidance.
        """
        try:
            # Extract user's initial idea
            user_idea = message.content
            
            # Generate clarifying questions
            clarification_response = await self._generate_clarifying_questions(user_idea)
            
            # If we have enough information, proceed to enhancement
            if clarification_response.get("sufficient_info", False):
                return await self._enhance_user_idea(user_idea, message.metadata)
            else:
                return AgentResponse(
                    agent_id=self.config.agent_id,
                    success=True,
                    message="I need to understand your idea better. Let me ask some clarifying questions.",
                    data={
                        "questions": clarification_response.get("questions", []),
                        "suggestions": clarification_response.get("suggestions", []),
                        "next_step": "clarification"
                    }
                )
                
        except Exception as e:
            logger.error(f"Failed guided enhancement flow: {e}")
            return AgentResponse(
                agent_id=self.config.agent_id,
                success=False,
                error=str(e)
            )
    
    async def _generate_clarifying_questions(self, user_idea: str) -> Dict[str, Any]:
        """Generate structured clarifying questions for user idea."""
        try:
            prompt = f"""
            User Idea: {user_idea}
            
            As an expert software architect in guided mode, analyze this idea and:
            
            1. Determine if you have sufficient information to proceed with design
            2. If not, generate 3-5 clarifying questions that will help understand:
               - The problem being solved
               - Target users/audience
               - Technical constraints
               - Success criteria
               - Preferred technologies (if any)
            
            3. Provide helpful suggestions or alternatives if the idea seems unclear
            
            Format your response as structured data.
            """
            
            response = await self.llm.generate_structured_response(
                prompt=prompt,
                response_format={
                    "sufficient_info": "boolean",
                    "questions": "array",
                    "suggestions": "array",
                    "assessment": "string"
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate clarifying questions: {e}")
            return {"sufficient_info": True, "questions": [], "suggestions": []}
    
    async def _enhance_user_idea(self, user_idea: str, metadata: Dict[str, Any]) -> AgentResponse:
        """Enhance and expand user's idea with architectural guidance."""
        try:
            # Research best practices and alternatives
            research_results = await self._research_best_practices(user_idea)
            
            # Generate enhanced architecture
            enhancement_prompt = f"""
            Original User Idea: {user_idea}
            
            Research Results: {research_results}
            
            As an expert architect, enhance this idea by:
            
            1. **Problem Definition**: Clearly define the problem being solved
            2. **Solution Architecture**: Propose a well-structured solution
            3. **Technology Recommendations**: Suggest appropriate, accessible technologies
            4. **Implementation Phases**: Break down into manageable phases
            5. **Success Metrics**: Define how success will be measured
            6. **Risk Mitigation**: Identify potential risks and mitigation strategies
            
            Focus on:
            - Accessibility for the user's skill level
            - Modern, maintainable approaches
            - Clear next steps
            - Learning opportunities
            
            Provide a comprehensive but approachable architectural plan.
            """
            
            enhanced_response = await self.llm.generate_structured_response(
                prompt=enhancement_prompt,
                response_format={
                    "problem_definition": "string",
                    "solution_architecture": "string",
                    "technology_stack": "array",
                    "implementation_phases": "array",
                    "success_metrics": "array",
                    "risks_and_mitigation": "array",
                    "next_steps": "array",
                    "learning_resources": "array"
                }
            )
            
            # Create grand plan in memory
            grand_plan = await self._create_grand_plan_from_enhancement(enhanced_response, user_idea)
            
            return AgentResponse(
                agent_id=self.config.agent_id,
                success=True,
                message="I've enhanced your idea with a comprehensive architectural plan!",
                data={
                    "enhanced_plan": enhanced_response,
                    "grand_plan_id": grand_plan,
                    "mode": "guided_enhancement",
                    "next_action": "review_and_approve"
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to enhance user idea: {e}")
            return AgentResponse(
                agent_id=self.config.agent_id,
                success=False,
                error=str(e)
            )
    
    # ================================
    # STANDARDIZED OPTIMIZATION MODE
    # ================================
    
    async def _standardized_optimization_flow(self, message: AgentMessage, validation_result: 'ValidationResult') -> AgentResponse:
        """
        Mode 2: Standardized Optimization for experienced users.
        Focus on feasibility, optimization, and best-in-class alternatives.
        """
        try:
            user_concept = message.content
            
            # Conduct deep feasibility analysis
            feasibility_analysis = await self._analyze_feasibility(user_concept)
            
            # Research alternatives and optimizations
            optimization_results = await self._research_optimizations(user_concept)
            
            # Generate optimized architecture
            optimized_response = await self._generate_optimized_architecture(
                user_concept, 
                feasibility_analysis, 
                optimization_results
            )
            
            # Create comprehensive plan
            grand_plan = await self._create_grand_plan_from_optimization(optimized_response, user_concept)
            
            return AgentResponse(
                agent_id=self.config.agent_id,
                success=True,
                message="I've analyzed and optimized your concept with industry best practices.",
                data={
                    "feasibility_analysis": feasibility_analysis,
                    "optimized_architecture": optimized_response,
                    "grand_plan_id": grand_plan,
                    "mode": "standardized_optimization",
                    "next_action": "technical_review"
                }
            )
            
        except Exception as e:
            logger.error(f"Failed standardized optimization flow: {e}")
            return AgentResponse(
                agent_id=self.config.agent_id,
                success=False,
                error=str(e)
            )
    
    async def _analyze_feasibility(self, concept: str) -> Dict[str, Any]:
        """Conduct deep feasibility analysis for experienced users."""
        try:
            feasibility_prompt = f"""
            Technical Concept: {concept}
            
            Conduct a comprehensive feasibility analysis covering:
            
            1. **Technical Feasibility**:
               - Architecture complexity assessment
               - Technology maturity evaluation
               - Integration challenges
               - Scalability considerations
               
            2. **Resource Requirements**:
               - Development effort estimation
               - Infrastructure needs
               - Skill requirements
               - Third-party dependencies
               
            3. **Risk Assessment**:
               - Technical risks
               - Market/adoption risks
               - Maintenance challenges
               - Security considerations
               
            4. **Alternative Approaches**:
               - Simpler alternatives
               - More robust alternatives
               - Hybrid approaches
               
            Provide detailed, technical analysis suitable for experienced developers.
            """
            
            response = await self.llm.generate_structured_response(
                prompt=feasibility_prompt,
                response_format={
                    "technical_feasibility": {
                        "complexity_score": "number",
                        "assessment": "string",
                        "challenges": "array"
                    },
                    "resource_requirements": {
                        "effort_estimate": "string",
                        "infrastructure": "array",
                        "skills_needed": "array"
                    },
                    "risk_assessment": {
                        "high_risks": "array",
                        "medium_risks": "array",
                        "mitigation_strategies": "array"
                    },
                    "alternatives": "array",
                    "recommendation": "string"
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed feasibility analysis: {e}")
            return {}
    
    async def _research_optimizations(self, concept: str) -> Dict[str, Any]:
        """Research optimizations and best-in-class alternatives."""
        try:
            # This would integrate with web search or knowledge bases
            # For now, using LLM knowledge
            
            optimization_prompt = f"""
            Technical Concept: {concept}
            
            Research and recommend optimizations based on:
            
            1. **Industry Best Practices**:
               - Current industry standards
               - Proven architectural patterns
               - Performance optimization techniques
               
            2. **Technology Stack Optimization**:
               - Best-in-class technology choices
               - Modern alternatives to legacy approaches
               - Emerging technologies to consider
               
            3. **Architecture Patterns**:
               - Scalable design patterns
               - Maintainability improvements
               - Security-first approaches
               
            4. **Performance Considerations**:
               - Bottleneck identification
               - Optimization strategies
               - Monitoring and observability
               
            Focus on practical, proven optimizations for production systems.
            """
            
            response = await self.llm.generate_structured_response(
                prompt=optimization_prompt,
                response_format={
                    "best_practices": "array",
                    "technology_optimizations": "array",
                    "architecture_patterns": "array",
                    "performance_optimizations": "array",
                    "security_enhancements": "array",
                    "monitoring_strategy": "string"
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed optimization research: {e}")
            return {}
    
    async def _generate_optimized_architecture(
        self, 
        concept: str, 
        feasibility: Dict[str, Any], 
        optimizations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate final optimized architecture specification."""
        try:
            architecture_prompt = f"""
            Original Concept: {concept}
            
            Feasibility Analysis: {feasibility}
            
            Optimization Research: {optimizations}
            
            Generate a comprehensive, optimized architecture specification:
            
            1. **Executive Summary**
            2. **System Architecture**
            3. **Technology Stack** (with justifications)
            4. **Data Architecture**
            5. **Security Architecture**
            6. **Deployment Architecture**
            7. **Development Workflow**
            8. **Testing Strategy**
            9. **Monitoring & Observability**
            10. **Scalability Plan**
            
            Ensure the architecture is:
            - Production-ready
            - Scalable and maintainable
            - Security-first
            - Performance-optimized
            - Well-documented
            """
            
            response = await self.llm.generate_structured_response(
                prompt=architecture_prompt,
                response_format={
                    "executive_summary": "string",
                    "system_architecture": "string",
                    "technology_stack": {
                        "frontend": "array",
                        "backend": "array", 
                        "database": "array",
                        "infrastructure": "array",
                        "monitoring": "array"
                    },
                    "data_architecture": "string",
                    "security_architecture": "string",
                    "deployment_architecture": "string",
                    "development_workflow": "string",
                    "testing_strategy": "string",
                    "monitoring_observability": "string",
                    "scalability_plan": "string",
                    "implementation_phases": "array"
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate optimized architecture: {e}")
            return {}
    
    # ================================
    # RESEARCH AND KNOWLEDGE INTEGRATION
    # ================================
    
    async def _research_best_practices(self, idea: str) -> Dict[str, Any]:
        """Research best practices and current solutions for the idea."""
        try:
            # This would integrate with web search, documentation APIs, etc.
            # For now, using LLM knowledge
            
            research_prompt = f"""
            User Idea: {idea}
            
            Research current best practices and solutions:
            
            1. **Similar Existing Solutions**:
               - Popular tools/frameworks that solve similar problems
               - Their strengths and limitations
               
            2. **Technology Trends**:
               - Current popular technology choices
               - Emerging technologies relevant to this domain
               
            3. **Common Pitfalls**:
               - Typical mistakes in this domain
               - How to avoid them
               
            4. **Success Patterns**:
               - Proven approaches that work well
               - Key success factors
               
            Focus on practical, accessible information for implementation.
            """
            
            response = await self.llm.generate_structured_response(
                prompt=research_prompt,
                response_format={
                    "existing_solutions": "array",
                    "technology_trends": "array",
                    "common_pitfalls": "array",
                    "success_patterns": "array",
                    "recommended_resources": "array"
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to research best practices: {e}")
            return {}
    
    # ================================
    # MEMORY INTEGRATION
    # ================================
    
    async def _create_grand_plan_from_enhancement(
        self, 
        enhancement: Dict[str, Any], 
        original_idea: str
    ) -> Optional[str]:
        """Create grand plan in memory from guided enhancement."""
        try:
            grand_plan = GrandPlan(
                title=f"Enhanced Plan: {original_idea[:50]}...",
                description=enhancement.get("solution_architecture", ""),
                vision=enhancement.get("problem_definition", ""),
                success_criteria=enhancement.get("success_metrics", []),
                tech_stack=enhancement.get("technology_stack", []),
                architecture_type="guided_enhancement",
                created_by_agent=self.config.agent_id
            )
            
            plan_id = await self.memory.create_grand_plan(grand_plan)
            
            # Create initial milestones and tasks
            if plan_id:
                await self._create_implementation_tasks(plan_id, enhancement)
            
            return plan_id
            
        except Exception as e:
            logger.error(f"Failed to create grand plan from enhancement: {e}")
            return None
    
    async def _create_grand_plan_from_optimization(
        self, 
        optimization: Dict[str, Any], 
        original_concept: str
    ) -> Optional[str]:
        """Create grand plan in memory from standardized optimization."""
        try:
            grand_plan = GrandPlan(
                title=f"Optimized Architecture: {original_concept[:50]}...",
                description=optimization.get("system_architecture", ""),
                vision=optimization.get("executive_summary", ""),
                success_criteria=["Production-ready system", "Scalable architecture", "Security compliance"],
                tech_stack=self._flatten_tech_stack(optimization.get("technology_stack", {})),
                architecture_type="standardized_optimization",
                created_by_agent=self.config.agent_id
            )
            
            plan_id = await self.memory.create_grand_plan(grand_plan)
            
            # Create detailed implementation phases
            if plan_id:
                await self._create_optimization_tasks(plan_id, optimization)
            
            return plan_id
            
        except Exception as e:
            logger.error(f"Failed to create grand plan from optimization: {e}")
            return None
    
    async def _create_implementation_tasks(self, plan_id: str, enhancement: Dict[str, Any]):
        """Create implementation tasks from enhancement plan."""
        try:
            phases = enhancement.get("implementation_phases", [])
            
            for i, phase in enumerate(phases):
                task = Task(
                    title=f"Phase {i+1}: {phase.get('name', f'Phase {i+1}')}",
                    description=phase.get("description", str(phase)),
                    parent_plan_id=plan_id,
                    priority=TaskPriority.HIGH if i == 0 else TaskPriority.MEDIUM,
                    domain_tags=["implementation", "guided"],
                    created_by_agent=self.config.agent_id
                )
                
                await self.memory.create_task(task)
                
        except Exception as e:
            logger.error(f"Failed to create implementation tasks: {e}")
    
    async def _create_optimization_tasks(self, plan_id: str, optimization: Dict[str, Any]):
        """Create detailed tasks from optimization plan."""
        try:
            phases = optimization.get("implementation_phases", [])
            
            for i, phase in enumerate(phases):
                task = Task(
                    title=f"Phase {i+1}: {phase.get('name', f'Phase {i+1}')}",
                    description=phase.get("description", str(phase)),
                    parent_plan_id=plan_id,
                    priority=TaskPriority.HIGH,
                    domain_tags=["optimization", "production"],
                    tech_stack_tags=self._flatten_tech_stack(optimization.get("technology_stack", {})),
                    created_by_agent=self.config.agent_id
                )
                
                await self.memory.create_task(task)
                
        except Exception as e:
            logger.error(f"Failed to create optimization tasks: {e}")
    
    def _flatten_tech_stack(self, tech_stack: Dict[str, List[str]]) -> List[str]:
        """Flatten nested tech stack dictionary to list."""
        flattened = []
        for category, technologies in tech_stack.items():
            if isinstance(technologies, list):
                flattened.extend(technologies)
            else:
                flattened.append(str(technologies))
        return flattened
    
    # ================================
    # MODE-SPECIFIC PROMPTS
    # ================================
    
    def _get_guided_enhancement_prompt(self) -> str:
        """Get system prompt for guided enhancement mode."""
        return """
        You are an expert Software Architect in Guided Enhancement mode, designed to help novice and intermediate developers.
        
        Your approach:
        - Ask clarifying questions to understand the user's vision
        - Break down complex concepts into simple, manageable parts
        - Provide structured choices rather than overwhelming options
        - Connect disparate ideas and show relationships
        - Offer hints and gentle guidance
        - Focus on learning opportunities
        - Prioritize accessible, modern technologies
        - Emphasize best practices in an approachable way
        
        Always maintain an encouraging, educational tone while being thorough and professional.
        """
    
    def _get_standardized_optimization_prompt(self) -> str:
        """Get system prompt for standardized optimization mode."""
        return """
        You are an expert Software Architect in Standardized Optimization mode, working with experienced developers.
        
        Your approach:
        - Conduct deep technical analysis
        - Focus on production-ready, scalable solutions
        - Research and recommend best-in-class alternatives
        - Identify potential risks and mitigation strategies
        - Optimize for performance, security, and maintainability
        - Provide detailed technical specifications
        - Consider enterprise-grade requirements
        - Leverage industry standards and proven patterns
        
        Maintain a professional, technical tone with comprehensive analysis and actionable recommendations.
        """ 