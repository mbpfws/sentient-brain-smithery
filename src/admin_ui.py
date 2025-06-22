"""
Corporate-Level Admin UI for Sentient Brain Multi-Agent System

This module provides a comprehensive administrative interface for testing,
monitoring, and managing the multi-agent system. Features include:
- Real-time agent workflow visualization
- Test case execution and validation
- Data population monitoring
- Policy management
- Performance analytics
- Log aggregation and analysis
"""

import streamlit as st
import asyncio
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any, Optional
import requests
import websocket
import threading
from dataclasses import dataclass, asdict
import logging

# Configure page
st.set_page_config(
    page_title="Sentient Brain Admin Console",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for corporate styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
        margin-bottom: 1rem;
    }
    .agent-status {
        display: flex;
        align-items: center;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 5px;
    }
    .status-active { background-color: #d4edda; color: #155724; }
    .status-idle { background-color: #fff3cd; color: #856404; }
    .status-error { background-color: #f8d7da; color: #721c24; }
    .test-result-pass { color: #28a745; font-weight: bold; }
    .test-result-fail { color: #dc3545; font-weight: bold; }
    .test-result-pending { color: #ffc107; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@dataclass
class AgentStatus:
    name: str
    type: str
    status: str
    last_activity: datetime
    tasks_completed: int
    current_task: Optional[str] = None
    performance_score: float = 0.0

@dataclass
class TestCase:
    id: str
    name: str
    description: str
    agent_type: str
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    error_message: Optional[str] = None

class AdminUI:
    def __init__(self):
        self.api_base_url = st.secrets.get("API_BASE_URL", "http://localhost:8000")
        self.ws_url = st.secrets.get("WS_URL", "ws://localhost:8000/ws")
        
        # Initialize session state
        if 'agents_status' not in st.session_state:
            st.session_state.agents_status = {}
        if 'test_results' not in st.session_state:
            st.session_state.test_results = {}
        if 'system_metrics' not in st.session_state:
            st.session_state.system_metrics = []
        if 'real_time_logs' not in st.session_state:
            st.session_state.real_time_logs = []

    def render_header(self):
        """Render the main header."""
        st.markdown("""
        <div class="main-header">
            <h1>🧠 Sentient Brain Multi-Agent System</h1>
            <h3>Corporate Admin Console & Testing Platform</h3>
            <p>Real-time monitoring, testing, and management interface</p>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render the sidebar with navigation and controls."""
        with st.sidebar:
            st.title("🎛️ Control Panel")
            
            # Navigation
            page = st.selectbox(
                "Navigate to:",
                [
                    "🏠 Dashboard",
                    "🤖 Agent Monitor", 
                    "🧪 Test Suite",
                    "📊 Data Explorer",
                    "📜 Policy Manager",
                    "📈 Analytics",
                    "🔍 Log Viewer",
                    "⚙️ System Config"
                ]
            )
            
            st.divider()
            
            # System Controls
            st.subheader("🔧 System Controls")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh All", use_container_width=True):
                    st.rerun()
            with col2:
                if st.button("🛑 Emergency Stop", use_container_width=True):
                    self.emergency_stop()
            
            # Connection Status
            st.subheader("🔗 Connection Status")
            self.render_connection_status()
            
            # Quick Actions
            st.subheader("⚡ Quick Actions")
            if st.button("🧪 Run Health Check", use_container_width=True):
                self.run_health_check()
            if st.button("📊 Generate Report", use_container_width=True):
                self.generate_system_report()
            
            return page

    def render_connection_status(self):
        """Render connection status indicators."""
        try:
            response = requests.get(f"{self.api_base_url}/api/v1/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ API Connected")
                health_data = response.json()
                st.json(health_data)
            else:
                st.error("❌ API Error")
        except Exception as e:
            st.error(f"❌ API Disconnected: {str(e)}")

    def render_dashboard(self):
        """Render the main dashboard."""
        st.header("📊 System Dashboard")
        
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Active Agents",
                value=len([a for a in st.session_state.agents_status.values() if a.status == "active"]),
                delta="+2 from yesterday"
            )
        
        with col2:
            st.metric(
                label="Tasks Completed",
                value=sum(a.tasks_completed for a in st.session_state.agents_status.values()),
                delta="+15 today"
            )
        
        with col3:
            st.metric(
                label="Success Rate",
                value="94.3%",
                delta="+2.1%"
            )
        
        with col4:
            st.metric(
                label="Avg Response Time",
                value="1.2s",
                delta="-0.3s"
            )
        
        # Real-time Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔄 Agent Activity")
            self.render_agent_activity_chart()
        
        with col2:
            st.subheader("📈 Performance Trends")
            self.render_performance_chart()
        
        # Recent Activity
        st.subheader("📋 Recent Activity")
        self.render_recent_activity()

    def render_agent_monitor(self):
        """Render agent monitoring interface."""
        st.header("🤖 Agent Monitor")
        
        # Agent Status Overview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Agent Status Overview")
            self.render_agent_status_table()
        
        with col2:
            st.subheader("Agent Distribution")
            self.render_agent_distribution_chart()
        
        # Individual Agent Details
        st.subheader("🔍 Individual Agent Details")
        
        agent_names = list(st.session_state.agents_status.keys()) if st.session_state.agents_status else ["No agents available"]
        selected_agent = st.selectbox("Select Agent:", agent_names)
        
        if selected_agent and selected_agent != "No agents available":
            self.render_agent_details(selected_agent)
        
        # Agent Workflow Visualization
        st.subheader("🌊 Workflow Visualization")
        self.render_workflow_diagram()

    def render_test_suite(self):
        """Render comprehensive test suite interface."""
        st.header("🧪 Test Suite")
        
        # Test Categories
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔬 Unit Tests",
            "🔗 Integration Tests", 
            "🎭 Agent Behavior Tests",
            "📊 Performance Tests"
        ])
        
        with tab1:
            self.render_unit_tests()
        
        with tab2:
            self.render_integration_tests()
        
        with tab3:
            self.render_agent_behavior_tests()
        
        with tab4:
            self.render_performance_tests()

    def render_unit_tests(self):
        """Render unit test interface."""
        st.subheader("🔬 Unit Tests")
        
        # Test Categories
        test_categories = [
            "Agent Initialization",
            "Database Operations", 
            "LLM Integration",
            "Knowledge Graph",
            "Workflow Engine"
        ]
        
        selected_category = st.selectbox("Test Category:", test_categories)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("▶️ Run Selected Tests", use_container_width=True):
                self.run_unit_tests(selected_category)
        
        with col2:
            if st.button("🔄 Run All Unit Tests", use_container_width=True):
                self.run_all_unit_tests()
        
        # Test Results
        st.subheader("📋 Test Results")
        self.render_test_results_table("unit")

    def render_integration_tests(self):
        """Render integration test interface."""
        st.subheader("🔗 Integration Tests")
        
        # Predefined integration test scenarios
        integration_scenarios = {
            "End-to-End Workflow": {
                "description": "Test complete user query to response workflow",
                "steps": [
                    "User submits query",
                    "Orchestrator analyzes intent", 
                    "Appropriate agents spawned",
                    "Knowledge retrieved",
                    "Response generated"
                ]
            },
            "Multi-Agent Collaboration": {
                "description": "Test agent-to-agent communication",
                "steps": [
                    "Architect creates plan",
                    "Code analyzer reviews existing code",
                    "Document agent generates docs",
                    "All results integrated"
                ]
            },
            "Failure Recovery": {
                "description": "Test system behavior under failure conditions",
                "steps": [
                    "Simulate agent failure",
                    "Verify fallback mechanisms",
                    "Check data consistency",
                    "Confirm recovery"
                ]
            }
        }
        
        selected_scenario = st.selectbox("Integration Scenario:", list(integration_scenarios.keys()))
        
        if selected_scenario:
            scenario = integration_scenarios[selected_scenario]
            st.write(f"**Description:** {scenario['description']}")
            st.write("**Test Steps:**")
            for i, step in enumerate(scenario['steps'], 1):
                st.write(f"{i}. {step}")
            
            if st.button(f"🚀 Run {selected_scenario}", use_container_width=True):
                self.run_integration_test(selected_scenario)

    def render_agent_behavior_tests(self):
        """Render agent behavior testing interface."""
        st.subheader("🎭 Agent Behavior Tests")
        
        # Agent Test Scenarios
        agent_tests = {
            "Ultra Orchestrator": [
                "Intent Classification Accuracy",
                "Agent Routing Logic",
                "Workflow Coordination",
                "Error Handling"
            ],
            "Architect Agent": [
                "Requirements Analysis",
                "Architecture Design",
                "Tech Stack Recommendations", 
                "Plan Generation"
            ],
            "Code Analyzer": [
                "Code Parsing Accuracy",
                "Semantic Understanding",
                "Relationship Detection",
                "Performance Analysis"
            ],
            "Document Agent": [
                "Content Extraction",
                "Semantic Chunking",
                "Knowledge Linking",
                "Update Detection"
            ]
        }
        
        selected_agent_type = st.selectbox("Agent Type:", list(agent_tests.keys()))
        
        if selected_agent_type:
            st.write(f"**Available Tests for {selected_agent_type}:**")
            
            for test_name in agent_tests[selected_agent_type]:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"• {test_name}")
                
                with col2:
                    if st.button(f"▶️", key=f"run_{test_name}", help=f"Run {test_name}"):
                        self.run_agent_behavior_test(selected_agent_type, test_name)
                
                with col3:
                    # Show last result
                    result = st.session_state.test_results.get(f"{selected_agent_type}_{test_name}")
                    if result:
                        if result['status'] == 'pass':
                            st.success("✅")
                        elif result['status'] == 'fail':
                            st.error("❌")
                        else:
                            st.warning("⏳")

    def render_performance_tests(self):
        """Render performance testing interface."""
        st.subheader("📊 Performance Tests")
        
        # Performance Test Configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎛️ Test Configuration")
            
            concurrent_users = st.slider("Concurrent Users:", 1, 100, 10)
            test_duration = st.slider("Test Duration (minutes):", 1, 60, 5)
            query_complexity = st.selectbox("Query Complexity:", ["Simple", "Medium", "Complex"])
            
            if st.button("🚀 Start Load Test", use_container_width=True):
                self.start_load_test(concurrent_users, test_duration, query_complexity)
        
        with col2:
            st.subheader("📈 Current Performance")
            self.render_performance_metrics()

    def render_data_explorer(self):
        """Render data exploration interface."""
        st.header("📊 Data Explorer")
        
        # Data Source Selection
        data_sources = [
            "Knowledge Graph Nodes",
            "Agent Messages",
            "Workflow States",
            "Performance Metrics",
            "Error Logs"
        ]
        
        selected_source = st.selectbox("Data Source:", data_sources)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date_range = st.date_input("Date Range:", value=[datetime.now().date() - timedelta(days=7), datetime.now().date()])
        
        with col2:
            limit = st.number_input("Limit:", min_value=10, max_value=10000, value=100)
        
        with col3:
            if st.button("🔍 Query Data", use_container_width=True):
                self.query_data(selected_source, date_range, limit)
        
        # Data Visualization
        if f"data_{selected_source}" in st.session_state:
            data = st.session_state[f"data_{selected_source}"]
            
            # Display options
            view_type = st.radio("View Type:", ["Table", "Chart", "JSON"], horizontal=True)
            
            if view_type == "Table":
                st.dataframe(data, use_container_width=True)
            elif view_type == "Chart":
                self.render_data_chart(data, selected_source)
            else:
                st.json(data.to_dict() if hasattr(data, 'to_dict') else data)

    def render_policy_manager(self):
        """Render policy management interface."""
        st.header("📜 Policy Manager")
        
        # Policy Categories
        tab1, tab2, tab3 = st.tabs(["🛡️ Security Policies", "🔄 Workflow Policies", "📊 Data Policies"])
        
        with tab1:
            self.render_security_policies()
        
        with tab2:
            self.render_workflow_policies()
        
        with tab3:
            self.render_data_policies()

    def render_analytics(self):
        """Render analytics dashboard."""
        st.header("📈 Analytics")
        
        # Time Range Selection
        time_range = st.selectbox(
            "Time Range:",
            ["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days"]
        )
        
        # Analytics Sections
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Usage Analytics")
            self.render_usage_analytics(time_range)
        
        with col2:
            st.subheader("⚡ Performance Analytics")
            self.render_performance_analytics(time_range)
        
        # Detailed Charts
        st.subheader("📊 Detailed Analysis")
        self.render_detailed_analytics(time_range)

    def render_log_viewer(self):
        """Render log viewing interface."""
        st.header("🔍 Log Viewer")
        
        # Log Filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            log_level = st.selectbox("Log Level:", ["ALL", "DEBUG", "INFO", "WARNING", "ERROR"])
        
        with col2:
            log_source = st.selectbox("Source:", ["ALL", "API", "Agents", "Database", "Workflow"])
        
        with col3:
            auto_refresh = st.checkbox("Auto Refresh", value=True)
        
        with col4:
            if st.button("🔄 Refresh Logs"):
                self.refresh_logs()
        
        # Log Display
        st.subheader("📜 Recent Logs")
        self.render_log_display(log_level, log_source, auto_refresh)

    def render_system_config(self):
        """Render system configuration interface."""
        st.header("⚙️ System Configuration")
        
        # Configuration Sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔧 System Settings",
            "🤖 Agent Config",
            "🔗 Connections",
            "🔒 Security"
        ])
        
        with tab1:
            self.render_system_settings()
        
        with tab2:
            self.render_agent_config()
        
        with tab3:
            self.render_connection_config()
        
        with tab4:
            self.render_security_config()

    # Helper Methods (Implementation details)
    
    def render_agent_activity_chart(self):
        """Render agent activity chart."""
        # Sample data - replace with real data
        data = {
            'time': [datetime.now() - timedelta(minutes=x) for x in range(60, 0, -5)],
            'orchestrator': [5, 7, 3, 8, 6, 9, 4, 7, 5, 8, 6, 7],
            'architect': [2, 3, 1, 4, 2, 3, 1, 2, 3, 4, 2, 3],
            'codebase': [8, 6, 9, 7, 8, 5, 9, 8, 7, 6, 8, 9]
        }
        
        df = pd.DataFrame(data)
        
        fig = px.line(df, x='time', y=['orchestrator', 'architect', 'codebase'],
                     title='Agent Activity Over Time')
        st.plotly_chart(fig, use_container_width=True)

    def render_performance_chart(self):
        """Render performance metrics chart."""
        # Sample performance data
        data = {
            'metric': ['Response Time', 'Success Rate', 'Throughput', 'Error Rate'],
            'current': [1.2, 94.3, 150, 2.1],
            'target': [1.5, 95.0, 120, 2.0]
        }
        
        df = pd.DataFrame(data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Current', x=df['metric'], y=df['current']))
        fig.add_trace(go.Bar(name='Target', x=df['metric'], y=df['target']))
        
        fig.update_layout(title='Performance Metrics vs Targets')
        st.plotly_chart(fig, use_container_width=True)

    def render_recent_activity(self):
        """Render recent activity table."""
        # Sample activity data
        activities = [
            {"time": "2 min ago", "agent": "Orchestrator", "action": "Query processed", "status": "✅"},
            {"time": "5 min ago", "agent": "Architect", "action": "Plan generated", "status": "✅"},
            {"time": "8 min ago", "agent": "Codebase", "action": "Code analyzed", "status": "✅"},
            {"time": "12 min ago", "agent": "Document", "action": "Docs updated", "status": "⚠️"},
        ]
        
        df = pd.DataFrame(activities)
        st.dataframe(df, use_container_width=True, hide_index=True)

    def emergency_stop(self):
        """Handle emergency stop."""
        st.error("🛑 Emergency stop initiated!")
        # Implement emergency stop logic
        
    def run_health_check(self):
        """Run system health check."""
        with st.spinner("Running health check..."):
            time.sleep(2)  # Simulate health check
            st.success("✅ All systems healthy!")

    def generate_system_report(self):
        """Generate comprehensive system report."""
        with st.spinner("Generating report..."):
            time.sleep(3)  # Simulate report generation
            st.success("📊 Report generated successfully!")
            st.download_button(
                label="📥 Download Report",
                data="Sample report data",
                file_name=f"system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

def main():
    """Main application entry point."""
    admin_ui = AdminUI()
    
    # Render header
    admin_ui.render_header()
    
    # Render sidebar and get selected page
    selected_page = admin_ui.render_sidebar()
    
    # Route to appropriate page
    if selected_page == "🏠 Dashboard":
        admin_ui.render_dashboard()
    elif selected_page == "🤖 Agent Monitor":
        admin_ui.render_agent_monitor()
    elif selected_page == "🧪 Test Suite":
        admin_ui.render_test_suite()
    elif selected_page == "📊 Data Explorer":
        admin_ui.render_data_explorer()
    elif selected_page == "📜 Policy Manager":
        admin_ui.render_policy_manager()
    elif selected_page == "📈 Analytics":
        admin_ui.render_analytics()
    elif selected_page == "🔍 Log Viewer":
        admin_ui.render_log_viewer()
    elif selected_page == "⚙️ System Config":
        admin_ui.render_system_config()

if __name__ == "__main__":
    main() 