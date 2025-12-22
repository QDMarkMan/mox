# Project Architecture

This document describes the high-level architecture of the MolX project.

## Architecture Diagram

```mermaid
graph TD
    subgraph Frontend ["molx_client (React/Vite)"]
        UI["React UI Components"]
        Chat["Chat Interface"]
        Vis["SAR Visualizations (Plotly/D3)"]
    end

    subgraph Backend ["molx_server (FastAPI)"]
        API["REST API Routes"]
        Stream["SSE Streaming Handler"]
        Session["Session Management"]
    end

    subgraph AgenticCore ["molx_agent (LangGraph)"]
        Orchestrator["MolxAgent Orchestrator"]
        Graph["LangGraph Pipeline"]
        
        subgraph Agents ["Specialized Agents"]
            IC["Intent Classifier"]
            Planner["Task Planner"]
            DC["Data Cleaner"]
            SAR["SAR Analysis Agent"]
            Reporter["Report Generator"]
        end
        
        subgraph Tools ["Domain Tools"]
            RDKit["RDKit Integration"]
            SARVis["SAR Visualizer"]
            HTML["HTML Report Builder"]
        end
    end

    subgraph Core ["molx_core"]
        Memory["Session Recorder / Persistence"]
        Config["Shared Configuration"]
    end

    UI <--> API
    API --> Orchestrator
    Orchestrator --> Graph
    Graph --> IC
    Graph --> Planner
    Graph --> DC
    Graph --> SAR
    Graph --> Reporter
    
    DC --> RDKit
    SAR --> RDKit
    Reporter --> SARVis
    Reporter --> HTML
    
    Orchestrator --> Memory
    API --> Session
```

## Component Descriptions

### 1. Frontend (`molx_client`)
- **React/Vite**: Modern frontend stack for a responsive user interface.
- **Chat Interface**: Interactive chat for users to query the SAR agent.
- **SAR Visualizations**: Interactive plots and molecular structures rendered in the browser.

### 2. Backend (`molx_server`)
- **FastAPI**: High-performance Python web framework.
- **SSE Streaming**: Supports real-time token-by-token streaming from the agent to the client.
- **Session Management**: Handles user sessions and conversation history.

### 3. Agentic Core (`molx_agent`)
- **LangGraph**: Orchestrates the multi-agent workflow as a stateful graph.
- **Orchestrator**: The entry point that manages the graph execution.
- **Specialized Agents**: Each agent handles a specific part of the SAR pipeline (e.g., cleaning data, planning tasks, generating reports).
- **Domain Tools**: Specialized tools for cheminformatics (RDKit) and visualization.

### 4. Core (`molx_core`)
- **Memory**: Provides persistence for conversation history and agent states.
- **Config**: Centralized configuration management for the entire project.
