"""RetailRAG — a multi-source agentic RAG framework for the retail benchmark.

RetailRAG is the RAG system evaluated in the paper; RetailBridge is the
benchmark it is measured on. It plans a query
across sources, executes SQL and dense retrieval, and revises when a step comes
back wrong:

    from retailrag.pipeline import PlanRevisePipeline
    rag = PlanRevisePipeline()
    trace = rag.answer("...")

Scoring lives in retailrag.evaluation. The command-line entry point is
evaluate.py at the repository root.
"""
