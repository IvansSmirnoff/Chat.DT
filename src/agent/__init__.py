"""Answering IFC-Bench questions over the graph: single-shot and agentic.

Runs in one process next to Neo4j (laptop, server or a Colab VM). The prompt
context is built from the live graph (``context.build_graph_context``), so
nothing has to be shipped between machines; a bundle is only that context
written to disk.
"""
