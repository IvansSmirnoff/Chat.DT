"""Colab-side client for the Chat.DT API.

This package is deliberately kept outside ``src.eval`` so that importing it
does not pull in ``src.eval.neo4j_exec`` (which requires the ``neo4j`` driver).
A Colab runtime can install only ``requirements-base.txt`` plus the LLM stack
and still be able to drive the whole experiment loop against a remote API.
"""
