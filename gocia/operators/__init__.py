"""
gocia.operators

Genetic operators for GOCIA.

Submodules
----------
base            Abstract GeneticOperator base class, shared utilities, registry
graph_splice    Graph-based splice (two parents → two children)
graph_merge     Graph-based merge (two parents → one child)
mutation        Single-structure mutations: add, remove, displace

All operators are registered in OPERATOR_REGISTRY on import:

    from gocia.operators.base import OPERATOR_REGISTRY
    # trigger registration by importing the operator modules
    import gocia.operators.graph_splice
    import gocia.operators.graph_merge
    import gocia.operators.mutation
"""
