class AgenticRAGError(Exception):
    """Base exception class for Agentic RAG system."""
    pass

class RetrievalError(AgenticRAGError):
    """Exception raised when document retrieval fails."""
    pass

class GradingError(AgenticRAGError):
    """Exception raised when document grading fails."""
    pass

class QueryRewriteError(AgenticRAGError):
    """Exception raised when query rewriting fails."""
    pass

class GenerationError(AgenticRAGError):
    """Exception raised when response generation fails."""
    pass

class HallucinationDetectionError(AgenticRAGError):
    """Exception raised when hallucination detection fails."""
    pass

class MaxLoopCountExceededError(AgenticRAGError):
    """Exception raised when the maximum loop count is exceeded."""
    pass