import pytest

from app.services import language, rerank, retrieval


@pytest.fixture(autouse=True)
def _reset_module_caches():
    """The service modules cache heavy singletons in module globals; reset them
    around every test so nothing leaks between tests and no test
    accidentally loads a real multi-hundred-MB model from disk."""
    retrieval._model_cache = None
    retrieval._meta_min_cache = None
    retrieval._index_cache = None
    retrieval._scopus_cache = None
    language._lid_model_cache = None
    rerank._ce_cache = None
    yield
    retrieval._model_cache = None
    retrieval._meta_min_cache = None
    retrieval._index_cache = None
    retrieval._scopus_cache = None
    language._lid_model_cache = None
    rerank._ce_cache = None
