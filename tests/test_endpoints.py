from unittest.mock import MagicMock, patch

import pandas as pd
from fastapi.testclient import TestClient

from app.api.v1.endpoints import ask as ask_module
from app.core.config import DEFAULT_TOPK
from app.main import app

client = TestClient(app)


class TestHealth:
    def test_todo_ok_devuelve_true(self):
        with (
            patch(
                "app.api.v1.endpoints.health.load_pkl_and_model",
                return_value=(MagicMock(), MagicMock()),
            ),
            patch("app.api.v1.endpoints.health.load_faiss", return_value=MagicMock()),
            patch(
                "app.api.v1.endpoints.health.load_scopus_csv", return_value=MagicMock()
            ),
            patch(
                "app.api.v1.endpoints.health._get_lid_model", return_value=MagicMock()
            ),
        ):
            response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["global_status"] == "Online"
        assert all(s["status"] == "ok" for s in body["groups"][0]["services"])

    def test_artefacto_faltante_devuelve_503(self):
        with patch(
            "app.api.v1.endpoints.health.load_pkl_and_model",
            side_effect=FileNotFoundError("no pkl"),
        ):
            response = client.get("/health")
        assert response.status_code == 503
        body = response.json()
        assert body["global_status"] == "Offline"
        pkl_service = next(
            s for s in body["groups"][0]["services"] if s["name"] == "modelo-embeddings"
        )
        assert pkl_service["status"] == "error"


class TestOpenAPI:
    def test_schema_uses_gateway_prefix(self):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["servers"] == [{"url": "/api/rag", "description": "Gateway"}]
        assert "/health" not in schema["paths"]


def _reranked_df():
    return pd.DataFrame(
        {
            "title": ["T1"],
            "abstract": ["contenido de prueba"],
            "authors": ["Perez, J."],
            "doi": ["10.1/x"],
            "scopus_id": ["1"],
            "year": ["2021"],
            "score": [0.8],
        }
    )


class TestAsk:
    def test_query_vacia_devuelve_400(self):
        response = client.post("/ask", json={"query": "   "})
        assert response.status_code == 400

    def test_fallo_detectando_idioma_devuelve_500(self):
        with patch(
            "app.api.v1.endpoints.ask.detect_lang_any",
            side_effect=ValueError("no valido"),
        ):
            response = client.post("/ask", json={"query": "hola"})
        assert response.status_code == 500
        assert "idioma" in response.json()["detail"].lower()

    def test_fallo_en_busqueda_devuelve_500(self):
        with (
            patch("app.api.v1.endpoints.ask.detect_lang_any", return_value=("es", 0.9)),
            patch(
                "app.api.v1.endpoints.ask.search_full_scopus",
                side_effect=RuntimeError("boom"),
            ),
        ):
            response = client.post("/ask", json={"query": "hola"})
        assert response.status_code == 500
        assert "búsqueda" in response.json()["detail"]

    def test_fallo_en_rerank_devuelve_500(self):
        with (
            patch("app.api.v1.endpoints.ask.detect_lang_any", return_value=("es", 0.9)),
            patch(
                "app.api.v1.endpoints.ask.search_full_scopus",
                return_value=_reranked_df(),
            ),
            patch(
                "app.api.v1.endpoints.ask.rerank_with_cross_encoder",
                side_effect=RuntimeError("boom"),
            ),
        ):
            response = client.post("/ask", json={"query": "hola"})
        assert response.status_code == 500
        assert "ranking" in response.json()["detail"]

    def test_fallo_construyendo_bloques_devuelve_500(self):
        with (
            patch("app.api.v1.endpoints.ask.detect_lang_any", return_value=("es", 0.9)),
            patch(
                "app.api.v1.endpoints.ask.search_full_scopus",
                return_value=_reranked_df(),
            ),
            patch(
                "app.api.v1.endpoints.ask.rerank_with_cross_encoder",
                return_value=_reranked_df(),
            ),
            patch(
                "app.api.v1.endpoints.ask.build_context_blocks",
                side_effect=ValueError("vacio"),
            ),
        ):
            response = client.post("/ask", json={"query": "hola"})
        assert response.status_code == 500
        assert "bloques" in response.json()["detail"]

    def test_fallo_generando_respuesta_llm_devuelve_500(self):
        blocks_raw = [
            {
                "cite_id": "1",
                "title": "T1",
                "text": "contenido",
                "authors_mention": "Perez",
                "authors_raw": "Perez, J.",
                "year": "2021",
                "doi_raw": "10.1/x",
            }
        ]
        with (
            patch("app.api.v1.endpoints.ask.detect_lang_any", return_value=("es", 0.9)),
            patch(
                "app.api.v1.endpoints.ask.search_full_scopus",
                return_value=_reranked_df(),
            ),
            patch(
                "app.api.v1.endpoints.ask.rerank_with_cross_encoder",
                return_value=_reranked_df(),
            ),
            patch(
                "app.api.v1.endpoints.ask.build_context_blocks", return_value=blocks_raw
            ),
            patch(
                "app.api.v1.endpoints.ask.generate_with_ollama_http",
                side_effect=RuntimeError("ollama caido"),
            ),
        ):
            response = client.post("/ask", json={"query": "hola"})
        assert response.status_code == 500
        assert "LLM" in response.json()["detail"]

    def test_flujo_completo_devuelve_200_con_respuesta(self):
        blocks_raw = [
            {
                "cite_id": "1",
                "title": "T1",
                "text": "contenido",
                "authors_mention": "Perez",
                "authors_raw": "Perez, J.",
                "year": "2021",
                "doi_raw": "10.1/x",
            }
        ]
        with (
            patch("app.api.v1.endpoints.ask.detect_lang_any", return_value=("es", 0.9)),
            patch(
                "app.api.v1.endpoints.ask.search_full_scopus",
                return_value=_reranked_df(),
            ),
            patch(
                "app.api.v1.endpoints.ask.rerank_with_cross_encoder",
                return_value=_reranked_df(),
            ),
            patch(
                "app.api.v1.endpoints.ask.build_context_blocks", return_value=blocks_raw
            ),
            patch(
                "app.api.v1.endpoints.ask.generate_with_ollama_http",
                return_value="Segun Perez [1], esto es una respuesta.",
            ),
        ):
            response = client.post("/ask", json={"query": "hola", "topk": 10})

        assert response.status_code == 200
        body = response.json()
        assert body["query"] == "hola"
        assert body["blocks"][0]["cite_id"] == "1"
        assert "[1] Perez; J." in body["used_refs_report"]
        assert set(body["timing_ms"].keys()) == {
            "search",
            "rerank",
            "generate",
            "total",
        }

    def test_fallo_auditando_citas_no_rompe_la_respuesta(self):
        blocks_raw = [
            {
                "cite_id": "1",
                "title": "T1",
                "text": "contenido",
                "authors_mention": "Perez",
                "authors_raw": "Perez, J.",
                "year": "2021",
                "doi_raw": "10.1/x",
            }
        ]
        with (
            patch("app.api.v1.endpoints.ask.detect_lang_any", return_value=("es", 0.9)),
            patch(
                "app.api.v1.endpoints.ask.search_full_scopus",
                return_value=_reranked_df(),
            ),
            patch(
                "app.api.v1.endpoints.ask.rerank_with_cross_encoder",
                return_value=_reranked_df(),
            ),
            patch(
                "app.api.v1.endpoints.ask.build_context_blocks", return_value=blocks_raw
            ),
            patch(
                "app.api.v1.endpoints.ask.generate_with_ollama_http",
                return_value="respuesta",
            ),
            patch(
                "app.api.v1.endpoints.ask.render_used_refs_report",
                side_effect=RuntimeError("auditoria rota"),
            ),
        ):
            response = client.post("/ask", json={"query": "hola"})

        assert response.status_code == 200
        assert "No se pudo auditar" in response.json()["used_refs_report"]

    def test_usa_topk_por_defecto_si_no_se_especifica(self):
        with (
            patch("app.api.v1.endpoints.ask.detect_lang_any", return_value=("es", 0.9)),
            patch(
                "app.api.v1.endpoints.ask.search_full_scopus",
                return_value=_reranked_df(),
            ) as search,
            patch(
                "app.api.v1.endpoints.ask.rerank_with_cross_encoder",
                return_value=_reranked_df(),
            ),
            patch(
                "app.api.v1.endpoints.ask.build_context_blocks",
                return_value=[
                    {
                        "cite_id": "1",
                        "title": "T1",
                        "text": "c",
                        "authors_mention": None,
                        "authors_raw": None,
                        "year": None,
                        "doi_raw": None,
                    }
                ],
            ),
            patch(
                "app.api.v1.endpoints.ask.generate_with_ollama_http", return_value="ok"
            ),
        ):
            client.post("/ask", json={"query": "hola"})
        search.assert_called_once_with("hola", topk=DEFAULT_TOPK)
        assert ask_module.DEFAULT_TOPK == DEFAULT_TOPK
