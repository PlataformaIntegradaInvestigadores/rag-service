from unittest.mock import MagicMock, patch

import pytest
import requests

from app.services import ollama_client


class TestGenerateWithOllamaHttp:
    def test_sin_target_iso_lanza_runtime_error(self):
        with pytest.raises(RuntimeError):
            ollama_client.generate_with_ollama_http("prompt", target_iso=None)

    def test_target_iso_invalido_lanza_runtime_error(self):
        with pytest.raises(RuntimeError):
            ollama_client.generate_with_ollama_http("prompt", target_iso="esp")

    def test_llamada_exitosa_devuelve_contenido(self):
        fake_response = MagicMock()
        fake_response.json.return_value = {"message": {"content": "respuesta generada"}}
        with patch(
            "app.services.ollama_client.requests.post", return_value=fake_response
        ) as post:
            out = ollama_client.generate_with_ollama_http(
                "prompt", target_iso="es", base_url="http://ollama:11434"
            )
        assert out == "respuesta generada"
        post.assert_called_once()
        args, kwargs = post.call_args
        assert args[0] == "http://ollama:11434/api/chat"
        assert kwargs["json"]["model"] == ollama_client.OLLAMA_MODEL

    def test_respuesta_vacia_lanza_runtime_error(self):
        fake_response = MagicMock()
        fake_response.json.return_value = {"message": {"content": "   "}}
        with patch(
            "app.services.ollama_client.requests.post", return_value=fake_response
        ):
            with pytest.raises(RuntimeError):
                ollama_client.generate_with_ollama_http("prompt", target_iso="es")

    def test_error_http_se_propaga(self):
        fake_response = MagicMock()
        fake_response.raise_for_status.side_effect = requests.HTTPError("500")
        with patch(
            "app.services.ollama_client.requests.post", return_value=fake_response
        ):
            with pytest.raises(requests.HTTPError):
                ollama_client.generate_with_ollama_http("prompt", target_iso="es")

    def test_normaliza_base_url_con_slash_final(self):
        fake_response = MagicMock()
        fake_response.json.return_value = {"message": {"content": "ok"}}
        with patch(
            "app.services.ollama_client.requests.post", return_value=fake_response
        ) as post:
            ollama_client.generate_with_ollama_http(
                "p", target_iso="es", base_url="http://ollama:11434/"
            )
        assert post.call_args[0][0] == "http://ollama:11434/api/chat"
