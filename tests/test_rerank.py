from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.core.utils import _safe_str
from app.services import rerank


class TestSafeStr:
    def test_string_pasa_igual(self):
        assert _safe_str("hola") == "hola"

    def test_none_es_cadena_vacia(self):
        assert _safe_str(None) == ""

    def test_nan_es_cadena_vacia(self):
        assert _safe_str(float("nan")) == ""

    def test_numero_se_convierte_a_string(self):
        assert _safe_str(42) == "42"

    def test_objeto_cuyo_str_falla_devuelve_cadena_vacia(self):
        class Roto:
            def __str__(self):
                raise RuntimeError("no puedo")

        assert _safe_str(Roto()) == ""


class TestMinmax:
    def test_normaliza_a_0_1(self):
        out = rerank._minmax(np.array([0.0, 5.0, 10.0]))
        assert out.tolist() == pytest.approx([0.0, 0.5, 1.0])

    def test_rango_constante_devuelve_ceros(self):
        out = rerank._minmax(np.array([3.0, 3.0, 3.0]))
        assert out.tolist() == [0.0, 0.0, 0.0]

    def test_todo_nan_devuelve_ceros(self):
        out = rerank._minmax(np.array([np.nan, np.nan]))
        assert out.tolist() == [0.0, 0.0]


class TestFirstNonempty:
    def test_toma_la_primera_columna_no_vacia(self):
        row = pd.Series({"title": "", "abstract": "  ", "summary": "Real content"})
        assert rerank._first_nonempty(row, ["title", "abstract", "summary"]) == (
            "Real content"
        )

    def test_sin_columnas_preferidas_busca_por_nombre(self):
        row = pd.Series({"chunk_text_extra": "hola mundo", "other": "x"})
        out = rerank._first_nonempty(row, ["title", "abstract"])
        assert "hola mundo" in out

    def test_sin_nada_util_devuelve_vacio(self):
        row = pd.Series({"id": 1})
        assert rerank._first_nonempty(row, ["title"]) == ""


class TestGetCrossEncoder:
    def test_carga_una_sola_vez_y_cachea(self):
        fake_ce = MagicMock()
        with patch("app.services.rerank.CrossEncoder", return_value=fake_ce) as ce_cls:
            ce1 = rerank.get_cross_encoder("modelo-x")
            ce2 = rerank.get_cross_encoder("modelo-x")
        ce_cls.assert_called_once_with("modelo-x", device="cpu")
        assert ce1 is fake_ce
        assert ce1 is ce2


class TestRerankWithCrossEncoder:
    def _df(self):
        return pd.DataFrame(
            {
                "title": ["Uno", "Dos", "Tres"],
                "abstract": ["texto uno", "texto dos", "texto tres"],
                "score": [0.2, 0.9, 0.5],
            }
        )

    def test_dataframe_vacio_lanza_value_error(self):
        with pytest.raises(ValueError):
            rerank.rerank_with_cross_encoder("q", pd.DataFrame())

    def test_fusiona_ce_y_score_denso(self):
        ce = MagicMock()
        ce.predict.return_value = [0.1, 0.9, 0.5]
        with patch("app.services.rerank.get_cross_encoder", return_value=ce):
            out = rerank.rerank_with_cross_encoder("q", self._df())

        assert list(out.columns).count("score_final") == 1
        # el mejor CE+denso debe quedar primero tras el sort descendente
        assert out.iloc[0]["title"] == "Dos"

    def test_sin_fusion_usa_solo_ce(self):
        ce = MagicMock()
        ce.predict.return_value = [0.1, 0.9, 0.5]
        with patch("app.services.rerank.get_cross_encoder", return_value=ce):
            out = rerank.rerank_with_cross_encoder(
                "q", self._df(), fuse_with_dense=False
            )
        assert "score_dense_norm" not in out.columns
        assert out.iloc[0]["title"] == "Dos"

    def test_respeta_batch_size(self):
        ce = MagicMock()
        ce.predict.side_effect = [[0.1], [0.9], [0.5]]
        with patch("app.services.rerank.get_cross_encoder", return_value=ce):
            rerank.rerank_with_cross_encoder("q", self._df(), batch_size=1)
        assert ce.predict.call_count == 3
