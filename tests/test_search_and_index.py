import pickle
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from app.services import retrieval


def _meta_min_df():
    return pd.DataFrame(
        {
            "vec_id": [0, 1, 2],
            "chunk_uid": ["c0", "c1", "c2"],
            "doc_id": ["d0", "d1", "d2"],
            "chunk_id": [0, 0, 0],
            "scopus_id": ["1", "2", "3"],
        }
    )


class TestLoadPklAndModel:
    def test_carga_desde_pkl_y_cachea(self):
        pkl_bytes = pickle.dumps({"meta_min": _meta_min_df(), "model": "some-model"})
        fake_model = MagicMock()
        with (
            patch("builtins.open", mock_open(read_data=pkl_bytes)),
            patch(
                "app.services.retrieval.SentenceTransformer", return_value=fake_model
            ) as st,
        ):
            model, meta_min = retrieval.load_pkl_and_model(emb_max_seq_len=128)
            model2, meta_min2 = retrieval.load_pkl_and_model()

        st.assert_called_once_with("some-model", device="cpu")
        assert model is fake_model
        assert model is model2
        assert meta_min is meta_min2
        assert fake_model.max_seq_length == 128

    def test_usa_modelo_por_defecto_si_pkl_no_lo_especifica(self):
        pkl_bytes = pickle.dumps({"meta_min": _meta_min_df()})
        with (
            patch("builtins.open", mock_open(read_data=pkl_bytes)),
            patch(
                "app.services.retrieval.SentenceTransformer", return_value=MagicMock()
            ) as st,
        ):
            retrieval.load_pkl_and_model()
        st.assert_called_once_with("intfloat/multilingual-e5-small", device="cpu")

    def test_limita_max_seq_length_a_512(self):
        pkl_bytes = pickle.dumps({"meta_min": _meta_min_df()})
        fake_model = MagicMock()
        with (
            patch("builtins.open", mock_open(read_data=pkl_bytes)),
            patch(
                "app.services.retrieval.SentenceTransformer", return_value=fake_model
            ),
        ):
            retrieval.load_pkl_and_model(emb_max_seq_len=9999)
        assert fake_model.max_seq_length == 512


class TestLoadFaiss:
    def test_carga_y_cachea_el_indice(self):
        fake_index = MagicMock(ntotal=42)
        with patch(
            "app.services.retrieval.faiss.read_index", return_value=fake_index
        ) as read_index:
            idx1 = retrieval.load_faiss()
            idx2 = retrieval.load_faiss()
        read_index.assert_called_once_with(retrieval.FAISS_PATH)
        assert idx1 is fake_index
        assert idx1 is idx2


class TestLoadScopusCsv:
    def test_carga_y_normaliza_scopus_id_a_string(self):
        df = pd.DataFrame({"scopus_id": [1, 2], "title": ["a", "b"]})
        with patch("app.services.retrieval.pd.read_csv", return_value=df):
            out = retrieval.load_scopus_csv()
        assert list(out["scopus_id"]) == ["1", "2"]
        assert all(isinstance(v, str) for v in out["scopus_id"])

    def test_sin_columna_scopus_id_lanza_value_error(self):
        df = pd.DataFrame({"title": ["a"]})
        with patch("app.services.retrieval.pd.read_csv", return_value=df):
            with pytest.raises(ValueError):
                retrieval.load_scopus_csv()

    def test_cachea_entre_llamadas(self):
        df = pd.DataFrame({"scopus_id": [1]})
        with patch("app.services.retrieval.pd.read_csv", return_value=df) as read_csv:
            retrieval.load_scopus_csv()
            retrieval.load_scopus_csv()
        read_csv.assert_called_once()


class TestE5EncodeQuery:
    def test_antepone_prefijo_query_y_normaliza(self):
        model = MagicMock()
        model.encode.return_value = np.array([[0.1, 0.2]])
        out = retrieval.e5_encode_query(model, "IA en salud")
        model.encode.assert_called_once_with(
            ["query: IA en salud"], normalize_embeddings=True, convert_to_numpy=True
        )
        assert out.dtype == np.float32


class TestSearchFullScopus:
    def _setup(self, meta_min=None, scopus_df=None):
        meta_min = meta_min if meta_min is not None else _meta_min_df()
        scopus_df = (
            scopus_df
            if scopus_df is not None
            else pd.DataFrame({"scopus_id": ["1", "2", "3"], "title": ["A", "B", "C"]})
        )
        model = MagicMock()
        model.encode.return_value = np.zeros((1, 4), dtype=np.float32)
        index = MagicMock()
        index.search.return_value = (
            np.array([[0.9, 0.5, 0.1]], dtype=np.float32),
            np.array([[0, 1, 2]]),
        )
        return model, meta_min, scopus_df, index

    def test_une_topk_con_columnas_del_csv(self):
        model, meta_min, scopus_df, index = self._setup()
        with (
            patch(
                "app.services.retrieval.load_pkl_and_model",
                return_value=(model, meta_min),
            ),
            patch("app.services.retrieval.load_faiss", return_value=index),
            patch("app.services.retrieval.load_scopus_csv", return_value=scopus_df),
        ):
            out = retrieval.search_full_scopus("consulta", topk=3)

        assert list(out["vec_id"]) == [0, 1, 2]
        assert list(out["score"]) == pytest.approx([0.9, 0.5, 0.1])
        assert "title" in out.columns

    def test_sin_scopus_id_en_meta_min_lanza_value_error(self):
        meta_min = pd.DataFrame({"vec_id": [0, 1, 2], "chunk_uid": ["c0", "c1", "c2"]})
        model, _, scopus_df, index = self._setup()
        with (
            patch(
                "app.services.retrieval.load_pkl_and_model",
                return_value=(model, meta_min),
            ),
            patch("app.services.retrieval.load_faiss", return_value=index),
            patch("app.services.retrieval.load_scopus_csv", return_value=scopus_df),
        ):
            with pytest.raises(ValueError):
                retrieval.search_full_scopus("consulta", topk=3)
