from unittest.mock import MagicMock, patch

import pytest

from app.services import language


class TestGetLidModel:
    def test_carga_una_sola_vez_y_cachea(self):
        fake_model = MagicMock()
        with patch(
            "app.services.language.fasttext.load_model", return_value=fake_model
        ) as load:
            m1 = language._get_lid_model()
            m2 = language._get_lid_model()
        load.assert_called_once_with(language.LID_MODEL_PATH)
        assert m1 is fake_model
        assert m2 is fake_model


class TestDetectLangAny:
    def _fake_model(self, label="__label__es", prob=0.97):
        model = MagicMock()
        model.predict.return_value = ([label], [prob])
        return model

    def test_texto_vacio_lanza_value_error(self):
        with pytest.raises(ValueError):
            language.detect_lang_any("   ")

    def test_detecta_idioma_valido(self):
        with patch(
            "app.services.language.fasttext.load_model",
            return_value=self._fake_model(),
        ):
            iso, conf = language.detect_lang_any("Hola mundo")
        assert iso == "es"
        assert conf == pytest.approx(0.97)

    def test_normaliza_etiqueta_con_region_a_iso_de_2_letras(self):
        model = self._fake_model(label="__label__en-us", prob=0.5)
        with patch("app.services.language.fasttext.load_model", return_value=model):
            iso, _ = language.detect_lang_any("hello world")
        assert iso == "en"

    def test_etiqueta_invalida_lanza_value_error(self):
        model = self._fake_model(label="__label__123", prob=0.9)
        with patch("app.services.language.fasttext.load_model", return_value=model):
            with pytest.raises(ValueError):
                language.detect_lang_any("texto")
