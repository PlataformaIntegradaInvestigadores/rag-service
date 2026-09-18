from unittest.mock import patch

import pandas as pd
import pytest

from app.services import context


class TestShorten:
    def test_texto_corto_no_se_trunca(self):
        assert context._shorten("hola", 100) == "hola"

    def test_texto_largo_se_trunca_con_elipsis(self):
        out = context._shorten("a" * 20, 10)
        assert out == "a" * 7 + "..."
        assert len(out) == 10

    def test_colapsa_espacios(self):
        assert context._shorten("hola   \n  mundo", 100) == "hola mundo"


class TestSplitAuthors:
    def test_vacio_devuelve_lista_vacia(self):
        assert context._split_authors("") == []

    def test_separa_por_punto_y_coma(self):
        assert context._split_authors("Perez, J.; Gomez, A.") == [
            "Perez, J.",
            "Gomez, A.",
        ]

    def test_separa_por_and_literal(self):
        assert context._split_authors("Perez and Gomez") == ["Perez", "Gomez"]

    def test_fallback_a_coma_si_no_hay_punto_y_coma(self):
        assert context._split_authors("Perez, Gomez, Diaz") == [
            "Perez",
            "Gomez",
            "Diaz",
        ]


class TestLastName:
    def test_vacio_devuelve_vacio(self):
        assert context._last_name("") == ""

    def test_formato_apellido_coma_nombre(self):
        assert context._last_name("Perez, Juan") == "Perez"

    def test_formato_nombre_apellido_toma_ultimo_token(self):
        assert context._last_name("Juan Perez") == "Perez"


class TestFormatAuthorsForMention:
    def test_sin_autores_devuelve_none(self):
        assert context._format_authors_for_mention("") is None

    def test_un_solo_nombre_sin_separadores(self):
        assert context._format_authors_for_mention("Cristina") == "Cristina"

    def test_autores_presentes_pero_sin_apellido_util_devuelve_none(self):
        # _split_authors ya filtra tokens vacios, asi que este branch
        # defensivo de build_context solo se alcanza forzando un
        # _split_authors que devuelva un token en blanco.
        with patch("app.services.context._split_authors", return_value=["   "]):
            assert context._format_authors_for_mention("x") is None

    def test_dos_autores(self):
        out = context._format_authors_for_mention("Perez, Juan; Gomez, Ana")
        assert out == "Perez y Gomez"

    def test_tres_o_mas_autores_usa_et_al(self):
        out = context._format_authors_for_mention("Perez, J.; Gomez, A.; Diaz, L.")
        assert out == "Perez et al."

    def test_un_solo_autor_formato_apellido_coma_nombre(self):
        # _split_authors no distingue "Apellido, Nombre" (1 autor) de una
        # lista de apellidos separados por coma sin punto y coma (N autores);
        # sin punto y coma cae al fallback por coma y esto termina leyendo
        # "Perez, Juan" como 2 autores ("Perez" y "Juan"). Comportamiento
        # real observado, no necesariamente el deseado. Verificado contra
        # scopusdata.csv real: 0 filas de autor único contienen coma en toda
        # la columna 'authors' (autor único siempre viene como "Nombre
        # Apellido" sin coma), así que esta ambigüedad nunca se dispara con
        # los datos de producción actuales.
        assert context._format_authors_for_mention("Perez, Juan") == "Perez y Juan"


class TestExtractYear:
    def test_extrae_anio_de_columna_year(self):
        row = pd.Series({"year": "2021"})
        assert context._extract_year(row) == "2021"

    def test_extrae_anio_de_cover_date(self):
        row = pd.Series({"cover_date": "2019-05-01"})
        assert context._extract_year(row) == "2019"

    def test_sin_anio_devuelve_none(self):
        row = pd.Series({"title": "sin fecha"})
        assert context._extract_year(row) is None


class TestBuildContextBlocks:
    def _df(self):
        return pd.DataFrame(
            {
                "title": ["T1", "T2"],
                "abstract": ["A" * 2000, "resumen corto"],
                "authors": ["Perez, J.", "Gomez, A."],
                "doi": ["10.1/x", ""],
                "scopus_id": ["1", "2"],
                "year": ["2020", "2021"],
            }
        )

    def test_vacio_lanza_value_error(self):
        with pytest.raises(ValueError):
            context.build_context_blocks(pd.DataFrame())

    def test_arma_bloques_con_top_k(self):
        blocks = context.build_context_blocks(self._df(), top_k=1)
        assert len(blocks) == 1
        assert blocks[0]["cite_id"] == "1"
        # "Perez, J." sin ";" cae al fallback por coma de _split_authors,
        # ver nota en TestFormatAuthorsForMention.
        assert blocks[0]["authors_mention"] == "Perez y J."

    def test_recorta_abstract_largo(self):
        blocks = context.build_context_blocks(self._df(), top_k=1, max_chunk_chars=20)
        assert len(blocks[0]["text"]) == 20

    def test_top_k_mayor_que_filas_no_falla(self):
        blocks = context.build_context_blocks(self._df(), top_k=50)
        assert len(blocks) == 2


class TestDoiUrl:
    def test_vacio_devuelve_s_d(self):
        assert context._doi_url("") == "s/d"

    def test_doi_plano_se_convierte_a_url(self):
        assert context._doi_url("10.1/x") == "https://doi.org/10.1/x"

    def test_url_completa_no_se_duplica(self):
        url = "https://doi.org/10.1/x"
        assert context._doi_url(url) == url


class TestAuthorsCiteLine:
    def test_sin_autores(self):
        assert context._authors_cite_line("") == "Autor(es) no disponibles"

    def test_con_autores(self):
        assert context._authors_cite_line("Perez, J.; Gomez, A.") == (
            "Perez, J.; Gomez, A."
        )


class TestRenderFuentesFromBlocks:
    def test_formatea_cada_bloque_numerado(self):
        blocks = [
            {
                "authors_raw": "Perez, J.",
                "title": "T1",
                "doi_raw": "10.1/x",
            }
        ]
        out = context.render_fuentes_from_blocks(blocks)
        assert out.startswith("[1] Perez; J.;")
        assert "https://doi.org/10.1/x" in out


class TestComposePrompt:
    def _blocks(self):
        return [
            {
                "title": "Un titulo",
                "text": "contenido relevante",
                "authors_mention": "Perez",
                "authors_raw": "Perez, J.",
                "year": "2020",
                "doi_raw": "10.1/x",
            }
        ]

    def test_incluye_idioma_objetivo_y_pregunta(self):
        prompt = context.compose_prompt("mi pregunta", self._blocks(), target_iso="en")
        assert "'en'" in prompt
        assert "mi pregunta" in prompt

    def test_target_iso_invalido_cae_a_es(self):
        prompt = context.compose_prompt("q", self._blocks(), target_iso="xx-yy")
        assert "'es'" in prompt

    def test_trunca_a_max_chars(self):
        prompt = context.compose_prompt(
            "q", self._blocks(), max_chars=50, target_iso="es"
        )
        assert len(prompt) == 50
