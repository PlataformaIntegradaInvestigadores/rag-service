from app.services import refs


class TestExtractUsedRefs:
    def test_extrae_citas_en_orden_de_aparicion(self):
        text = "Segun [2] y tambien [1], ademas [2] otra vez."
        assert refs.extract_used_refs(text, n_max=5) == [2, 1]

    def test_ignora_indices_fuera_de_rango(self):
        text = "Ver [1] y [99]."
        assert refs.extract_used_refs(text, n_max=1) == [1]

    def test_sin_citas_devuelve_lista_vacia(self):
        assert refs.extract_used_refs("sin citas aqui", n_max=3) == []


class TestRenderUsedRefsReport:
    def _blocks(self):
        return [
            {"authors_raw": "Perez, J.", "title": "T1", "doi_raw": "10.1/x"},
            {"authors_raw": "Gomez, A.", "title": "T2", "doi_raw": ""},
        ]

    def test_sin_citas_devuelve_mensaje_fijo(self):
        out = refs.render_used_refs_report("sin nada", self._blocks())
        assert out == "No se detectaron citas [n] en el texto."

    def test_con_citas_lista_las_fuentes_usadas(self):
        out = refs.render_used_refs_report("uso [2] y [1]", self._blocks())
        assert "[2] Gomez; A." in out
        assert "[1] Perez; J." in out
        assert out.index("[2]") < out.index("[1]")
