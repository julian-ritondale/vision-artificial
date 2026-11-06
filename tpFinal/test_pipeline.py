"""Tests sintéticos del pipeline y del CLI (sin GPU, sin modelos). Ejecutar: python -m unittest test_pipeline -v"""
import unittest
import warnings
from unittest import mock

import chess
import cv2
import numpy as np
import requests

import chess_cli as c

INICIAL = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
NOMBRES = {"r": "rook", "n": "knight", "b": "bishop", "q": "queen", "k": "king", "p": "pawn"}


def _build():
    m = [[""] * 8 for _ in range(8)]
    for i, fila in enumerate(chess.Board(INICIAL + " w - - 0 1").__str__().split("\n")):
        for j, ch in enumerate(fila.split()):
            if ch != ".":
                m[i][j] = f"{'white' if ch.isupper() else 'black'}_{NOMBRES[ch.lower()]}"
    return m


class TestGeometria(unittest.TestCase):
    def test_ordenar_esquinas(self):
        pts = np.array([[90, 100], [10, 10], [10, 95], [100, 8]])
        o = c.ordenar_esquinas(pts)
        np.testing.assert_array_equal(o, [[10, 10], [100, 8], [90, 100], [10, 95]])

    def test_homografia_roundtrip(self):
        esq = np.array([[120, 80], [700, 120], [760, 640], [60, 600]], dtype=np.float32)
        H = c.calcular_homografia(esq)
        proy = cv2.perspectiveTransform(esq.reshape(-1, 1, 2), H).reshape(-1, 2)
        np.testing.assert_allclose(proy, [[0, 0], [800, 0], [800, 800], [0, 800]], atol=1e-2)
        Hi = np.linalg.inv(H)
        back = cv2.perspectiveTransform(proy.reshape(-1, 1, 2).astype(np.float32), Hi).reshape(-1, 2)
        np.testing.assert_allclose(back, esq, atol=1e-1)

    def test_mapear_celdas_y_conflicto(self):
        H = np.eye(3)
        dets = [
            {"bbox": [0, 0, 100, 50], "clase": "white_pawn", "conf": 0.9},     # bottom-center (50,50) -> (0,0)
            {"bbox": [710, 600, 790, 799], "clase": "black_king", "conf": 0.8},  # (750,799) -> (7,7)
            {"bbox": [0, 0, 100, 60], "clase": "white_rook", "conf": 0.5},     # conflicto, pierde
            {"bbox": [-50, 0, 50, 2000], "clase": "black_queen", "conf": 0.7}, # clip a fila 7, col 0
        ]
        m = c.mapear_piezas(dets, H)
        self.assertEqual(m[0][0], "white_pawn")
        self.assertEqual(m[7][7], "black_king")
        self.assertEqual(m[7][0], "black_queen")
        self.assertEqual(sum(1 for f in m for x in f if x), 3)


class TestOrientacion(unittest.TestCase):
    def test_cuatro_direcciones(self):
        base = np.array(_build(), dtype=object)  # blancas abajo = 'arriba'
        # imagen donde las blancas avanzan hacia: arriba->k0, izquierda->rotar a la inversa, etc.
        # Se construye la imagen rotando la posición correcta y se espera recuperar el FEN inicial.
        casos = {"arriba": 0, "izquierda": 1, "abajo": 2, "derecha": -1}
        # derecha: blancas en el borde izquierdo -> imagen = rot90(base, -1) (horario: abajo pasa a la izquierda)
        for direccion, k_img in casos.items():
            imagen = np.rot90(base, k=k_img).tolist()
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # la heurística no debe contradecir
                norm = c.normalizar_orientacion(imagen, direccion)
            self.assertEqual(c.matriz_a_fen(norm).split()[0], INICIAL, direccion)

    def test_posicion_asimetrica(self):
        base = _build()
        base[4][4] = "white_pawn"
        base[3][3] = "black_pawn"
        esperado = c.matriz_a_fen(base)
        for d, k in {"arriba": 0, "izquierda": 1, "abajo": 2, "derecha": -1}.items():
            self.assertEqual(c.matriz_a_fen(c.normalizar_orientacion(np.rot90(np.array(base, dtype=object), k).tolist(), d)), esperado)

    def test_warning_si_contradice(self):
        with self.assertWarns(UserWarning):
            c.normalizar_orientacion(_build(), "abajo")


class TestFen(unittest.TestCase):
    def test_inicial(self):
        self.assertEqual(c.matriz_a_fen(_build()), INICIAL + " w - - 0 1")
        self.assertTrue(c.matriz_a_fen(_build(), "b").endswith(" b - - 0 1"))

    def test_compresion(self):
        m = [[""] * 8 for _ in range(8)]
        m[0][0], m[0][7], m[7][3], m[7][4] = "black_rook", "black_king", "white_queen", "white_king"
        self.assertEqual(c.matriz_a_fen(m).split()[0], "r6k/8/8/8/8/8/8/3QK3")


class TestEnroque(unittest.TestCase):
    def test_conversion(self):
        b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        self.assertEqual(c.uci_a_movimiento(b, "e1h1").uci(), "e1g1")
        self.assertEqual(c.uci_a_movimiento(b, "e1a1").uci(), "e1c1")
        self.assertEqual(c.uci_a_movimiento(b, "e1g1").uci(), "e1g1")
        b.turn = chess.BLACK
        self.assertEqual(c.uci_a_movimiento(b, "e8a8").uci(), "e8c8")

    def test_ilegal(self):
        with self.assertRaises(c.ErrorCLI) as cm:
            c.uci_a_movimiento(chess.Board(), "e2e5")
        self.assertEqual(cm.exception.codigo, c.EXIT_JUGADA_ILEGAL)

    def test_evaluacion(self):
        self.assertEqual(c.formatear_evaluacion(19, None), "+0.19")
        self.assertIn("mate en 3", c.formatear_evaluacion(None, 3))


def resp(status, data=None):
    r = mock.Mock()
    r.status_code = status
    r.json.return_value = data
    return r


class TestLichess(unittest.TestCase):
    def codigo(self, **kw):
        with mock.patch("requests.get", **kw):
            with self.assertRaises(c.ErrorCLI) as cm:
                c.extraer_jugada(c.consultar_lichess("x"))
        return cm.exception

    def test_ok(self):
        with mock.patch("requests.get", return_value=resp(200, {"pvs": [{"moves": "e2e4 e7e5", "cp": 19}]})):
            self.assertEqual(c.extraer_jugada(c.consultar_lichess("x")), ("e2e4", 19, None))

    def test_errores(self):
        self.assertIn("404", self.codigo(return_value=resp(404)).mensaje)
        self.assertIn("429", self.codigo(return_value=resp(429)).mensaje)
        self.assertIn("timeout", self.codigo(side_effect=requests.exceptions.Timeout()).mensaje)
        self.assertIn("conexión", self.codigo(side_effect=requests.exceptions.ConnectionError()).mensaje)
        self.assertIn("pvs", self.codigo(return_value=resp(200, {"fen": "x"})).mensaje)


class TestValidacion(unittest.TestCase):
    def test_sin_rey(self):
        fen = "8/8/8/8/8/8/8/8 w - - 0 1"
        with self.assertRaises(c.ErrorCLI) as cm:
            c.validar_posicion(fen)
        self.assertIn("NO_WHITE_KING", cm.exception.mensaje)
        self.assertIn(fen, cm.exception.mensaje)
        c.validar_posicion(fen, forzar=True)  # no lanza

    def test_main_errores(self):
        self.assertEqual(c.main(["no_existe.jpg", "arriba"]), c.EXIT_ARCHIVO)

    def test_main_flujo_mockeado(self):
        """Flujo completo con modelos y Lichess simulados: posición inicial, jugada e2e4."""
        base = _build()
        dets = []
        for i in range(8):
            for j in range(8):
                if base[i][j]:
                    dets.append({"bbox": [j * 100 + 10, i * 100 + 10, j * 100 + 90, i * 100 + 95],
                                 "clase": base[i][j], "conf": 0.9})
        modelo = mock.Mock()
        with mock.patch.object(c, "cargar_modelos", return_value=(modelo, modelo)), \
             mock.patch.object(c, "esquinas_desde_yolo",
                               return_value=np.array([[0, 0], [800, 0], [800, 800], [0, 800]], np.float32)), \
             mock.patch.object(c, "detectar_piezas", return_value=dets), \
             mock.patch("cv2.imread", return_value=np.zeros((800, 800, 3), np.uint8)), \
             mock.patch("pathlib.Path.is_file", return_value=True), \
             mock.patch("requests.get", return_value=resp(200, {"pvs": [{"moves": "e2e4", "cp": 19}]})):
            self.assertEqual(c.main(["x.jpg", "arriba"]), 0)

    def test_sin_jugada(self):
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # ahogado
        self.assertTrue(board.is_game_over())


if __name__ == "__main__":
    unittest.main()
