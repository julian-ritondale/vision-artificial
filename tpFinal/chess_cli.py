#!/usr/bin/env python3
"""Digitaliza un tablero de ajedrez a partir de una foto y consulta a Lichess la mejor jugada.

Pipeline: YOLO (esquinas) -> homografía -> YOLO (piezas) -> matriz 8x8 ->
orientación (dirección indicada por el usuario) -> FEN -> Lichess cloud-eval.

Uso:
    python chess_cli.py IMAGEN DIRECCION [--turno w|b] [--debug] [--forzar]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PESOS_ESQUINAS = BASE_DIR / "models" / "corners.pt"
PESOS_PIEZAS = BASE_DIR / "models" / "pieces.pt"

LADO = 800          # lado del tablero rectificado (px)
CELDA = LADO // 8   # lado de cada celda (px)
LICHESS_URL = "https://lichess.org/api/cloud-eval"
DIRECCIONES = ("arriba", "abajo", "izquierda", "derecha")

# Cantidad de rotaciones antihorarias (np.rot90) para dejar las blancas abajo.
# Convención: la dirección es hacia donde avanzan las blancas EN LA IMAGEN.
K_POR_DIRECCION = {"arriba": 0, "derecha": 1, "abajo": 2, "izquierda": 3}

SIMBOLO = {
    "white_pawn": "P", "white_knight": "N", "white_bishop": "B",
    "white_rook": "R", "white_queen": "Q", "white_king": "K",
    "black_pawn": "p", "black_knight": "n", "black_bishop": "b",
    "black_rook": "r", "black_queen": "q", "black_king": "k",
}

# Códigos de salida
EXIT_OK, EXIT_ARCHIVO, EXIT_ESQUINAS, EXIT_POSICION, EXIT_SIN_JUGADA, EXIT_LICHESS, EXIT_JUGADA_ILEGAL = 0, 2, 3, 4, 5, 6, 7


class ErrorCLI(Exception):
    """Error esperado: se informa en stderr sin traceback."""

    def __init__(self, mensaje: str, codigo: int = 1) -> None:
        super().__init__(mensaje)
        self.mensaje = mensaje
        self.codigo = codigo


# --------------------------------------------------------------------------
# Pipeline geométrico
# --------------------------------------------------------------------------
def ordenar_esquinas(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos como TL, TR, BR, BL.

    TL = menor x+y, BR = mayor x+y, TR = menor y-x, BL = mayor y-x.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    suma = pts.sum(axis=1)
    dif = pts[:, 1] - pts[:, 0]
    return np.array(
        [pts[np.argmin(suma)], pts[np.argmin(dif)], pts[np.argmax(suma)], pts[np.argmax(dif)]],
        dtype=np.float32,
    )


def esquinas_desde_yolo(results: Any) -> np.ndarray:
    """Extrae los centros de bbox de las 4 detecciones de mayor confianza (ordenadas).

    Lanza ErrorCLI si hay menos de 4 detecciones.
    """
    boxes = results[0].boxes
    n = 0 if boxes is None else len(boxes)
    if n < 4:
        raise ErrorCLI(
            f"Se detectaron {n} esquinas y se necesitan 4. Probá con una foto más nítida, "
            "con mejor iluminación y con el tablero completo y visible.",
            EXIT_ESQUINAS,
        )
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    top = np.argsort(-conf)[:4]
    centros = np.stack([(xyxy[top, 0] + xyxy[top, 2]) / 2, (xyxy[top, 1] + xyxy[top, 3]) / 2], axis=1)
    return ordenar_esquinas(centros)


def calcular_homografia(esquinas: np.ndarray, lado: int = LADO) -> np.ndarray:
    """Homografía H de la imagen (esquinas TL,TR,BR,BL) al cuadrado lado x lado.

    s·[x', y', 1]^T = H·[x, y, 1]^T
    """
    import cv2

    dst = np.array([[0, 0], [lado, 0], [lado, lado], [0, lado]], dtype=np.float32)
    H, _ = cv2.findHomography(np.asarray(esquinas, dtype=np.float32), dst)
    if H is None:
        raise ErrorCLI("No se pudo calcular la homografía (esquinas degeneradas).", EXIT_ESQUINAS)
    return H


def mapear_piezas(detecciones: list[dict], H: np.ndarray, lado: int = LADO) -> list[list[str]]:
    """Asigna cada detección a una celda; devuelve matriz 8x8 de strings ('' = vacía).

    Cada detección es un dict con 'bbox' (x1,y1,x2,y2), 'clase' (str) y 'conf' (float).
    Se usa el punto bottom-center del bbox. Ante conflicto gana la mayor confianza.
    """
    import cv2

    celda = lado / 8
    matriz = [[""] * 8 for _ in range(8)]
    mejor = np.zeros((8, 8))
    if not detecciones:
        return matriz
    pts = np.array([[[(d["bbox"][0] + d["bbox"][2]) / 2, d["bbox"][3]]] for d in detecciones], dtype=np.float32)
    mapeados = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    for d, (x, y) in zip(detecciones, mapeados):
        col = int(np.clip(x // celda, 0, 7))
        fila = int(np.clip(y // celda, 0, 7))
        if d["conf"] > mejor[fila, col]:
            mejor[fila, col] = d["conf"]
            matriz[fila][col] = d["clase"]
    return matriz


def normalizar_orientacion(matriz: list[list[str]], direccion: str) -> list[list[str]]:
    """Rota la matriz para que las blancas queden abajo (fila 0 = rank 8, col 0 = columna a).

    `direccion` es hacia dónde avanzan las blancas en la imagen. La heurística de densidad
    de piezas blancas solo verifica y emite un warning si contradice al usuario.
    """
    if direccion not in K_POR_DIRECCION:
        raise ErrorCLI(f"Dirección inválida: {direccion!r}. Opciones: {', '.join(DIRECCIONES)}.")
    arr = np.array(matriz, dtype=object)
    sugerida = _heuristica_direccion(arr)
    if sugerida is not None and sugerida != direccion:
        warnings.warn(
            f"La heurística de piezas sugiere que las blancas avanzan hacia '{sugerida}', "
            f"pero indicaste '{direccion}'. Se respeta lo indicado; revisá la dirección.",
            stacklevel=2,
        )
    return np.rot90(arr, k=K_POR_DIRECCION[direccion]).tolist()


def _heuristica_direccion(arr: np.ndarray) -> str | None:
    """Estima hacia dónde avanzan las blancas según dónde se concentran (2 filas/cols por borde)."""
    blanca = np.vectorize(lambda s: str(s).startswith("white_"))(arr)
    borde = {
        "arriba": blanca[6:, :].sum(),      # blancas abajo -> avanzan hacia arriba
        "abajo": blanca[:2, :].sum(),
        "derecha": blanca[:, :2].sum(),     # blancas a la izquierda -> avanzan a la derecha
        "izquierda": blanca[:, 6:].sum(),
    }
    ordenado = sorted(borde.items(), key=lambda kv: -kv[1])
    if ordenado[0][1] == 0 or ordenado[0][1] == ordenado[1][1]:
        return None
    return ordenado[0][0]


def matriz_a_fen(matriz: list[list[str]], turno: str = "w") -> str:
    """Convierte la matriz 8x8 (fila 0 = rank 8) a FEN con sufijo '{turno} - - 0 1'."""
    filas = []
    for fila in matriz:
        s, vacias = "", 0
        for celda in fila:
            if not celda:
                vacias += 1
                continue
            if vacias:
                s += str(vacias)
                vacias = 0
            s += SIMBOLO[celda]
        if vacias:
            s += str(vacias)
        filas.append(s)
    return "/".join(filas) + f" {turno} - - 0 1"


# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------
def cargar_modelos(pesos_esquinas: Path, pesos_piezas: Path) -> tuple[Any, Any]:
    """Carga los dos modelos YOLO (import diferido de ultralytics)."""
    from ultralytics import YOLO

    return YOLO(str(pesos_esquinas)), YOLO(str(pesos_piezas))


def detectar_piezas(modelo: Any, imagen: np.ndarray) -> list[dict]:
    """Corre el modelo de piezas y devuelve detecciones normalizadas (bbox, clase, conf)."""
    res = modelo.predict(imagen, verbose=False)[0]
    dets = []
    for b in res.boxes:
        nombre = res.names[int(b.cls)]
        if nombre not in SIMBOLO:
            continue
        dets.append({"bbox": b.xyxy[0].cpu().numpy().tolist(), "clase": nombre, "conf": float(b.conf)})
    return dets


def guardar_debug(imagen: np.ndarray, H: np.ndarray, detecciones: list[dict], ruta: Path) -> None:
    """Guarda el tablero rectificado con grilla y detecciones."""
    import cv2

    rect = cv2.warpPerspective(imagen, H, (LADO, LADO))
    for i in range(9):
        cv2.line(rect, (i * CELDA, 0), (i * CELDA, LADO), (0, 255, 0), 1)
        cv2.line(rect, (0, i * CELDA), (LADO, i * CELDA), (0, 255, 0), 1)
    if detecciones:
        pts = np.array([[[(d["bbox"][0] + d["bbox"][2]) / 2, d["bbox"][3]]] for d in detecciones], dtype=np.float32)
        for d, (x, y) in zip(detecciones, cv2.perspectiveTransform(pts, H).reshape(-1, 2)):
            cv2.circle(rect, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(rect, SIMBOLO[d["clase"]], (int(x) + 6, int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.imwrite(str(ruta), rect)


# --------------------------------------------------------------------------
# Ajedrez / Lichess
# --------------------------------------------------------------------------
def validar_posicion(fen: str, forzar: bool = False) -> "chess.Board":  # noqa: F821
    """Construye el tablero y valida la posición (a menos que `forzar`)."""
    import chess

    try:
        board = chess.Board(fen)
    except ValueError as e:
        raise ErrorCLI(f"python-chess rechazó el FEN generado: {e}\nFEN: {fen}", EXIT_POSICION)
    if not forzar and not board.is_valid():
        estado = board.status()
        motivos = [f.name for f in chess.Status if f.value and (estado & f)]
        raise ErrorCLI(
            f"Posición inválida según python-chess: {', '.join(motivos)}.\nFEN generado: {fen}\n"
            "Revisá la detección o usá --forzar para consultar igual.",
            EXIT_POSICION,
        )
    return board


def consultar_lichess(fen: str, timeout: float = 15.0) -> dict:
    """Consulta cloud-eval de Lichess; devuelve el JSON de la respuesta."""
    import requests

    try:
        r = requests.get(LICHESS_URL, params={"fen": fen, "multiPv": 1}, timeout=timeout)
    except requests.exceptions.Timeout:
        raise ErrorCLI(f"Lichess no respondió en {timeout:.0f} s (timeout).", EXIT_LICHESS)
    except requests.exceptions.ConnectionError:
        raise ErrorCLI("No hay conexión con Lichess. Revisá tu conexión a internet.", EXIT_LICHESS)
    except requests.exceptions.RequestException as e:
        raise ErrorCLI(f"Error de red al consultar Lichess: {e}", EXIT_LICHESS)
    if r.status_code == 404:
        raise ErrorCLI("Lichess no tiene esta posición en su caché de cloud-eval (404). "
                       "Es probable que la detección tenga errores; revisá el FEN con --debug.", EXIT_LICHESS)
    if r.status_code == 429:
        raise ErrorCLI("Lichess limitó las consultas (429, rate limit). Esperá un minuto y reintentá.", EXIT_LICHESS)
    if r.status_code != 200:
        raise ErrorCLI(f"Lichess respondió con HTTP {r.status_code}.", EXIT_LICHESS)
    try:
        return r.json()
    except ValueError:
        raise ErrorCLI("Lichess devolvió una respuesta que no es JSON válido.", EXIT_LICHESS)


def extraer_jugada(data: dict) -> tuple[str, int | None, int | None]:
    """Devuelve (uci, cp, mate) del primer pv."""
    pvs = data.get("pvs") if isinstance(data, dict) else None
    if not pvs or not pvs[0].get("moves"):
        raise ErrorCLI("La respuesta de Lichess no contiene variantes (pvs).", EXIT_LICHESS)
    pv = pvs[0]
    return pv["moves"].split()[0], pv.get("cp"), pv.get("mate")


def uci_a_movimiento(board: "chess.Board", uci: str) -> "chess.Move":  # noqa: F821
    """Convierte el UCI de Lichess a un Move legal, normalizando el enroque.

    Lichess expresa el enroque como rey->torre (e1h1); se pasa a notación estándar (e1g1).
    Lanza ErrorCLI si la jugada es ilegal.
    """
    import chess

    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        raise ErrorCLI(f"Lichess devolvió una jugada con formato inválido: {uci!r}.", EXIT_JUGADA_ILEGAL)
    pieza = board.piece_at(move.from_square)
    destino = board.piece_at(move.to_square)
    if (pieza is not None and pieza.piece_type == chess.KING and destino is not None
            and destino.piece_type == chess.ROOK and destino.color == pieza.color):
        # rey toma torre propia = enroque; mapear a la casilla estándar del rey
        rank = chess.square_rank(move.from_square)
        archivo = 6 if chess.square_file(move.to_square) > chess.square_file(move.from_square) else 2
        move = chess.Move(move.from_square, chess.square(archivo, rank))
    if move not in board.legal_moves:
        raise ErrorCLI(f"Lichess devolvió una jugada ilegal para este tablero: {uci}.", EXIT_JUGADA_ILEGAL)
    return move


def formatear_evaluacion(cp: int | None, mate: int | None) -> str:
    """Formatea la evaluación: '+0.19' (peones, desde las blancas) o 'mate en N'."""
    if mate is not None:
        return f"mate en {abs(mate)} ({'a favor de las blancas' if mate > 0 else 'a favor de las negras'})"
    if cp is not None:
        return f"{cp / 100:+.2f}"
    return "sin evaluación"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parsear_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Digitaliza un tablero de ajedrez desde una foto y pide a Lichess la mejor jugada.")
    p.add_argument("imagen", metavar="IMAGEN", help="Ruta a la foto del tablero.")
    p.add_argument("direccion", metavar="DIRECCION", choices=DIRECCIONES,
                   help="Hacia dónde avanzan las blancas en la imagen: " + ", ".join(DIRECCIONES)
                   + " ('arriba' = las blancas están abajo en la foto).")
    p.add_argument("--turno", choices=["w", "b"], default="w", help="Quién mueve (por defecto: w).")
    p.add_argument("--pesos-esquinas", type=Path, default=PESOS_ESQUINAS,
                   help=f"Pesos del modelo de esquinas (por defecto: {PESOS_ESQUINAS}).")
    p.add_argument("--pesos-piezas", type=Path, default=PESOS_PIEZAS,
                   help=f"Pesos del modelo de piezas (por defecto: {PESOS_PIEZAS}).")
    p.add_argument("--debug", action="store_true", help="Guarda debug_rectificado.png con grilla y detecciones.")
    p.add_argument("--forzar", action="store_true", help="Consulta a Lichess aunque la posición sea inválida.")
    return p.parse_args(argv)


def ejecutar(args: argparse.Namespace) -> int:
    """Corre el flujo completo; lanza ErrorCLI ante errores esperados."""
    import chess
    import cv2

    imagen_path = Path(args.imagen)
    if not imagen_path.is_file():
        raise ErrorCLI(f"No existe la imagen: {imagen_path}", EXIT_ARCHIVO)
    for etiqueta, ruta in (("esquinas", args.pesos_esquinas), ("piezas", args.pesos_piezas)):
        if not Path(ruta).is_file():
            raise ErrorCLI(f"No existen los pesos de {etiqueta}: {ruta}. "
                           "Entrenalos en el notebook y copialos a tpFinal/models/.", EXIT_ARCHIVO)
    imagen = cv2.imread(str(imagen_path))
    if imagen is None:
        raise ErrorCLI(f"No se pudo leer la imagen (formato inválido): {imagen_path}", EXIT_ARCHIVO)

    print("Cargando modelos...")
    m_esq, m_pie = cargar_modelos(args.pesos_esquinas, args.pesos_piezas)

    esquinas = esquinas_desde_yolo(m_esq.predict(imagen, verbose=False))
    H = calcular_homografia(esquinas)
    detecciones = detectar_piezas(m_pie, imagen)
    matriz = mapear_piezas(detecciones, H)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        matriz = normalizar_orientacion(matriz, args.direccion)
    for w in ws:
        print(f"ADVERTENCIA: {w.message}", file=sys.stderr)
    fen = matriz_a_fen(matriz, args.turno)

    if args.debug:
        guardar_debug(imagen, H, detecciones, Path("debug_rectificado.png"))
        print("Debug guardado en debug_rectificado.png")

    print(f"FEN: {fen}")
    board = validar_posicion(fen, args.forzar)
    print(board)

    if board.is_game_over() or not any(board.legal_moves):
        if board.is_checkmate():
            motivo = "jaque mate"
        elif board.is_game_over():
            motivo = f"tablas/fin de partida ({board.result()})"
        else:
            motivo = "no hay jugadas legales"
        raise ErrorCLI(f"No hay jugada posible: {motivo}.", EXIT_SIN_JUGADA)

    uci, cp, mate = extraer_jugada(consultar_lichess(fen))
    move = uci_a_movimiento(board, uci)
    print(f"Mejor jugada (UCI): {move.uci()}")
    print(f"Mejor jugada (SAN): {board.san(move)}")
    print(f"Evaluación: {formatear_evaluacion(cp, mate)}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada del CLI."""
    args = parsear_args(argv)
    try:
        return ejecutar(args)
    except ErrorCLI as e:
        print(f"ERROR: {e.mensaje}", file=sys.stderr)
        return e.codigo


if __name__ == "__main__":
    sys.exit(main())
