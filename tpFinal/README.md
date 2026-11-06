# TP Final - Visión Artificial: tablero de ajedrez a FEN y mejor jugada

1. Ejecutá `TP_Final_VA.ipynb` en Google Colab (GPU) para entrenar los dos modelos YOLO.
2. Copiá `corners.pt` y `pieces.pt` a `tpFinal/models/`.
3. Instalá dependencias y ejecutá el CLI:

```bash
pip install -r requirements.txt
python chess_cli.py foto.jpg arriba --turno w --debug
```

`DIRECCION` (`arriba`, `abajo`, `izquierda`, `derecha`) indica hacia dónde avanzan las blancas en la imagen
(`arriba` = las blancas están abajo en la foto). Opciones: `--turno w|b`, `--pesos-esquinas`, `--pesos-piezas`,
`--debug` (guarda `debug_rectificado.png`), `--forzar` (consulta aunque la posición sea inválida).

La mejor jugada se pide a la API cloud-eval de Lichess (sin token). Solo cubre posiciones en su caché y el FEN
generado no incluye enroque ni al paso.
