import argparse
from pathlib import Path
import sys
import cv2
from ultralytics import YOLO
import yaml

# Rutas predeterminadas relativas al directorio donde se encuentra este script
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "best.pt"
DEFAULT_DATA = SCRIPT_DIR / "data.yaml"
DEFAULT_OUTPUT = SCRIPT_DIR / "output"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_labels_from_yaml(yaml_path: Path) -> dict[int, str] | None:
    """Carga los nombres de las clases desde un archivo data.yaml."""
    if not yaml_path.exists():
        print(f"[AVISO] No se encontró el archivo de etiquetas en {yaml_path}")
        return None

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or "names" not in data:
            print(f"[AVISO] El archivo {yaml_path} no contiene la clave 'names'.")
            return None

        names = data["names"]
        if isinstance(names, list):
            return {i: name for i, name in enumerate(names)}
        elif isinstance(names, dict):
            return {int(k): v for k, v in names.items()}
        else:
            print(f"[AVISO] Formato de 'names' desconocido en {yaml_path}.")
            return None
    except Exception as exc:
        print(f"[AVISO] Error al leer {yaml_path}: {exc}")
        return None


def load_model(model_path: Path, data_path: Path | None = None) -> YOLO:
    """Carga el modelo YOLOv8 y sincroniza sus etiquetas con data.yaml si está disponible."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de pesos del modelo en: {model_path}"
        )

    print(f"Cargando modelo desde: {model_path.resolve()}")
    model = YOLO(str(model_path))

    # Sincronizar etiquetas desde data.yaml si está presente
    if data_path is not None:
        yaml_names = load_labels_from_yaml(data_path)
        if yaml_names:
            if hasattr(model, "model") and hasattr(model.model, "names"):
                model.model.names = yaml_names
            print(f"Etiquetas cargadas desde {data_path.name}: {yaml_names}")

    print(f"Clases configuradas ({len(model.names)}): {model.names}")
    return model


def collect_images(sources: list[str]) -> list[Path]:
    """Recolecta y valida todas las rutas de imágenes a procesar."""
    image_paths: list[Path] = []

    for src in sources:
        path = Path(src).expanduser().resolve()
        if not path.exists():
            print(f"[ERROR] Ruta inexistente: {src}")
            continue

        if path.is_dir():
            found = [
                f for f in sorted(path.iterdir())
                if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
            ]
            if not found:
                print(f"[AVISO] No se hallaron imágenes compatibles en la carpeta: {path}")
            image_paths.extend(found)
        elif path.is_file():
            if path.suffix.lower() not in VALID_EXTENSIONS:
                print(
                    f"[AVISO] El archivo '{path.name}' tiene extensión no habitual, "
                    f"se intentará procesar igualmente."
                )
            image_paths.append(path)

    return image_paths


def show_annotated_window(window_title: str, image) -> bool:
    """Muestra una imagen anotada en una ventana OpenCV con soporte para salir o continuar.
    
    Retorna True si el usuario solicita cancelar el procesamiento restante (presionando 'q' o ESC).
    """
    try:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        print(f"  -> Ventana abierta: '{window_title}'")
        print("     [Controles]: Cualquier tecla para continuar | 'q' o ESC para salir")

        user_quit = False
        while True:
            cv2.imshow(window_title, image)
            key = cv2.waitKey(50) & 0xFF

            # Salir con 'q', 'Q' o ESC
            if key in (ord("q"), ord("Q"), 27):
                user_quit = True
                break

            # Avanzar si se presiona cualquier otra tecla
            if key != 255:
                break

            # Salir si el usuario cerró la ventana con la cruz (X)
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyWindow(window_title)
        return user_quit
    except Exception as exc:
        print(f"  [AVISO] No se pudo mostrar la ventana interactiva: {exc}")
        return False


def process_images(
    model: YOLO,
    images: list[Path],
    conf: float = 0.25,
    save: bool = True,
    show: bool = True,
    output_dir: Path = DEFAULT_OUTPUT,
    device: str | None = None,
) -> None:
    """Ejecuta la detección de objetos sobre cada imagen y muestra/guarda los resultados."""
    if not images:
        print("[INFO] No hay imágenes válidas para procesar.")
        return

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nIniciando detección sobre {len(images)} imagen(es) con umbral conf={conf}...")

    try:
        user_quit = False
        for img_path in images:
            print("\n" + "=" * 60)
            print(f"Imagen: {img_path.name}")
            print(f"Ruta:   {img_path}")

            try:
                results = model.predict(
                    source=str(img_path),
                    conf=conf,
                    device=device,
                    verbose=False,
                )
            except Exception as exc:
                print(f"[ERROR] Falló la inferencia sobre {img_path}: {exc}")
                continue

            for result in results:
                boxes = result.boxes
                num_detections = len(boxes) if boxes is not None else 0
                print(f"Detecciones encontradas: {num_detections}")

                if num_detections > 0 and boxes is not None:
                    for idx, box in enumerate(boxes):
                        cls_id = int(box.cls[0].item())
                        cls_name = model.names.get(cls_id, f"Clase_{cls_id}")
                        confidence = float(box.conf[0].item())
                        coords = [round(val, 2) for val in box.xyxy[0].tolist()]

                        print(
                            f"  [{idx + 1}] {cls_name:<10} "
                            f"| Confianza: {confidence * 100:>5.1f}% "
                            f"| Bounding Box: {coords}"
                        )
                else:
                    print(f"  (No se superó el umbral de confianza {conf})")

                # Obtener la imagen con las cajas y etiquetas dibujadas
                annotated_img = result.plot()

                # Guardar imagen con anotaciones en disco
                if save:
                    output_file = output_dir / f"pred_{img_path.name}"
                    cv2.imwrite(str(output_file), annotated_img)
                    print(f"-> Imagen con predicciones guardada en: {output_file.resolve()}")

                # Mostrar ventana interactiva con la imagen y cajas asignadas (estilo TP1/TP2/TP3)
                if show:
                    window_title = f"Deteccion YOLOv8 - {img_path.name}"
                    user_quit = show_annotated_window(window_title, annotated_img)
                    if user_quit:
                        print("\n[INFO] Visualización finalizada por el usuario.")
                        break

            if user_quit:
                break

    finally:
        cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("Procesamiento finalizado con éxito.")


def parse_arguments() -> argparse.Namespace:
    """Configura y parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Detección de objetos con modelo YOLOv8n entrenado (TP4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Ejemplos de uso:
  python main.py imagen.jpg
  python main.py imagen1.jpg imagen2.png --conf 0.4
  python main.py --source /ruta/a/carpeta/
  python main.py vt4_main.jpg --no-save
  python main.py vt4_main.jpg --no-show
""",
    )

    parser.add_argument(
        "images",
        nargs="*",
        help="Ruta(s) de una o más imágenes o carpetas a procesar.",
    )
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        default=None,
        help="Ruta alternativa de imagen o carpeta (opcional).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Ruta al archivo .pt del modelo (por defecto: {DEFAULT_MODEL.name}).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help=f"Ruta al archivo data.yaml con las etiquetas (por defecto: {DEFAULT_DATA.name}).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Umbral mínimo de confianza para detección [0.0 - 1.0] (por defecto: 0.25).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directorio donde se guardarán los resultados (por defecto: tp4/output).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="No abrir la ventana gráfica (útil para ejecución por lotes o entornos sin display).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Evitar guardar las imágenes con predicciones en disco.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo de cómputo (e.g. 'cpu', 'mps', '0'). Por defecto detecta automáticamente.",
    )

    return parser.parse_args()


def interactive_prompt() -> list[str]:
    """Solicita interactivamente la ruta de una imagen o carpeta al usuario."""
    print("=" * 60)
    print("DETECCIÓN DE OBJETOS YOLOv8 - TP4")
    print("=" * 60)
    print("No se pasaron imágenes como argumento en la línea de comandos.")
    print("Ingrese la ruta de una imagen o carpeta para analizar,")
    print("o presione ENTER / escriba 'q' para salir.")
    print("-" * 60)

    try:
        user_input = input("Ruta de imagen/carpeta: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nOperación cancelada.")
        return []

    if not user_input or user_input.lower() in {"q", "exit", "salir"}:
        return []

    return [user_input]


def main() -> None:
    args = parse_arguments()

    # Determinar fuentes de entrada
    sources: list[str] = []
    if args.source:
        sources.append(args.source)
    if args.images:
        sources.extend(args.images)

    # Si no se pasó ninguna ruta, entrar en modo interactivo
    if not sources:
        sources = interactive_prompt()
        if not sources:
            print("No se indicó ninguna imagen. Saliendo.")
            sys.exit(0)

    try:
        model = load_model(args.model, args.data)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    image_paths = collect_images(sources)
    if not image_paths:
        print("[ERROR] No se encontraron archivos válidos para procesar.")
        sys.exit(1)

    process_images(
        model=model,
        images=image_paths,
        conf=args.conf,
        save=not args.no_save,
        show=not args.no_show,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
