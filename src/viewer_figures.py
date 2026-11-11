"""
Capturas do visualizador CCO (Módulos 1, 2 e 5) para o relatório e a apresentação.

Usa o PRÓPRIO CCOViewer de viewer_3d.py em modo off-screen (sem abrir janela) e
chama os mesmos métodos que as teclas do visualizador acionam (translate,
rotate, scale, set_orthographic, view_front, ...). Assim as figuras mostram
exatamente o que o usuário vê ao executar `make run-3d` / `make run-2d`.

Executar:  make screenshots   (ou  python main.py --mode screenshots)

Figuras geradas (em report/figures e presentation/figures):
  mod1_primitivas.png     - tubos com espessura proporcional x constante; modelo 2D
  mod2_transformacoes.png - original, translação, rotação e escala (câmera fixa)
  mod2_projecoes.png      - mesma câmera em projeção perspectiva x ortográfica
  mod2_vistas.png         - vistas frontal, lateral, superior e isométrica
  mod5_viewer_aa.png      - recorte ampliado sem AA, com MSAA e com SSAA
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from src.vtk_loader import load_vtk_cco
from src.viewer_3d import CCOViewer
from src.didactic_plots import finalize_png, REPORT_MAX_WIDTH, SLIDES_MAX_WIDTH

WINDOW = (900, 800)   # resolução (px) de cada captura


def _new_viewer(model, aa="msaa"):
    """Cria um CCOViewer off-screen, sem HUD, com o antialiasing pedido."""
    v = CCOViewer(model, off_screen=True, window_size=WINDOW)
    v.show_hud = False
    v.show_scalar_bar = False
    v.plotter.set_background("white")
    if aa is None:
        v.aa_idx = 0
        v.plotter.disable_anti_aliasing()
    else:
        v.aa_idx = CCOViewer.AA_MODES.index(aa)
        v.plotter.enable_anti_aliasing(aa)
    v._draw()
    v.plotter.show(auto_close=False)       # inicializa o render window off-screen
    return v


def _shot(viewer):
    """Renderiza o estado atual do viewer e devolve a imagem RGB (H, W, 3)."""
    viewer.plotter.render()
    return viewer.plotter.screenshot(return_img=True)


def _save(fig, filename, report_dir, pres_dir):
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(pres_dir, exist_ok=True)
    path = os.path.join(report_dir, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    # Versões RGB e mais leves (ver finalize_png): compilam rápido no Overleaf.
    finalize_png(path, os.path.join(pres_dir, filename), SLIDES_MAX_WIDTH)
    finalize_png(path, path, REPORT_MAX_WIDTH)
    plt.close(fig)
    print(f"  [Salvo] {path}")


def _crop_common(images, pad=20):
    """
    Recorta TODAS as imagens com a mesma caixa: a união das regiões não brancas.
    Usar a mesma caixa preserva a posição relativa da árvore entre os quadros
    (necessário para enxergar a translação, a rotação e a escala).
    """
    mask = np.zeros(images[0].shape[:2], dtype=bool)
    for img in images:
        mask |= np.any(img[..., :3] < 245, axis=2)
    ys, xs = np.nonzero(mask)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, mask.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, mask.shape[1])
    return [img[y0:y1, x0:x1] for img in images]


def _grid(images, titles, ncols, figsize, suptitle=None, colorbar=None):
    """Monta as capturas lado a lado; colorbar=(r_min, r_max) adiciona a escala do raio."""
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    if colorbar is not None:
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=matplotlib.colors.Normalize(*colorbar))
        fig.colorbar(sm, ax=list(axes), shrink=0.6, label="Raio (atributo do VTK, modelo 3D)")
    return fig


def figure_primitives(model_3d, model_2d, report_dir, pres_dir):
    """Módulo 1: tubos com espessura proporcional ao raio x espessura constante."""
    v = _new_viewer(model_3d)
    v.view_iso()
    prop = _shot(v)
    v.toggle_thickness()                   # tecla C
    const = _shot(v)
    v.plotter.close()

    v2 = _new_viewer(model_2d)
    v2.view_front()
    flat = _shot(v2)
    v2.plotter.close()

    fig = _grid(
        _crop_common([prop, const]) + _crop_common([flat]),
        ["3D: cor e espessura ∝ raio", "3D: espessura constante\n(só a cor codifica o raio)",
         "2D: cor e espessura ∝ raio"],
        ncols=3, figsize=(16, 5.4), colorbar=(model_3d.min_radius, model_3d.max_radius),
    )
    _save(fig, "mod1_primitivas.png", report_dir, pres_dir)


def figure_transformations(model_3d, report_dir, pres_dir):
    """Módulo 2: cada transformação aplicada a partir do original, com a câmera fixa."""
    v = _new_viewer(model_3d)
    v.view_side()                          # plano YZ: Y na horizontal, Z na vertical
    v.plotter.camera.zoom(0.7)             # deixa folga para a árvore se mover na imagem
    images, titles = [_shot(v)], ["Original (vista lateral, câmera fixa)"]

    v.set_axis("y")
    for _ in range(4):
        v.translate(+1)                    # tecla T (4x) no eixo Y
    images.append(_shot(v)); titles.append("Translação em Y (4 passos de 5% da diagonal)")
    v.reset_transformations()

    v.set_axis("x")
    for _ in range(3):
        v.rotate(+1)                       # tecla K (3x) no eixo X (perpendicular à tela)
    images.append(_shot(v)); titles.append("Rotação de 45° em X (pivô no centroide)")
    v.reset_transformations()

    for _ in range(5):
        v.scale(1.1)                       # tecla + (5x)
    images.append(_shot(v)); titles.append(f"Escala uniforme ×{1.1 ** 5:.2f} (centroide fixo)")
    v.plotter.close()

    fig = _grid(_crop_common(images), titles, ncols=4, figsize=(18, 5))
    _save(fig, "mod2_transformacoes.png", report_dir, pres_dir)


def figure_projections(model_3d, report_dir, pres_dir):
    """Módulo 2: mesma posição de câmera, projeção perspectiva x ortográfica."""
    v = _new_viewer(model_3d)
    v.view_iso()
    v.plotter.camera.zoom(1.4)             # câmera próxima: realça o encurtamento perspectivo
    persp = _shot(v)
    v.set_orthographic()                   # tecla O
    ortho = _shot(v)
    v.plotter.close()
    fig = _grid(_crop_common([persp, ortho]), ["Projeção perspectiva (tecla V)", "Projeção ortográfica (tecla O)"],
                ncols=2, figsize=(12, 5.6))
    _save(fig, "mod2_projecoes.png", report_dir, pres_dir)


def figure_views(model_3d, report_dir, pres_dir):
    """Módulo 2: quatro posições do observador."""
    v = _new_viewer(model_3d)
    images = []
    for fn in (v.view_front, v.view_side, v.view_top, v.view_iso):
        fn()
        images.append(_shot(v))
    v.plotter.close()
    fig = _grid(_crop_common(images), ["Frontal (B): plano XY", "Lateral (N): plano YZ",
                         "Superior (U): plano XZ", "Isométrica (I)"],
                ncols=4, figsize=(18, 4.8))
    _save(fig, "mod2_vistas.png", report_dir, pres_dir)


def figure_viewer_antialiasing(model_3d, report_dir, pres_dir):
    """Módulo 5 (API): mesmo quadro sem AA, com MSAA e com SSAA, ampliado num recorte."""
    images = []
    for aa in (None, "msaa", "ssaa"):
        v = _new_viewer(model_3d, aa=aa)
        v.toggle_thickness()               # tubos finos e constantes: mais bordas inclinadas
        v.view_iso()
        img = _shot(v)
        v.plotter.close()
        h, w = img.shape[:2]
        images.append(img[int(0.30 * h):int(0.55 * h), int(0.35 * w):int(0.65 * w)])
    fig = _grid(images, ["Sem antialiasing", "MSAA (multisampling)", "SSAA (supersampling)"],
                ncols=3, figsize=(16, 4.6),
                suptitle="Antialiasing no visualizador (tecla A): recorte ampliado da mesma região")
    for ax in fig.axes:
        for im in ax.get_images():
            im.set_interpolation("nearest")    # mostra os pixels sem suavização do Matplotlib
    _save(fig, "mod5_viewer_aa.png", report_dir, pres_dir)


def generate_viewer_figures():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report_dir = os.path.join(base_dir, "report", "figures")
    pres_dir = os.path.join(base_dir, "presentation", "figures")
    model_3d = load_vtk_cco(os.path.join(base_dir, "data", "tree3D_Nterm064.vtk"))
    model_2d = load_vtk_cco(os.path.join(base_dir, "data", "tree2D_Nterm064.vtk"))

    print("=== GERANDO CAPTURAS DO VISUALIZADOR (MÓDULOS 1, 2 E 5) ===")
    figure_primitives(model_3d, model_2d, report_dir, pres_dir)
    figure_transformations(model_3d, report_dir, pres_dir)
    figure_projections(model_3d, report_dir, pres_dir)
    figure_views(model_3d, report_dir, pres_dir)
    figure_viewer_antialiasing(model_3d, report_dir, pres_dir)
    print("=== CAPTURAS GERADAS COM SUCESSO! ===")


if __name__ == "__main__":
    matplotlib.use("Agg")
    generate_viewer_figures()
