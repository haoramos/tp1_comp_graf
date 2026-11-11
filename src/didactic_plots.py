"""
Figuras didáticas dos Módulos 3 a 7 do TP1 (geradas com Matplotlib a partir dos
algoritmos implementados em rasterization.py e clipping.py).

Executar:  make plots   (ou  python main.py --mode plots)
"""
import os
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from src.vtk_loader import load_vtk_cco
from src.rasterization import bresenham_line, scanline_fill_polygon
from src.clipping import (
    ClipStats, cohen_sutherland_clip, liang_barsky_clip, sutherland_hodgman_clip,
    rect_polygon, find_overlapping_edges,
)

# Janela de recorte (xmin, ymin, xmax, ymax) usada nas demonstrações do Módulo 6.
DEFAULT_CLIP_WINDOW = (-0.022, -0.020, 0.022, 0.025)


def _save_fig(fig, filename, report_dir, pres_dir):
    """Salva a figura em report/figures e copia para presentation/figures."""
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(pres_dir, exist_ok=True)
    rep_path = os.path.join(report_dir, filename)
    pres_path = os.path.join(pres_dir, filename)
    fig.savefig(rep_path, dpi=300, bbox_inches="tight")
    shutil.copyfile(rep_path, pres_path)
    print(f"  [Salvo] {rep_path}")


def plot_module3_bresenham(model, report_dir, pres_dir):
    """
    Módulo 3. Painel 1: exemplo numérico (2,3)->(14,10) com os pixels gerados
    sobre a reta ideal. Painel 2: todos os segmentos da árvore 2D projetados numa
    grade 60x60 e rasterizados com o mesmo bresenham_line.
    """
    grid_size = 60
    grid = np.zeros((grid_size, grid_size))
    pts = model.points[:, :2]
    min_xy = np.min(pts, axis=0)
    max_xy = np.max(pts, axis=0)
    # Escala ÚNICA para x e y (preserva a proporção da árvore) e árvore centralizada na grade.
    scale = (grid_size - 5) / float(np.max(max_xy - min_xy))
    offset = (grid_size - 1 - (max_xy - min_xy) * scale) / 2.0

    def to_grid(pt):
        g = (pt - min_xy) * scale + offset
        return int(round(g[0])), int(round(g[1]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    bx0, by0, bx1, by1 = 2, 3, 14, 10
    single_pixels = bresenham_line(bx0, by0, bx1, by1)
    sub_grid = np.zeros((14, 18))
    for px, py in single_pixels:
        sub_grid[py, px] = 1.0

    axes[0].imshow(sub_grid, cmap="Blues", origin="lower", extent=[-0.5, 17.5, -0.5, 13.5])
    axes[0].plot([bx0, bx1], [by0, by1], color="red", linewidth=2, marker="o", label="Reta Contínua Ideal")
    axes[0].set_xticks(range(0, 18))
    axes[0].set_yticks(range(0, 14))
    axes[0].grid(True, which="both", color="gray", linestyle="--", linewidth=0.5)
    axes[0].set_title(f"Módulo 3: Bresenham ({bx0},{by0}) até ({bx1},{by1})\n{len(single_pixels)} pixels gerados")
    axes[0].set_xlabel("X (pixels)")
    axes[0].set_ylabel("Y (pixels)")
    axes[0].legend(loc="upper left")

    for s0, s1 in model.segments:
        gx0, gy0 = to_grid(pts[s0])
        gx1, gy1 = to_grid(pts[s1])
        for px, py in bresenham_line(gx0, gy0, gx1, gy1):
            if 0 <= px < grid_size and 0 <= py < grid_size:
                grid[py, px] = 1.0

    axes[1].imshow(grid, cmap="inferno", origin="lower")
    axes[1].set_title(f"Árvore CCO ({model.filename}) Rasterizada\nvia Bresenham (Grade {grid_size}x{grid_size})")
    axes[1].set_xlabel("X (pixels)")
    axes[1].set_ylabel("Y (pixels)")

    plt.tight_layout()
    _save_fig(fig, "mod3_bresenham.png", report_dir, pres_dir)
    plt.close(fig)


def plot_module4_scanline(report_dir, pres_dir):
    """
    Scanline Fill (ET/AET). Painel 1: SÓ o preenchimento (nada desenhado por cima),
    para que se veja exatamente os pixels gerados, inclusive a linha inferior.
    Painel 2: o mesmo preenchimento com o contorno de Bresenham. Painel 3: a
    convenção de borda (polígono fechado x semi-aberto) num quadrado 0..10.
    """
    poly_verts = [(8, 6), (38, 10), (44, 28), (28, 42), (18, 26), (6, 32)]   # côncavo (vértice (18,26))
    filled = scanline_fill_polygon(poly_verts)
    size = 50

    fill_only = np.zeros((size, size))
    for px, py in filled:
        fill_only[py, px] = 1
    with_contour = fill_only.copy()
    n = len(poly_verts)
    for i in range(n):
        x0, y0 = poly_verts[i]
        x1, y1 = poly_verts[(i + 1) % n]
        for px, py in bresenham_line(x0, y0, x1, y1):
            with_contour[py, px] = 2

    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    closed = set(scanline_fill_polygon(square))
    half_open = set(scanline_fill_polygon(square, include_boundary=False))
    sq = np.zeros((12, 12))
    for px, py in closed:
        sq[py, px] = 2                      # só no fechado (borda direita/superior)
    for px, py in half_open:
        sq[py, px] = 1                      # também no semi-aberto

    cmap = ListedColormap(["white", "#1f77b4", "#ff7f0e"])
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))

    axes[0].imshow(fill_only, cmap=cmap, origin="lower", vmin=0, vmax=2, interpolation="nearest")
    axes[0].plot(*zip(*(poly_verts + [poly_verts[0]])), color="red", linewidth=1.0, linestyle="--", label="Polígono ideal")
    axes[0].set_title(f"Scanline Fill: apenas o preenchimento\n({len(filled)} pixels, y = {min(p[1] for p in filled)}..{max(p[1] for p in filled)})")
    axes[0].legend(loc="upper right")

    axes[1].imshow(with_contour, cmap=cmap, origin="lower", vmin=0, vmax=2, interpolation="nearest")
    axes[1].set_title("Preenchimento (azul) + contorno por Bresenham (laranja)")

    axes[2].imshow(sq, cmap=cmap, origin="lower", vmin=0, vmax=2, interpolation="nearest")
    axes[2].plot(*zip(*(square + [square[0]])), color="red", linewidth=1.0, linestyle="--")
    axes[2].set_title(f"Convenção de borda no quadrado 0..10\nazul: semi-aberto ({len(half_open)} px) | + laranja: fechado ({len(closed)} px)")
    axes[2].set_xticks(range(0, 12))
    axes[2].set_yticks(range(0, 12))
    axes[2].grid(True, color="gray", alpha=0.3, linestyle=":")

    for ax in axes:
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
    plt.tight_layout()
    _save_fig(fig, "mod4_scanline.png", report_dir, pres_dir)
    plt.close(fig)


def rasterize_capsules(p0, p1, widths, width_px, height_px, k):
    """
    Rasteriza segmentos "grossos" (cápsulas) por amostragem de cobertura.

    Cada pixel (i, j) cobre o quadrado [i, i+1] x [j, j+1] do plano contínuo e é
    dividido em k x k subamostras. Uma subamostra conta se sua distância ao
    segmento é <= largura/2. A cobertura do pixel é a fração de subamostras
    dentro do traço:
      k = 1  -> 1 amostra no CENTRO do pixel (sem antialiasing: pixel 0 ou 1);
      k > 1  -> supersampling (SSAA) k x k: tons de cinza entre 0 e 1.
    Sem normalização: um pixel 100% coberto vale 1, um 50% coberto vale 0,5.

    p0, p1  : arrays (N, 2) com os extremos, em coordenadas de pixel (contínuas).
    widths  : array (N,) com a espessura de cada traço, em pixels.
    """
    mask = np.zeros((height_px * k, width_px * k), dtype=bool)
    for a, b, w in zip(p0, p1, widths):
        r = w / 2.0
        lo = np.floor((np.minimum(a, b) - r) * k).astype(int) - 1
        hi = np.ceil((np.maximum(a, b) + r) * k).astype(int) + 1
        ix0, iy0 = max(lo[0], 0), max(lo[1], 0)
        ix1, iy1 = min(hi[0], width_px * k), min(hi[1], height_px * k)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        X, Y = np.meshgrid((np.arange(ix0, ix1) + 0.5) / k, (np.arange(iy0, iy1) + 0.5) / k)
        d = b - a
        len2 = float(d @ d)
        t = np.zeros_like(X) if len2 == 0 else np.clip(((X - a[0]) * d[0] + (Y - a[1]) * d[1]) / len2, 0.0, 1.0)
        dist2 = (X - (a[0] + t * d[0])) ** 2 + (Y - (a[1] + t * d[1])) ** 2
        mask[iy0:iy1, ix0:ix1] |= dist2 <= r * r
    return mask.reshape(height_px, k, width_px, k).mean(axis=(1, 3))


def _aa_stats(img, reference):
    """(nº de níveis de cinza distintos, erro quadrático médio (RMSE) em relação à cobertura de referência)."""
    levels = len(np.unique(np.round(img, 6)))
    rmse = float(np.sqrt(np.mean((img - reference) ** 2)))
    return levels, rmse


def plot_module5_antialiasing(model, report_dir, pres_dir):
    """
    Comparação justa com/sem antialiasing: a MESMA geometria e a MESMA espessura
    são rasterizadas variando apenas o número de amostras por pixel (1, 2x2, 4x4).
    Linha 1: um segmento inclinado. Linha 2: a árvore CCO 2D (alta frequência espacial).
    A referência do erro é a cobertura calculada com 16x16 amostras por pixel.
    """
    ks = [1, 2, 4]
    names = {1: "Sem antialiasing\n(1 amostra no centro do pixel)",
             2: "SSAA 2x2\n(4 amostras por pixel)",
             4: "SSAA 4x4\n(16 amostras por pixel)"}
    ref_k = 16

    # ---------------- Linha 1: segmento inclinado, espessura 1 px ----------------
    w1, h1 = 40, 24
    a = np.array([[3.0, 4.5]])
    b = np.array([[37.0, 16.7]])      # inclinação ~ 0,36: degraus longos e bem visíveis
    width1 = np.array([1.0])
    line_imgs = {k: rasterize_capsules(a, b, width1, w1, h1, k) for k in ks}
    line_ref = rasterize_capsules(a, b, width1, w1, h1, ref_k)

    # ---------------- Linha 2: árvore CCO 2D em baixa resolução ------------------
    n = 90
    pts = model.points[:, :2]
    min_xy, max_xy = pts.min(axis=0), pts.max(axis=0)
    scale = (n - 6) / float(np.max(max_xy - min_xy))        # escala única em x e y
    offset = (n - (max_xy - min_xy) * scale) / 2.0
    px = (pts - min_xy) * scale + offset
    t0, t1 = px[model.segments[:, 0]], px[model.segments[:, 1]]
    max_w = 3.0                                              # espessura (px) do vaso de maior raio
    tree_w = max_w * model.radii / model.max_radius          # proporcional ao raio: 0,5 a 3 px
    tree_imgs = {k: rasterize_capsules(t0, t1, tree_w, n, n, k) for k in ks}
    tree_ref = rasterize_capsules(t0, t1, tree_w, n, n, ref_k)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.6),
                             gridspec_kw={"height_ratios": [24 / 40 * 1.15, 1.0]})
    for col, k in enumerate(ks):
        lv, err = _aa_stats(line_imgs[k], line_ref)
        ax = axes[0, col]
        ax.imshow(line_imgs[k], cmap="gray_r", origin="lower", extent=(0, w1, 0, h1),
                  interpolation="nearest", vmin=0.0, vmax=1.0)
        ax.plot([a[0, 0], b[0, 0]], [a[0, 1], b[0, 1]], color="red", linewidth=0.8, alpha=0.9,
                label="Reta ideal")
        ax.set_title(f"{names[k]}\n{lv} níveis de cinza | erro vs. ref.: {err:.3f}", fontsize=10)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        if col == 0:
            ax.legend(loc="upper left", fontsize=8)

        lv, err = _aa_stats(tree_imgs[k], tree_ref)
        ax = axes[1, col]
        ax.imshow(tree_imgs[k], cmap="gray_r", origin="lower", extent=(0, n, 0, n),
                  interpolation="nearest", vmin=0.0, vmax=1.0)
        ax.set_title(f"Árvore CCO 2D ({n}x{n} px): {lv} níveis | erro vs. ref.: {err:.3f}", fontsize=10)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

    fig.suptitle("Módulo 5: aliasing × antialiasing por supersampling (mesma geometria e mesma espessura)",
                 fontsize=12)
    plt.tight_layout()
    _save_fig(fig, "mod5_antialiasing.png", report_dir, pres_dir)
    plt.close(fig)

    print("  [Módulo 5] níveis de cinza / erro (RMSE) vs. referência 16x16")
    for k in ks:
        l_lv, l_err = _aa_stats(line_imgs[k], line_ref)
        t_lv, t_err = _aa_stats(tree_imgs[k], tree_ref)
        print(f"     k={k}: reta -> {l_lv:3d} níveis, erro {l_err:.4f} | árvore -> {t_lv:3d} níveis, erro {t_err:.4f}")


def _clip_all(fn, model, window, stats=None):
    """Recorta todos os segmentos do modelo (plano XY); devolve a lista de resultados (ou None)."""
    pts = model.points[:, :2]
    out = []
    for s0, s1 in model.segments:
        p0, p1 = pts[s0], pts[s1]
        out.append(fn(p0[0], p0[1], p1[0], p1[1], *window, stats=stats))
    return out


def _time_per_segment_us(fn, model, window, reps=300):
    """Tempo médio por segmento (microssegundos), sem contabilizar estatísticas."""
    pts = model.points[:, :2]
    segs = [(pts[a][0], pts[a][1], pts[b][0], pts[b][1]) for a, b in model.segments]
    t0 = time.perf_counter()
    for _ in range(reps):
        for x0, y0, x1, y1 in segs:
            fn(x0, y0, x1, y1, *window)
    return (time.perf_counter() - t0) / (reps * len(segs)) * 1e6


def plot_module6_line_clipping(model, report_dir, pres_dir):
    """
    Módulo 6. Recorta os 127 segmentos da árvore 2D com Cohen-Sutherland e com
    Liang-Barsky, confere que os resultados são idênticos e gera (1) as imagens
    antes/depois e (2) a comparação de custo (iterações, divisões, tempo).
    """
    pts = model.points[:, :2]
    window = DEFAULT_CLIP_WINDOW
    xmin, ymin, xmax, ymax = window

    st_cs, st_lb = ClipStats(), ClipStats()
    res_cs = _clip_all(cohen_sutherland_clip, model, window, st_cs)
    res_lb = _clip_all(liang_barsky_clip, model, window, st_lb)

    # Os dois algoritmos devem produzir exatamente o mesmo recorte.
    mismatches = 0
    for a, b in zip(res_cs, res_lb):
        if (a is None) != (b is None) or (a is not None and not np.allclose(a, b, atol=1e-12)):
            mismatches += 1
    n_cs = sum(r is not None for r in res_cs)
    n_lb = sum(r is not None for r in res_lb)
    t_cs = _time_per_segment_us(cohen_sutherland_clip, model, window)
    t_lb = _time_per_segment_us(liang_barsky_clip, model, window)

    # ---------------- Figura 1: antes / Cohen-Sutherland / Liang-Barsky ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharex=True, sharey=True)
    for s0, s1 in model.segments:
        p0, p1 = pts[s0], pts[s1]
        axes[0].plot([p0[0], p1[0]], [p0[1], p1[1]], color="#1f77b4", linewidth=1.2)
    axes[0].add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2,
                                        edgecolor="red", facecolor="none", linestyle="--", label="Janela"))
    axes[0].set_title(f"Antes do Recorte ({model.num_segments} segmentos)")
    axes[0].legend(loc="lower right")

    for ax, results, color, title, st in (
        (axes[1], res_cs, "#d62728", "Cohen-Sutherland", st_cs),
        (axes[2], res_lb, "#2ca02c", "Liang-Barsky", st_lb),
    ):
        for s0, s1 in model.segments:
            p0, p1 = pts[s0], pts[s1]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="lightgray", linewidth=0.7, linestyle=":")
        for r in results:
            if r is not None:
                ax.plot([r[0], r[2]], [r[1], r[3]], color=color, linewidth=1.8)
        ax.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2,
                                       edgecolor="black", facecolor="none"))
        n_vis = sum(r is not None for r in results)
        ax.set_title(f"{title}: {n_vis} visíveis\n{st.iterations} iterações | {st.intersections} divisões (interseções/parâmetros)")

    for ax in axes:
        ax.set_xlabel("Coordenada X")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Coordenada Y")
    plt.tight_layout()
    _save_fig(fig, "mod6_line_clipping.png", report_dir, pres_dir)
    plt.close(fig)

    # ---------------- Figura 2: comparação de custo entre os algoritmos ----------------
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    labels = ["Cohen-Sutherland", "Liang-Barsky"]
    colors = ["#d62728", "#2ca02c"]
    panels = [
        ("Iterações (total)", [st_cs.iterations, st_lb.iterations]),
        ("Divisões calculadas (total)", [st_cs.intersections, st_lb.intersections]),
        ("Tempo por segmento (µs, Python)", [t_cs, t_lb]),
    ]
    for ax, (title, vals) in zip(axes[:3], panels):
        bars = ax.bar(labels, vals, color=colors)
        ax.set_title(title, fontsize=10)
        ax.bar_label(bars, fmt="%.2f" if isinstance(vals[0], float) else "%d")
        ax.tick_params(axis="x", labelsize=8)
    axes[3].bar(["Aceitos\ntrivialmente", "Rejeitados\ntrivialmente", "Recortados\n(com interseção)"],
                [st_cs.trivial_accept, st_cs.trivial_reject,
                 st_cs.segments - st_cs.trivial_accept - st_cs.trivial_reject], color="#d62728")
    axes[3].set_title("Cohen-Sutherland: como cada\nsegmento foi resolvido", fontsize=10)
    axes[3].tick_params(axis="x", labelsize=8)
    for p in axes[3].patches:
        axes[3].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"Módulo 6: custo dos algoritmos nos {model.num_segments} segmentos "
                 f"(resultados idênticos: {mismatches} divergências)", fontsize=11)
    plt.tight_layout()
    _save_fig(fig, "mod6_comparacao_algoritmos.png", report_dir, pres_dir)
    plt.close(fig)

    print("  [Módulo 6] Cohen-Sutherland vs Liang-Barsky")
    print(f"     visíveis: {n_cs} vs {n_lb} | divergências: {mismatches}")
    print(f"     iterações: {st_cs.iterations} vs {st_lb.iterations} | divisões: {st_cs.intersections} (interseções reta-borda) vs {st_lb.intersections} (parâmetros u = q/p)")
    print(f"     CS aceitos/rejeitados trivialmente: {st_cs.trivial_accept}/{st_cs.trivial_reject}")
    print(f"     tempo/segmento: {t_cs:.2f} µs vs {t_lb:.2f} µs (Python puro; depende da máquina)")


def _draw_polygon_pair(axes_row, subject, window, clipped, title_after, overlaps=None, correct=None,
                       legend_loc="upper right"):
    """Desenha 'antes' (eixo 0) e 'depois' (eixo 1) de um recorte de polígono."""
    sub_closed = list(subject) + [subject[0]]
    win_closed = list(window) + [window[0]]
    sx, sy = zip(*sub_closed)
    wx, wy = zip(*win_closed)

    ax = axes_row[0]
    ax.fill(sx, sy, color="cornflowerblue", alpha=0.45, label="Polígono original")
    ax.plot(sx, sy, color="navy", marker="o", linewidth=1.8)
    ax.plot(wx, wy, color="red", linestyle="--", linewidth=2, label="Janela convexa")
    ax.set_title(f"Antes do recorte ({len(subject)} vértices)")
    ax.legend(loc=legend_loc, fontsize=8)

    ax = axes_row[1]
    ax.plot(sx, sy, color="lightgray", linestyle=":", linewidth=1.2, label="Original (fora)")
    if clipped:
        cx, cy = zip(*(list(clipped) + [clipped[0]]))
        ax.fill(cx, cy, color="mediumseagreen", alpha=0.6, label="Saída do Sutherland-Hodgman")
        ax.plot(cx, cy, color="darkgreen", marker="o", linewidth=2)
    if overlaps:
        for (i, j), _ in overlaps:
            for k in (i, j):
                a, b = clipped[k], clipped[(k + 1) % len(clipped)]
                ax.plot([a[0], b[0]], [a[1], b[1]], color="red", linewidth=4, alpha=0.8, solid_capstyle="butt")
        ax.plot([], [], color="red", linewidth=4, label="Arestas sobrepostas")
    if correct:
        for region in correct:
            rx, ry = zip(*(list(region) + [region[0]]))
            ax.plot(rx, ry, color="black", linestyle="--", linewidth=1.5)
        ax.plot([], [], color="black", linestyle="--", label="Resultado correto (2 regiões)")
    ax.plot(wx, wy, color="red", linestyle="--", linewidth=2, label="Janela convexa")
    ax.set_title(title_after)
    ax.legend(loc=legend_loc, fontsize=8)
    for a in axes_row:
        a.grid(True, alpha=0.3)
        a.set_aspect("equal")
        a.set_xlabel("X")
        a.set_ylabel("Y")


def plot_module7_polygon_clipping(report_dir, pres_dir):
    """Módulo 7: Sutherland-Hodgman num polígono qualquer e num caso côncavo ("U")."""
    # Caso 1: polígono qualquer, janela quadrada (resultado correto: uma região).
    subject_polygon = [(-2, 4), (4, -3), (12, 2), (13, 8), (6, 13), (2, 8), (-3, 9)]
    clip_window = rect_polygon(0, 0, 10, 10)
    clipped_poly = sutherland_hodgman_clip(subject_polygon, clip_window)

    # Caso 2: polígono em "U" (côncavo) com janela y em [5, 10]: o resultado correto
    # são DUAS regiões (os dois braços), mas Sutherland-Hodgman devolve uma única lista.
    u_poly = [(0, 0), (10, 0), (10, 12), (7, 12), (7, 3), (3, 3), (3, 12), (0, 12)]
    u_window = rect_polygon(-1, 5, 11, 10)
    u_clipped = sutherland_hodgman_clip(u_poly, u_window)
    overlaps = find_overlapping_edges(u_clipped)
    correct = [rect_polygon(0, 5, 3, 10), rect_polygon(7, 5, 10, 10)]

    # Janela em sentido horário: agora é reorientada (antes devolvia lista vazia sem aviso).
    cw_result = sutherland_hodgman_clip(subject_polygon, clip_window[::-1])

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    _draw_polygon_pair(axes[0], subject_polygon, clip_window, clipped_poly,
                       f"Após Sutherland-Hodgman ({len(clipped_poly)} vértices)")
    _draw_polygon_pair(axes[1], u_poly, u_window, u_clipped,
                       f"Polígono côncavo: {len(u_clipped)} vértices, {len(overlaps)} par(es) de arestas sobrepostas",
                       overlaps=overlaps, correct=correct, legend_loc="lower center")
    plt.tight_layout()
    _save_fig(fig, "mod7_polygon_clipping.png", report_dir, pres_dir)
    plt.close(fig)

    print("  [Módulo 7] Sutherland-Hodgman")
    print(f"     caso 1: {len(clipped_poly)} vértices | janela horária reorientada: {len(cw_result)} vértices (igual: {len(cw_result) == len(clipped_poly)})")
    print(f"     caso U: {len(u_clipped)} vértices, arestas sobrepostas: {overlaps}")


def generate_all_didactic_figures():
    """Gera todas as figuras dos Módulos 3 a 7 a partir do modelo 2D."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vtk_2d_path = os.path.join(base_dir, "data", "tree2D_Nterm064.vtk")
    report_fig_dir = os.path.join(base_dir, "report", "figures")
    pres_fig_dir = os.path.join(base_dir, "presentation", "figures")

    print("=== GERANDO FIGURAS DIDÁTICAS DOS MÓDULOS 3 A 7 ===")
    model_2d = load_vtk_cco(vtk_2d_path)
    plot_module3_bresenham(model_2d, report_fig_dir, pres_fig_dir)
    plot_module4_scanline(report_fig_dir, pres_fig_dir)
    plot_module5_antialiasing(model_2d, report_fig_dir, pres_fig_dir)
    plot_module6_line_clipping(model_2d, report_fig_dir, pres_fig_dir)
    plot_module7_polygon_clipping(report_fig_dir, pres_fig_dir)
    print("=== TODAS AS FIGURAS FORAM GERADAS COM SUCESSO! ===")


if __name__ == "__main__":
    generate_all_didactic_figures()