"""
Módulos 6 e 7 do TP1: recorte de segmentos de reta e de polígonos.

Implementação própria do grupo (sem bibliotecas gráficas):
  - Cohen-Sutherland   (segmentos, obrigatório)
  - Liang-Barsky       (segmentos, paramétrico)
  - Sutherland-Hodgman (polígonos contra janela convexa)

Os algoritmos de segmentos aceitam um objeto ClipStats opcional que acumula
métricas de custo (iterações e interseções calculadas), usadas na comparação
entre os dois algoritmos.

Cyrus-Beck (terceiro algoritmo, apenas discutido no relatório): é a
generalização de Liang-Barsky para janelas CONVEXAS QUAISQUER. Em vez dos 4
testes fixos de uma janela retangular alinhada aos eixos, usa, para cada aresta
da janela, o vetor normal N_i e o produto N_i . (P1 - P0) para decidir se o
parâmetro t é de entrada ou de saída. Liang-Barsky é o caso particular em que
as normais são (+-1, 0) e (0, +-1).
"""
from dataclasses import dataclass


# ----------------------------------------------------------------------------
# Métricas de custo
# ----------------------------------------------------------------------------
@dataclass
class ClipStats:
    """Contadores acumulados ao longo de várias chamadas de recorte."""
    segments: int = 0          # segmentos processados
    iterations: int = 0        # CS: passagens do laço; LB: testes de borda (p_i, q_i)
    intersections: int = 0     # divisões calculadas: CS = interseções reta-borda; LB = parâmetros u = q/p
    trivial_accept: int = 0    # segmentos aceitos sem nenhuma interseção (CS)
    trivial_reject: int = 0    # segmentos rejeitados sem nenhuma interseção (CS)


def _is_degenerate(x0, y0, x1, y1, xmin, ymin, xmax, ymax, orig_len2):
    """
    True se o resultado do recorte tem comprimento nulo enquanto o segmento
    original tinha comprimento > 0: o segmento só TOCA a janela num ponto
    (por exemplo, um canto). Esse ponto não é desenhável e não deve contar
    como "visível".
    """
    diag = max(xmax - xmin, ymax - ymin)
    eps = 1e-12 * diag
    new_len2 = (x1 - x0) ** 2 + (y1 - y0) ** 2
    return orig_len2 > eps * eps and new_len2 <= eps * eps


# ----------------------------------------------------------------------------
# Módulo 6: recorte de segmentos
# ----------------------------------------------------------------------------
# Códigos de região (outcodes) para Cohen-Sutherland
INSIDE = 0  # 0000
LEFT = 1    # 0001
RIGHT = 2   # 0010
BOTTOM = 4  # 0100
TOP = 8     # 1000


def _compute_outcode(x, y, xmin, ymin, xmax, ymax):
    """
    Código de região de 4 bits (TOP BOTTOM RIGHT LEFT) do ponto (x, y) em
    relação à janela. 0000 = dentro. Dois pontos com AND != 0 estão do mesmo
    lado de fora (rejeição trivial); dois pontos com OR == 0 estão dentro
    (aceitação trivial).
    """
    code = INSIDE
    if x < xmin:
        code |= LEFT
    elif x > xmax:
        code |= RIGHT
    if y < ymin:
        code |= BOTTOM
    elif y > ymax:
        code |= TOP
    return code


def cohen_sutherland_clip(x0, y0, x1, y1, xmin, ymin, xmax, ymax, stats=None, drop_degenerate=True):
    """
    Cohen-Sutherland (implementação obrigatória).

    Devolve (x0', y0', x1', y1') com o trecho visível, ou None se não há trecho
    visível. Com drop_degenerate=True, um segmento que apenas toca a janela num
    ponto também devolve None. 'stats' (ClipStats) acumula o custo da chamada.
    """
    orig_len2 = (x1 - x0) ** 2 + (y1 - y0) ** 2
    code0 = _compute_outcode(x0, y0, xmin, ymin, xmax, ymax)
    code1 = _compute_outcode(x1, y1, xmin, ymin, xmax, ymax)
    accept = False
    n_inter = 0

    while True:
        if stats is not None:
            stats.iterations += 1
        if not (code0 | code1):                 # ambos dentro: aceitação trivial
            accept = True
            break
        if code0 & code1:                       # ambos do mesmo lado de fora: rejeição trivial
            break

        # Escolhe um extremo que está fora e o move para a interseção com a
        # borda indicada pelo seu código (uma borda por iteração).
        outcode_out = code0 if code0 else code1
        if outcode_out & TOP:
            x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0)
            y = ymax
        elif outcode_out & BOTTOM:
            x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0)
            y = ymin
        elif outcode_out & RIGHT:
            y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0)
            x = xmax
        else:  # LEFT
            y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0)
            x = xmin
        n_inter += 1

        if outcode_out == code0:
            x0, y0 = x, y
            code0 = _compute_outcode(x0, y0, xmin, ymin, xmax, ymax)
        else:
            x1, y1 = x, y
            code1 = _compute_outcode(x1, y1, xmin, ymin, xmax, ymax)

    if stats is not None:
        stats.segments += 1
        stats.intersections += n_inter
        if n_inter == 0:
            if accept:
                stats.trivial_accept += 1
            else:
                stats.trivial_reject += 1

    if not accept:
        return None
    if drop_degenerate and _is_degenerate(x0, y0, x1, y1, xmin, ymin, xmax, ymax, orig_len2):
        return None
    return (x0, y0, x1, y1)


def liang_barsky_clip(x0, y0, x1, y1, xmin, ymin, xmax, ymax, stats=None, drop_degenerate=True):
    """
    Liang-Barsky (algoritmo paramétrico).

    P(u) = P0 + u (P1 - P0), 0 <= u <= 1. Para cada borda i: p_i u <= q_i.
      p_i < 0 -> o segmento ENTRA na janela por essa borda (atualiza u1 = max)
      p_i > 0 -> o segmento SAI da janela por essa borda   (atualiza u2 = min)
      p_i = 0 -> paralelo à borda: rejeita se q_i < 0 (fora).
    O trecho visível é [u1, u2], se u1 <= u2.
    """
    dx = x1 - x0
    dy = y1 - y0
    p = (-dx, dx, -dy, dy)
    q = (x0 - xmin, xmax - x0, y0 - ymin, ymax - y0)
    orig_len2 = dx * dx + dy * dy
    if stats is not None:
        stats.segments += 1

    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if stats is not None:
            stats.iterations += 1
        if pi == 0:
            if qi < 0:
                return None  # paralelo e totalmente fora da janela
        else:
            u = qi / pi
            if stats is not None:
                stats.intersections += 1
            if pi < 0:
                if u > u2:
                    return None
                if u > u1:
                    u1 = u
            else:
                if u < u1:
                    return None
                if u < u2:
                    u2 = u

    nx0, ny0 = x0 + u1 * dx, y0 + u1 * dy
    nx1, ny1 = x0 + u2 * dx, y0 + u2 * dy
    if drop_degenerate and _is_degenerate(nx0, ny0, nx1, ny1, xmin, ymin, xmax, ymax, orig_len2):
        return None
    return (nx0, ny0, nx1, ny1)


# ----------------------------------------------------------------------------
# Módulo 7: recorte de polígonos
# ----------------------------------------------------------------------------
def rect_polygon(xmin, ymin, xmax, ymax):
    """Janela retangular como polígono em sentido anti-horário."""
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


def polygon_signed_area(poly):
    """Área com sinal (fórmula do cadarço): > 0 anti-horário, < 0 horário."""
    n = len(poly)
    s = 0.0
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return 0.5 * s


def is_convex(poly, tol=1e-12):
    """True se o polígono é convexo (todos os giros não nulos têm o mesmo sinal)."""
    n = len(poly)
    sign = 0
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        cx, cy = poly[(i + 2) % n]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if abs(cross) <= tol:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def _prepare_window(clip_window):
    """Valida a janela (>= 3 vértices, área não nula, convexa) e a reorienta para anti-horário."""
    window = [tuple(map(float, v)) for v in clip_window]
    if len(window) < 3:
        raise ValueError("A janela de recorte precisa de pelo menos 3 vértices.")
    area = polygon_signed_area(window)
    if abs(area) < 1e-15:
        raise ValueError("A janela de recorte é degenerada (área nula).")
    if not is_convex(window):
        raise ValueError("Sutherland-Hodgman exige janela de recorte CONVEXA.")
    if area < 0:
        window.reverse()   # sentido horário -> anti-horário
    return window


def sutherland_hodgman_clip(polygon, clip_window):
    """
    Sutherland-Hodgman: recorta 'polygon' (qualquer, inclusive côncavo) contra
    uma janela CONVEXA. A janela é validada e, se estiver em sentido horário,
    é reorientada automaticamente para anti-horário.

    Limitação clássica: ao recortar polígonos côncavos cujo resultado correto
    seria composto por mais de uma região, o algoritmo devolve UMA única lista
    de vértices, ligando as regiões por arestas sobrepostas ao longo da borda da
    janela (ver find_overlapping_edges). Para esses casos, use Weiler-Atherton.
    """
    window = _prepare_window(clip_window)

    def inside(p, a, b):
        # Produto vetorial (b - a) x (p - a) >= 0: p está à ESQUERDA (ou sobre) da
        # aresta a->b. Com a janela anti-horária, "à esquerda" = lado interno.
        return (b[0] - a[0]) * (p[1] - a[1]) >= (b[1] - a[1]) * (p[0] - a[0])

    def intersection(cp1, cp2, s, e):
        # Interseção da reta suporte da aresta da janela (cp1, cp2) com a aresta
        # do polígono (s, e), pela forma determinante de duas retas.
        dc = (cp1[0] - cp2[0], cp1[1] - cp2[1])
        dp = (s[0] - e[0], s[1] - e[1])
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = dc[0] * dp[1] - dc[1] * dp[0]
        if abs(n3) < 1e-12:
            return s
        return ((n1 * dp[0] - n2 * dc[0]) / n3, (n1 * dp[1] - n2 * dc[1]) / n3)

    # Recorta o polígono sucessivamente contra cada aresta (s -> cp) da janela.
    # Para cada aresta (e -> coord) do polígono há 4 casos:
    #   dentro -> dentro : emite coord
    #   fora   -> dentro : emite a interseção e coord
    #   dentro -> fora   : emite só a interseção
    #   fora   -> fora   : não emite nada
    output_list = [tuple(map(float, v)) for v in polygon]
    s = window[-1]

    for cp in window:
        input_list = output_list
        output_list = []
        if not input_list:
            break
        e = input_list[-1]
        for coord in input_list:
            if inside(coord, s, cp):
                if not inside(e, s, cp):
                    output_list.append(intersection(s, cp, e, coord))
                output_list.append(coord)
            elif inside(e, s, cp):
                output_list.append(intersection(s, cp, e, coord))
            e = coord
        s = cp
    return output_list


def find_overlapping_edges(poly, tol=1e-9):
    """
    Pares de arestas NÃO adjacentes de 'poly' que são colineares e se sobrepõem
    com comprimento > 0. Aparecem no resultado de Sutherland-Hodgman quando o
    recorte correto seria composto por várias regiões desconexas.
    Retorna lista de ((i, j), comprimento_da_sobreposição).
    """
    n = len(poly)
    found = []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ux, uy = b[0] - a[0], b[1] - a[1]
        lu = (ux * ux + uy * uy) ** 0.5
        if lu < tol:
            continue
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # arestas adjacentes (a última fecha o polígono no vértice 0)
            c, d = poly[j], poly[(j + 1) % n]
            if abs(ux * (c[1] - a[1]) - uy * (c[0] - a[0])) / lu > tol:
                continue
            if abs(ux * (d[1] - a[1]) - uy * (d[0] - a[0])) / lu > tol:
                continue
            tc = ((c[0] - a[0]) * ux + (c[1] - a[1]) * uy) / (lu * lu)
            td = ((d[0] - a[0]) * ux + (d[1] - a[1]) * uy) / (lu * lu)
            lo, hi = max(0.0, min(tc, td)), min(1.0, max(tc, td))
            if (hi - lo) * lu > tol:
                found.append(((i, j), (hi - lo) * lu))
    return found


if __name__ == "__main__":
    print("Testando o módulo de recorte (Cohen-Sutherland, Liang-Barsky e Sutherland-Hodgman)...")
    st_cs, st_lb = ClipStats(), ClipStats()
    cs_res = cohen_sutherland_clip(-2, -2, 12, 12, 0, 0, 10, 10, st_cs)
    lb_res = liang_barsky_clip(-2, -2, 12, 12, 0, 0, 10, 10, st_lb)
    print(f"  -> Cohen-Sutherland : {cs_res}  {st_cs}")
    print(f"  -> Liang-Barsky     : {lb_res}  {st_lb}")
    print(f"  -> Segmento que só toca o canto: CS={cohen_sutherland_clip(-1, 1, 1, -1, 0, 0, 10, 10)} "
          f"LB={liang_barsky_clip(-1, 1, 1, -1, 0, 0, 10, 10)}")

    poly_test = [(-2, 5), (5, -2), (12, 5), (5, 12)]
    win_test = [(0, 0), (10, 0), (10, 10), (0, 10)]
    sh_res = sutherland_hodgman_clip(poly_test, win_test)
    print(f"  -> Sutherland-Hodgman ({len(sh_res)} vértices após recorte): {sh_res}")
    print(f"  -> Janela horária reorientada: {len(sutherland_hodgman_clip(poly_test, win_test[::-1]))} vértices")