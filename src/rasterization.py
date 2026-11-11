"""
Módulos 3 e 4 do TP1: rasterização de segmentos (Bresenham) e de polígonos
(Scanline Fill com Tabela de Arestas e Arestas Ativas). Implementação própria.

Convenção de coordenadas: o pixel (i, j) tem o seu CENTRO em (i, j); os vértices
são dados nessas mesmas coordenadas.
"""
import math

_EPS = 1e-9


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list:
    """
    Algoritmo de Bresenham (versão com erro acumulado, válida nos 8 octantes).
    Retorna a lista de pixels (x, y), de (x0, y0) até (x1, y1), com
    max(|dx|, |dy|) + 1 pixels e conectividade de 8 vizinhos.

    'err' guarda (a menos de um fator 2) a diferença entre a distância da reta
    ideal ao próximo pixel em x e ao próximo pixel em y. A cada passo, o teste
    com e2 = 2*err decide, usando apenas somas e comparações INTEIRAS, se o
    pixel seguinte avança em x, em y ou em ambos (diagonal). Não há divisões
    nem ponto flutuante, ao contrário do DDA.
    """
    pixels = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1          # sentido do passo em x (trata os octantes "para trás")
    sy = 1 if y0 < y1 else -1          # sentido do passo em y
    err = dx - dy                      # erro inicial

    x, y = x0, y0
    while True:
        pixels.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:                   # reta ainda "à frente" em x: avança em x
            err -= dy
            x += sx
        if e2 < dx:                    # reta já "acima" do pixel: avança em y
            err += dx
            y += sy
    return pixels


def _scanline_pixels(vertices, closed_x):
    """
    Scanline Fill com Tabela de Arestas (ET) e Tabela de Arestas Ativas (AET).

    Regra de arestas (semi-aberta em y): uma aresta é ativa nas linhas
    ymin <= y < ymax, então um vértice compartilhado por duas arestas nunca é
    contado duas vezes e arestas horizontais são ignoradas. A coordenada x da
    aresta é atualizada de forma incremental: x += dx/dy a cada nova linha.

    closed_x=True : preenche de ceil(x_esq) até floor(x_dir)  (inclui as bordas em x)
    closed_x=False: preenche de ceil(x_esq) até ceil(x_dir)-1 (bordas esquerda-fechada, direita-aberta)
    """
    n = len(vertices)
    edge_table = {}                      # y inicial -> [[y_final(exclusivo), x, dx/dy], ...]
    y_first, y_last = None, None
    for i in range(n):
        xa, ya = vertices[i]
        xb, yb = vertices[(i + 1) % n]
        if ya == yb:
            continue                     # aresta horizontal
        if ya > yb:
            xa, ya, xb, yb = xb, yb, xa, ya
        slope = (xb - xa) / (yb - ya)    # dx/dy
        ys = math.ceil(ya - _EPS)        # primeira linha inteira coberta
        ye = math.ceil(yb - _EPS)        # primeira linha inteira NÃO coberta
        if ys >= ye:
            continue                     # a aresta não cruza nenhuma linha inteira
        x_start = xa + (ys - ya) * slope
        edge_table.setdefault(ys, []).append([ye, x_start, slope])
        y_first = ys if y_first is None else min(y_first, ys)
        y_last = ye - 1 if y_last is None else max(y_last, ye - 1)

    pixels = set()
    if y_first is None:
        return pixels

    active = []                          # AET
    for y in range(y_first, y_last + 1):
        active = [e for e in active if e[0] > y]        # remove arestas que terminaram
        active.extend(edge_table.get(y, []))            # insere arestas que começam em y
        active.sort(key=lambda e: e[1])                 # ordena por x
        for k in range(0, len(active) - 1, 2):          # preenche em pares (paridade)
            xl, xr = active[k][1], active[k + 1][1]
            x_from = math.ceil(xl - _EPS)
            x_to = math.floor(xr + _EPS) if closed_x else math.ceil(xr - _EPS) - 1
            for x in range(x_from, x_to + 1):
                pixels.add((x, y))
        for e in active:                                # atualização incremental
            e[1] += e[2]
    return pixels


def scanline_fill_polygon(polygon_vertices: list, include_boundary: bool = True) -> list:
    """
    Preenche um polígono (convexo ou côncavo, regra par-ímpar) e devolve os
    pixels (x, y), sem duplicatas, ordenados por y e depois por x.

    include_boundary=True  : polígono FECHADO. Preenche todo pixel cujo centro está
                             dentro OU sobre o contorno (inclui as linhas inferior e
                             superior). Obtido como a união de duas passadas da
                             regra semi-aberta: [ymin, ymax) e (ymin, ymax].
    include_boundary=False : convenção semi-aberta usada para que polígonos
                             vizinhos não se sobreponham (esquerda/baixo inclusos,
                             direita/cima excluídos).
    """
    if not polygon_vertices or len(polygon_vertices) < 3:
        return []
    verts = [(float(x), float(y)) for x, y in polygon_vertices]

    if include_boundary:
        lower = _scanline_pixels(verts, closed_x=True)                     # [ymin, ymax)
        mirrored = [(x, -y) for x, y in verts]                              # espelha em y -> (ymin, ymax]
        upper = {(x, -y) for x, y in _scanline_pixels(mirrored, closed_x=True)}
        pixels = lower | upper
    else:
        pixels = _scanline_pixels(verts, closed_x=False)
    return sorted(pixels, key=lambda p: (p[1], p[0]))


if __name__ == "__main__":
    print("Testando o módulo de rasterização...")
    line_pixels = bresenham_line(0, 0, 5, 3)
    print(f"Pixels gerados por Bresenham (0,0) até (5,3): {line_pixels}")

    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    filled = scanline_fill_polygon(square)
    ys = sorted({y for _, y in filled})
    print(f"Quadrado 0..10 (fechado): {len(filled)} pixels, linhas y={ys[0]}..{ys[-1]} (esperado 121, y=0..10)")
    half = scanline_fill_polygon(square, include_boundary=False)
    print(f"Quadrado 0..10 (semi-aberto): {len(half)} pixels (esperado 100)")