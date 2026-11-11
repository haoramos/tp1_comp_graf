"""
Módulo 1 do TP1: leitura manual de arquivos .vtk ASCII (DATASET POLYDATA) com
árvores arteriais CCO. Não usa nenhuma biblioteca de leitura de VTK.
"""
import os
import warnings
import numpy as np
from dataclasses import dataclass

# Nomes aceitos (sem diferenciar maiúsculas/minúsculas) para o atributo de raio.
RADIUS_NAMES = ("raio", "radius", "radii", "r")


@dataclass
class CCOTreeModel:
    """Estrutura de dados para armazenar a árvore arterial CCO carregada de um arquivo VTK."""
    filename: str
    points: np.ndarray      # (N_points, 3) -> coordenadas (x, y, z)
    segments: np.ndarray    # (N_segments, 2) -> índices (id_inicio, id_fim)
    radii: np.ndarray       # (N_segments,) -> raio de cada segmento
    is_3d: bool             # True se possuir variação em Z, False se for planar (2D)
    radius_name: str = "raio"   # nome do escalar do arquivo usado como raio

    @property
    def num_points(self) -> int:
        return self.points.shape[0]

    @property
    def num_segments(self) -> int:
        return self.segments.shape[0]

    @property
    def min_radius(self) -> float:
        return float(np.min(self.radii))

    @property
    def max_radius(self) -> float:
        return float(np.max(self.radii))

    def print_summary(self) -> None:
        """Exibe no console as informações exigidas no Módulo 1."""
        dim_str = "3D" if self.is_3d else "2D"
        print("=" * 60)
        print(f"Modelo CCO Carregado: {self.filename} ({dim_str})")
        print("-" * 60)
        print(f"  -> Número de pontos (vértices) : {self.num_points}")
        print(f"  -> Número de segmentos (vasos) : {self.num_segments}")
        print(f"  -> Atributo de raio usado      : '{self.radius_name}'")
        print(f"  -> Raio mínimo                 : {self.min_radius:.7f}")
        print(f"  -> Raio máximo                 : {self.max_radius:.7f}")
        print(f"  -> Bounding Box X              : [{np.min(self.points[:, 0]):.5f}, {np.max(self.points[:, 0]):.5f}]")
        print(f"  -> Bounding Box Y              : [{np.min(self.points[:, 1]):.5f}, {np.max(self.points[:, 1]):.5f}]")
        print(f"  -> Bounding Box Z              : [{np.min(self.points[:, 2]):.5f}, {np.max(self.points[:, 2]):.5f}]")
        print("=" * 60)


def _parse_vtk_ascii(filepath):
    """
    Lê o arquivo como fluxo de tokens (independe de quebras de linha) e devolve
    (points, cells, scalars), onde:
      points  : array (N, 3)
      cells   : lista de listas de índices (uma por célula de LINES)
      scalars : lista de (nome, seção 'cell'|'point', array 1D)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        tokens = f.read().split()

    points = None
    cells = []
    scalars = []
    section = None            # 'cell' após CELL_DATA, 'point' após POINT_DATA
    section_count = 0
    i, n = 0, len(tokens)

    while i < n:
        kw = tokens[i].upper()

        if kw == "POINTS":
            num = int(tokens[i + 1])
            i += 3                                   # POINTS N tipo
            points = np.array(tokens[i:i + 3 * num], dtype=np.float64).reshape((num, 3))
            i += 3 * num

        elif kw == "LINES":
            num_cells = int(tokens[i + 1])
            i += 3                                   # LINES N tamanho
            for _ in range(num_cells):
                k = int(tokens[i])
                cells.append([int(t) for t in tokens[i + 1:i + 1 + k]])
                i += 1 + k

        elif kw in ("CELL_DATA", "POINT_DATA"):
            section = "cell" if kw == "CELL_DATA" else "point"
            section_count = int(tokens[i + 1])
            i += 2

        elif kw == "SCALARS":
            name = tokens[i + 1]
            i += 3                                   # SCALARS nome tipo
            ncomp = 1
            if i < n and tokens[i].upper() != "LOOKUP_TABLE":
                ncomp = int(tokens[i])               # número de componentes (opcional)
                i += 1
            if i < n and tokens[i].upper() == "LOOKUP_TABLE":
                i += 2                               # LOOKUP_TABLE nome
            count = section_count * ncomp
            values = np.array(tokens[i:i + count], dtype=np.float64)
            i += count
            if ncomp == 1:
                scalars.append((name, section, values))

        else:
            i += 1

    if points is None:
        raise ValueError("Arquivo VTK sem seção POINTS.")
    return points, cells, scalars


def _pick_radius_array(scalars, filepath):
    """Escolhe o escalar de raio: pelo nome, ou o primeiro escalar do arquivo (com aviso)."""
    if not scalars:
        raise ValueError(f"{os.path.basename(filepath)}: nenhum atributo escalar encontrado (esperado 'raio').")
    for name, section, values in scalars:
        if name.lower() in RADIUS_NAMES:
            return name, section, values
    name, section, values = scalars[0]
    warnings.warn(
        f"{os.path.basename(filepath)}: escalar de raio não encontrado entre {RADIUS_NAMES}; "
        f"usando o primeiro escalar do arquivo: '{name}'."
    )
    return name, section, values


def load_vtk_cco(filepath: str) -> CCOTreeModel:
    """
    Lê um arquivo .vtk ASCII (POLYDATA) com uma árvore CCO.

    - Células LINES com mais de 2 vértices (polilinhas) são expandidas em
      segmentos consecutivos; cada segmento herda o atributo da sua célula.
    - O atributo de raio pode se chamar raio/radius/radii/r (ou ser o primeiro
      escalar do arquivo, com aviso). Se for dado por PONTO (POINT_DATA), o raio
      do segmento é o máximo entre os dois extremos.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo VTK não encontrado: {filepath}")

    points, cells, scalars = _parse_vtk_ascii(filepath)
    if not cells:
        raise ValueError(f"{os.path.basename(filepath)}: nenhuma célula LINES encontrada.")

    segments, cell_of_segment = [], []
    for cid, cell in enumerate(cells):
        for a, b in zip(cell[:-1], cell[1:]):
            segments.append((a, b))
            cell_of_segment.append(cid)
    segments_arr = np.array(segments, dtype=np.int32)
    cell_of_segment = np.array(cell_of_segment, dtype=np.int64)

    if segments_arr.min() < 0 or segments_arr.max() >= len(points):
        raise ValueError(f"{os.path.basename(filepath)}: índice de ponto fora do intervalo em LINES.")

    name, section, values = _pick_radius_array(scalars, filepath)
    if section == "cell" and len(values) == len(cells):
        radii_arr = values[cell_of_segment]
    elif section == "point" and len(values) == len(points):
        warnings.warn(f"Atributo '{name}' é por ponto; usando o máximo dos extremos de cada segmento.")
        radii_arr = np.maximum(values[segments_arr[:, 0]], values[segments_arr[:, 1]])
    else:
        raise ValueError(
            f"{os.path.basename(filepath)}: o escalar '{name}' tem {len(values)} valores, "
            f"incompatível com {len(cells)} células e {len(points)} pontos."
        )

    z_ptp = np.ptp(points[:, 2]) if len(points) > 0 else 0.0
    return CCOTreeModel(
        filename=os.path.basename(filepath),
        points=points,
        segments=segments_arr,
        radii=radii_arr.astype(np.float64),
        is_3d=bool(z_ptp > 1e-6),
        radius_name=name,
    )


if __name__ == "__main__":
    # Teste rápido do Módulo 1 (Leitura de dados 2D e 3D)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_files = [
        os.path.join(base_dir, "data", "tree2D_Nterm064.vtk"),
        os.path.join(base_dir, "data", "tree3D_Nterm064.vtk"),
    ]
    for tf in test_files:
        if os.path.exists(tf):
            load_vtk_cco(tf).print_summary()
        else:
            print(f"[AVISO] Coloque o arquivo {os.path.basename(tf)} na pasta 'data/' para testar.")