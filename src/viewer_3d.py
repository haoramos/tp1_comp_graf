"""
Visualizador CCO (Módulos 1 e 2 do TP1) com PyVista.

O que é implementado pelo grupo (NumPy):
  - matrizes homogêneas 4x4 de translação, rotação (X/Y/Z) e escala;
  - aplicação das matrizes aos pontos (coordenadas homogêneas);
  - matriz acumulada, impressa no terminal a cada operação;
  - construção da geometria "um segmento = um tubo" com raio próprio.

O que é fornecido pela biblioteca (PyVista/VTK):
  - geração dos tubos (vtkTubeFilter), renderização, câmera, projeções,
    mapa de cores e antialiasing (MSAA/SSAA).
"""
import numpy as np
import pyvista as pv

# Fator arbitrário que converte o atributo "raio" (unidade do arquivo) em
# unidades de mundo para desenhar os tubos. Só afeta a aparência: a espessura
# continua proporcional ao raio de cada segmento.
RADIUS_TO_WORLD = 0.0012

TRANSLATION_STEP = 0.05   # fração da diagonal da caixa envolvente do modelo
ROTATION_STEP_DEG = 15.0
SCALE_STEP = 1.1


# ----------------------------------------------------------------------------
# Matrizes homogêneas 4x4 (implementação própria)
# ----------------------------------------------------------------------------
def translation_matrix(dx, dy, dz):
    return np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1],
    ], dtype=np.float64)


def scaling_matrix(sx, sy, sz):
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)


def rotation_matrix(axis, angle_deg):
    """Rotação de angle_deg graus em torno do eixo 'x', 'y' ou 'z' (regra da mão direita)."""
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    if axis == "x":
        return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=np.float64)
    if axis == "z":
        return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    raise ValueError(f"Eixo inválido: {axis}")


def about_pivot(matrix, pivot):
    """Aplica 'matrix' em torno de 'pivot': T(p) · M · T(-p)."""
    return translation_matrix(*pivot) @ matrix @ translation_matrix(*(-np.asarray(pivot)))


def apply_matrix(points, matrix):
    """Aplica uma matriz 4x4 a uma lista de pontos (N,3) usando coordenadas homogêneas."""
    homogeneous = np.hstack((points, np.ones((len(points), 1))))
    return (homogeneous @ matrix.T)[:, :3]


# ----------------------------------------------------------------------------
# Geometria: um tubo por segmento, com o raio do próprio segmento
# ----------------------------------------------------------------------------
def build_segment_mesh(points, segments, radii):
    """
    Cria um PolyData em que cada segmento possui seus 2 pontos próprios
    (pontos duplicados nas bifurcações). Assim o raio de cada vaso é constante
    ao longo dele, em vez de ser interpolado com o raio dos vasos vizinhos.
    """
    n = len(segments)
    seg_points = points[segments].reshape(-1, 3)              # (2n, 3)
    ids = np.arange(2 * n).reshape(n, 2)
    lines = np.column_stack((np.full(n, 2), ids)).ravel()
    mesh = pv.PolyData(seg_points, lines=lines)
    mesh.point_data["Raio"] = np.repeat(radii, 2)             # mesmo raio nos 2 pontos
    return mesh


class CCOViewer:
    """
    Visualizador interativo. A árvore é transformada pelo GRUPO no espaço do
    mundo (self.points = M · pontos originais); a câmera do PyVista define o
    sistema de coordenadas da câmera (posição, foco, up) e a projeção.
    """
    AXIS_VECTORS = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
    AA_MODES = [None, "msaa", "ssaa"]
    AA_NAMES = {None: "desligado", "msaa": "MSAA", "ssaa": "SSAA (supersampling)"}

    def __init__(self, model, off_screen=False, window_size=None):
        """
        model       : CCOTreeModel lido por vtk_loader.load_vtk_cco.
        off_screen  : True renderiza sem abrir janela (usado para gerar as
                      capturas do relatório em src/viewer_figures.py).
        window_size : (largura, altura) em pixels da janela/imagem.
        """
        self.model = model
        self.points0 = model.points.copy()
        self.points = model.points.copy()
        self.matrix = np.eye(4, dtype=np.float64)           # matriz acumulada
        self.last_op = "nenhuma"

        # Eixo ativo para translação/rotação (modelo 2D é planar em XY: gira em Z)
        self.axis = "y" if model.is_3d else "z"
        self.proportional = True                            # espessura proporcional ao raio
        self.orthographic = False
        self.aa_idx = 1                                     # MSAA (padrão do PyVista)

        self.r_min = model.min_radius
        self.r_max = model.max_radius
        self.ratio = self.r_max / self.r_min
        self.base_radius = self.r_min * RADIUS_TO_WORLD
        self.constant_radius = self.base_radius * self.ratio * 0.35
        self.step_translation = TRANSLATION_STEP * float(np.linalg.norm(np.ptp(self.points0, axis=0)))

        self.plotter = pv.Plotter(
            title=f"TP1 - Computação Gráfica: {model.filename}",
            off_screen=off_screen,
            window_size=window_size,
        )
        self.show_hud = True                                # False nas capturas do relatório
        self.show_scalar_bar = True                         # idem (barra desenhada pelo Matplotlib)

    # ------------------------------------------------------------------ desenho
    def _tubes(self):
        mesh = build_segment_mesh(self.points, self.model.segments, self.model.radii)
        if self.proportional:
            # Raio do tubo = base_radius * (raio / r_min): proporcional ao atributo.
            return mesh.tube(radius=self.base_radius, scalars="Raio", radius_factor=self.ratio)
        return mesh.tube(radius=self.constant_radius)

    def _draw(self):
        # 'name' faz o PyVista substituir o ator anterior; reset_camera=False
        # mantém a posição da câmera ajustada pelo usuário.
        self.plotter.add_mesh(
            self._tubes(),
            scalars="Raio",
            cmap="coolwarm",
            clim=[self.r_min, self.r_max],
            smooth_shading=True,
            name="arvore",
            reset_camera=False,
            show_scalar_bar=self.show_scalar_bar,
            scalar_bar_args={"title": "Raio"},
        )
        self._update_hud()

    def _update_hud(self):
        """Texto no canto da janela com o resumo do Módulo 1 e o estado atual."""
        if not self.show_hud:
            self.plotter.render()
            return
        m = self.model
        text = (
            f"{m.filename} ({'3D' if m.is_3d else '2D'}) | pontos: {m.num_points} | "
            f"segmentos: {m.num_segments} | raio: [{self.r_min:.4f}, {self.r_max:.4f}]\n"
            f"Projeção: {'ortográfica' if self.orthographic else 'perspectiva'} | "
            f"AA: {self.AA_NAMES[self.AA_MODES[self.aa_idx]]} | "
            f"Espessura: {'proporcional ao raio' if self.proportional else 'constante'} | "
            f"Eixo ativo: {self.axis.upper()} | Última operação: {self.last_op}"
        )
        self.plotter.add_text(text, position="upper_left", font_size=9, name="hud")
        self.plotter.render()

    # ------------------------------------------------------- transformações
    def _pivot(self):
        # Centroide dos pontos: acompanha a árvore em rotações/translações/escalas,
        # então uma operação e sua inversa se cancelam exatamente.
        return self.points.mean(axis=0)

    def _apply(self, op_matrix, label):
        step = about_pivot(op_matrix, self._pivot())
        self.points = apply_matrix(self.points, step)
        self.matrix = step @ self.matrix
        self.last_op = label
        print(f"\n[Transformação: {label}]")
        print("Matriz acumulada 4x4 (mundo inicial -> mundo atual):")
        print(np.array2string(self.matrix, precision=4, suppress_small=True))
        self._draw()

    def set_axis(self, axis):
        self.axis = axis
        self._update_hud()

    def translate(self, sign):
        vec = np.array(self.AXIS_VECTORS[self.axis]) * self.step_translation * sign
        self._apply(translation_matrix(*vec), f"translação {'+' if sign > 0 else '-'}{self.axis.upper()}")

    def rotate(self, sign):
        angle = ROTATION_STEP_DEG * sign
        self._apply(rotation_matrix(self.axis, angle), f"rotação {angle:+.0f}° em {self.axis.upper()}")

    def scale(self, factor):
        self._apply(scaling_matrix(factor, factor, factor), f"escala x{factor:.3f}")

    def reset_transformations(self):
        self.points = self.points0.copy()
        self.matrix = np.eye(4, dtype=np.float64)
        self.last_op = "reset"
        print("\n[Transformações reiniciadas]")
        self._draw()

    # ------------------------------------------------------ câmera / projeção
    def set_orthographic(self):
        self.plotter.enable_parallel_projection()
        self.orthographic = True
        print("[Câmera] Projeção ORTOGRÁFICA")
        self._update_hud()

    def set_perspective(self):
        self.plotter.disable_parallel_projection()
        self.orthographic = False
        print("[Câmera] Projeção PERSPECTIVA")
        self._update_hud()

    def _print_camera(self, name):
        pos, focal, up = self.plotter.camera_position
        print(f"[Câmera] {name} | posição={np.round(pos, 4)} foco={np.round(focal, 4)} up={np.round(up, 4)}")

    def view_front(self):
        self.plotter.view_xy()
        self._print_camera("Vista frontal (plano XY, olhando ao longo de Z)")
        self._update_hud()

    def view_side(self):
        self.plotter.view_yz()
        self._print_camera("Vista lateral (plano YZ, olhando ao longo de X)")
        self._update_hud()

    def view_top(self):
        self.plotter.view_xz()
        self._print_camera("Vista superior (plano XZ, olhando ao longo de Y)")
        self._update_hud()

    def view_iso(self):
        self.plotter.view_isometric()
        self._print_camera("Vista isométrica")
        self._update_hud()

    # -------------------------------------------------- antialiasing / estilo
    def cycle_antialiasing(self):
        """Alterna: desligado -> MSAA -> SSAA (supersampling) -> desligado..."""
        self.aa_idx = (self.aa_idx + 1) % len(self.AA_MODES)
        mode = self.AA_MODES[self.aa_idx]
        if mode is None:
            self.plotter.disable_anti_aliasing()
        else:
            self.plotter.enable_anti_aliasing(mode)
        print(f"[Antialiasing] {self.AA_NAMES[mode]}")
        self._update_hud()

    def toggle_thickness(self):
        self.proportional = not self.proportional
        print(f"[Espessura] {'proporcional ao raio' if self.proportional else 'constante (só a cor codifica o raio)'}")
        self._draw()

    # ------------------------------------------------------------------ execução
    def _bind_keys(self):
        bindings = {
            "x": lambda: self.set_axis("x"),
            "y": lambda: self.set_axis("y"),
            "z": lambda: self.set_axis("z"),
            "t": lambda: self.translate(+1),
            "g": lambda: self.translate(-1),
            "k": lambda: self.rotate(+1),
            "j": lambda: self.rotate(-1),
            "plus": lambda: self.scale(SCALE_STEP),
            "equal": lambda: self.scale(SCALE_STEP),
            "KP_Add": lambda: self.scale(SCALE_STEP),
            "minus": lambda: self.scale(1.0 / SCALE_STEP),
            "KP_Subtract": lambda: self.scale(1.0 / SCALE_STEP),
            "0": self.reset_transformations,
            "o": self.set_orthographic,
            "v": self.set_perspective,
            "b": self.view_front,
            "n": self.view_side,
            "u": self.view_top,
            "i": self.view_iso,
            "a": self.cycle_antialiasing,
            "c": self.toggle_thickness,
        }
        for key, callback in bindings.items():
            self.plotter.add_key_event(key, callback)

    @staticmethod
    def _print_controls():
        print("\n" + "=" * 66)
        print(" CONTROLES DO VISUALIZADOR (TP1) ".center(66, "="))
        print(" Eixo ativo ........ [X] [Y] [Z]")
        print(" Translação ........ [T] +eixo   [G] -eixo")
        print(" Rotação (15°) ..... [K] +        [J] -      (em torno do eixo ativo)")
        print(" Escala (x1,1) ..... [+] ou [=] aumenta   [-] diminui")
        print(" Reiniciar ......... [0]")
        print(" Projeção .......... [O] ortográfica   [V] perspectiva")
        print(" Vistas ............ [B] frontal  [N] lateral  [U] superior  [I] isométrica")
        print(" Antialiasing ...... [A] alterna desligado / MSAA / SSAA")
        print(" Espessura ......... [C] proporcional ao raio / constante")
        print(" Mouse ............. arrastar = girar | botão direito/scroll = zoom | Shift+arrastar = mover")
        print("=" * 66 + "\n")

    def run(self):
        self.plotter.enable_anti_aliasing("msaa")
        self._draw()
        self._bind_keys()
        self._print_controls()
        self.plotter.add_axes()
        self.plotter.show()


def show_viewer(model):
    CCOViewer(model).run()


if __name__ == "__main__":
    import os
    from src.vtk_loader import load_vtk_cco

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "data", "tree3D_Nterm064.vtk")
    if os.path.exists(file_path):
        show_viewer(load_vtk_cco(file_path))