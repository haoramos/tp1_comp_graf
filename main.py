"""
TP1 - BCC327 Computação Gráfica (UFOP): ponto de entrada da aplicação.

Modos (--mode):
  viewer      visualizador CCO interativo (Módulos 1, 2 e 5) - PyVista/VTK
  plots       figuras didáticas dos Módulos 3 a 7 (Bresenham, Scanline, SSAA,
              recorte de segmentos e de polígonos) - só NumPy/Matplotlib
  screenshots capturas off-screen do visualizador para o relatório

Exemplos:
  python main.py --mode viewer --model 3d
  python main.py --mode plots
"""
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="TP1 - Computação Gráfica (UFOP)")
    parser.add_argument(
        "--mode", type=str, default="viewer", choices=["viewer", "plots", "screenshots"],
        help="viewer: visualizador CCO interativo (PyVista) | "
             "plots: gera as figuras dos módulos 3 a 7 (Matplotlib, sem janela) | "
             "screenshots: gera as capturas do visualizador (módulos 1, 2 e 5) sem abrir janela",
    )
    parser.add_argument(
        "--model", type=str, default="3d", choices=["2d", "3d"],
        help="Modelo CCO usado no modo viewer: 2d ou 3d",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Importações feitas só quando necessárias: o modo 'plots' não exige PyVista/VTK
    # nem qualquer biblioteca de janela.
    if args.mode == "plots":
        import matplotlib
        matplotlib.use("Agg")          # gera PNGs sem precisar de janela/display
        from src.didactic_plots import generate_all_didactic_figures
        generate_all_didactic_figures()
        return

    if args.mode == "screenshots":
        import matplotlib
        matplotlib.use("Agg")
        from src.viewer_figures import generate_viewer_figures
        generate_viewer_figures()
        return

    filename = "tree2D_Nterm064.vtk" if args.model == "2d" else "tree3D_Nterm064.vtk"
    file_path = os.path.join(base_dir, "data", filename)
    if not os.path.exists(file_path):
        print(f"[Erro] Arquivo não encontrado: {file_path}")
        return

    from src.vtk_loader import load_vtk_cco
    model = load_vtk_cco(file_path)
    model.print_summary()

    from src.viewer_3d import show_viewer
    show_viewer(model)


if __name__ == "__main__":
    main()