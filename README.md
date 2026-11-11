# TP1 – Visualização Geométrica, Rasterização e Recorte de Árvores Arteriais CCO

**BCC327 – Computação Gráfica · UFOP · 2026.2** · Prof. Dr. Rafael Alves Bonfim de Queiroz

Aplicação que lê, representa, transforma, rasteriza e recorta árvores arteriais geradas pelo método
*Constrained Constructive Optimization* (CCO), a partir dos modelos 2D e 3D fornecidos em formato VTK.

## 1. Integrantes

- Heitor Augusto de Oliveira Ramos
- Rafael Stylita Duarte

## 2. Tecnologia gráfica

- **Linguagem:** Python 3.12
- **Visualizador interativo (Módulos 1, 2 e 5):** PyVista 0.43.3 sobre VTK 9.3.0 (OpenGL)
- **Módulos didáticos (3 a 7):** algoritmos próprios em Python/NumPy; o Matplotlib é usado **apenas para desenhar** as figuras

## 3. Dependências e versões

| Pacote | Versão | Uso |
|---|---|---|
| Python | 3.12 | interpretador (o `vtk 9.3.0` e o `numpy 1.26.4` só têm pacotes prontos até o 3.12) |
| NumPy | 1.26.4 | matrizes homogêneas 4×4, vetores, amostragem de cobertura |
| PyVista | 0.43.3 | janela, câmera, projeções, tubos, mapa de cores, MSAA/SSAA |
| VTK | 9.3.0 | backend de renderização do PyVista |
| Matplotlib | 3.8.3 | desenho das figuras dos Módulos 3 a 7 e montagem das capturas |

As versões estão fixadas em [`requirements.txt`](requirements.txt).

Requisitos do sistema: `make` e o módulo `venv` do Python (Ubuntu/WSL: `sudo apt install python3.12-venv make`).
O visualizador interativo precisa de um ambiente gráfico com OpenGL (no WSL, o WSLg).

## 4. Instalação

```bash
make install
```

Cria o ambiente virtual `venv/` na raiz e instala as dependências do `requirements.txt`.
Se o `python3` do sistema não for o 3.12, indique o interpretador:

```bash
make install PYTHON=python3.12
```

O Makefile detecta Windows e usa `venv/Scripts/python.exe` no lugar de `venv/bin/python3`.

## 5. Execução

| Comando | O que faz |
|---|---|
| `make run-3d` | visualizador CCO interativo com o modelo 3D |
| `make run-2d` | visualizador CCO interativo com o modelo 2D |
| `make plots` | gera as figuras dos Módulos 3 a 7 (não abre janela) |
| `make screenshots` | gera as capturas do visualizador para os Módulos 1, 2 e 5 (renderização *off-screen*) |
| `make figures` | `plots` + `screenshots` |
| `make clean` | remove `venv/` e todos os `__pycache__/` |

Sem `make` (com o venv ativado):

```bash
python main.py --mode viewer --model 3d    # ou --model 2d
python main.py --mode plots
python main.py --mode screenshots
```

As figuras são gravadas em `report/figures/` e copiadas para `presentation/figures/`.

## 6. Arquivos VTK

Esperados em `data/`, com estes nomes:

```
data/tree2D_Nterm064.vtk    # modelo 2D (128 pontos, 127 segmentos)
data/tree3D_Nterm064.vtk    # modelo 3D (128 pontos, 127 segmentos)
```

São os arquivos de `dados_modelos2D.zip` e `dados_modelos3D.zip` fornecidos pelo professor (VTK legado ASCII, `DATASET POLYDATA`,
células `LINES` e escalar `raio` por célula). Para usar outro modelo, substitua o arquivo ou ajuste o nome em `main.py`.

## 7. Comandos do visualizador (`make run-3d` / `make run-2d`)

| Tecla / mouse | Ação |
|---|---|
| `X` `Y` `Z` | escolhe o eixo ativo (o modelo 2D começa em Z) |
| `T` / `G` | translação + / − no eixo ativo (5% da diagonal da caixa envolvente) |
| `K` / `J` | rotação de +15° / −15° em torno do eixo ativo, com pivô no centroide |
| `+` ou `=` / `-` | escala uniforme ×1,1 / ÷1,1 com o centroide fixo |
| `0` | reinicia as transformações |
| `O` / `V` | projeção ortográfica / perspectiva |
| `B` `N` `U` `I` | vista frontal (plano XY), lateral (YZ), superior (XZ), isométrica |
| `A` | alterna antialiasing: desligado → MSAA → SSAA |
| `C` | espessura proporcional ao raio ↔ espessura constante (a cor continua codificando o raio) |
| arrastar (botão esquerdo) | gira a câmera |
| botão direito ou *scroll* | zoom |
| `Shift` + arrastar | desloca a câmera (*pan*) |
| `Q` | fecha a janela |

A cada transformação o terminal mostra a matriz 4×4 acumulada; a cada mudança de vista mostra posição, foco e vetor *up* da câmera.
O canto superior esquerdo da janela exibe nº de pontos e segmentos, raio mín./máx., projeção, AA, espessura e última operação.

## 8. Roteiro mínimo de teste

1. `make install`.
2. `make run-3d`
   - no terminal: 128 pontos, 127 segmentos, raio mín. 0,2052 e máx. 1,0178;
   - `T`, `K`, `+`: a árvore se move, gira e cresce; a matriz 4×4 aparece no terminal; `0` volta ao original;
   - `O` e `V`: compare as projeções; `B`, `N`, `U`, `I`: quatro posições do observador;
   - `A`: compare as bordas dos tubos sem AA, com MSAA e com SSAA; `C`: espessura proporcional × constante.
3. `make run-2d`: `B` mostra a árvore planar; `K` gira em Z; `I` mostra o plano em perspectiva.
4. `make plots`: gera `mod3` a `mod7` e imprime as métricas dos Módulos 5 (níveis de cinza e erro), 6 (iterações, divisões,
   tempo, 0 divergências entre os algoritmos) e 7 (vértices e arestas sobrepostas).
5. `make screenshots`: gera `mod1_primitivas`, `mod2_transformacoes`, `mod2_projecoes`, `mod2_vistas` e `mod5_viewer_aa`.
6. Testes rápidos de cada módulo: `python -m src.vtk_loader`, `python -m src.rasterization`, `python -m src.clipping`.

## 9. Módulos do enunciado × código

| Módulo | Arquivos | Figuras |
|---|---|---|
| 1 – Leitura, primitivas e atributos | `src/vtk_loader.py`, `src/viewer_3d.py` | `mod1_primitivas.png` |
| 2 – Transformações e observador | `src/viewer_3d.py` | `mod2_transformacoes.png`, `mod2_projecoes.png`, `mod2_vistas.png` |
| 3 – Rasterização de linhas (Bresenham) | `src/rasterization.py` | `mod3_bresenham.png` |
| 4 – Rasterização de polígonos (Scanline) | `src/rasterization.py` | `mod4_scanline.png` |
| 5 – Aliasing e antialiasing | `src/didactic_plots.py` (SSAA próprio), tecla `A` | `mod5_antialiasing.png`, `mod5_viewer_aa.png` |
| 6 – Recorte de segmentos | `src/clipping.py` | `mod6_line_clipping.png`, `mod6_comparacao_algoritmos.png` |
| 7 – Recorte de polígonos | `src/clipping.py` | `mod7_polygon_clipping.png` |

## 10. Implementado pelo grupo × fornecido por bibliotecas

**Implementado pelo grupo (Python/NumPy, sem funções gráficas prontas):**

- leitor de VTK ASCII (`POINTS`, `LINES`, `CELL_DATA`/`POINT_DATA`, `SCALARS`), com expansão de polilinhas;
- matrizes homogêneas 4×4 de translação, rotação (X, Y, Z) e escala, composição em torno de um pivô e aplicação aos pontos;
- geometria “um segmento = um tubo”, com o raio de cada vaso constante ao longo dele;
- Bresenham (8 octantes, aritmética inteira);
- Scanline Fill com Tabela de Arestas (ET) e Tabela de Arestas Ativas (AET), regra par-ímpar;
- supersampling por cobertura (1, 2×2, 4×4 amostras por pixel) e métricas de erro;
- Cohen–Sutherland (obrigatório) e Liang–Barsky, com contadores de custo;
- Sutherland–Hodgman, com validação/reorientação da janela e detecção das arestas sobrepostas em polígonos côncavos.

**Fornecido por bibliotecas:**

- **PyVista/VTK:** janela e interação de mouse, câmera e matrizes de visualização/projeção, projeção ortográfica e perspectiva,
  geração das malhas dos tubos (`vtkTubeFilter`), mapa de cores `coolwarm`, iluminação/sombreamento, MSAA e SSAA do visualizador.
- **Matplotlib:** apenas exibição das grades de pixels e montagem das figuras.

Cyrus–Beck e Weiler–Atherton não foram implementados; são discutidos no relatório.

## 11. Estrutura de diretórios

```
TP1_grafica/
├── main.py                  # ponto de entrada (--mode viewer | plots | screenshots)
├── Makefile
├── requirements.txt
├── README.md
├── src/
│   ├── vtk_loader.py        # Módulo 1: leitura do VTK
│   ├── viewer_3d.py         # Módulos 1, 2 e 5: visualizador PyVista
│   ├── rasterization.py     # Módulos 3 e 4: Bresenham e Scanline
│   ├── clipping.py          # Módulos 6 e 7: Cohen-Sutherland, Liang-Barsky, Sutherland-Hodgman
│   ├── didactic_plots.py    # figuras dos Módulos 3 a 7
│   └── viewer_figures.py    # capturas off-screen do visualizador
├── data/                    # arquivos VTK
├── report/
│   ├── relatorio.tex
│   ├── relatorio.pdf
│   └── figures/
└── presentation/
    ├── apresentacao.tex
    ├── apresentacao.pdf
    └── figures/
```

Para compilar os documentos: `pdflatex relatorio.tex` (duas vezes) dentro de `report/` e `pdflatex apresentacao.tex` (duas vezes)
dentro de `presentation/`.

## 12. Entrega

Conforme o enunciado, o `.zip` **não** deve conter `venv/`, `__pycache__/` nem arquivos auxiliares do LaTeX
(`*.aux`, `*.log`, `*.out`, `*.toc`, `*.nav`, `*.snm`). Rode `make clean` antes de compactar.
Todos esses arquivos são recriados por `make install` e pela compilação dos `.tex`.
