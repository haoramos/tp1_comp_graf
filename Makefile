# TP1 - BCC327 Computação Gráfica (UFOP)
# Uso: make install  (uma vez)  e depois  make run-3d | run-2d | plots | screenshots
#
# PYTHON pode ser trocado na linha de comando, p.ex.:  make install PYTHON=python3.12
# (vtk 9.3.0 e numpy 1.26.4 só têm pacotes prontos até o Python 3.12)

PYTHON ?= python3

# O venv guarda o interpretador em venv/bin (Linux/macOS) ou venv/Scripts (Windows).
ifeq ($(OS),Windows_NT)
    VENV_PY := venv/Scripts/python.exe
else
    VENV_PY := venv/bin/python3
endif

.PHONY: help install run-3d run-2d plots screenshots figures clean

help:
	@echo "========================================================"
	@echo " COMANDOS DISPONÍVEIS PARA O TP1 DE COMPUTAÇÃO GRÁFICA "
	@echo "========================================================"
	@echo " make install     - Cria o venv e instala as dependências"
	@echo " make run-3d      - Visualizador CCO interativo (modelo 3D)"
	@echo " make run-2d      - Visualizador CCO interativo (modelo 2D)"
	@echo " make plots       - Figuras didáticas dos Módulos 3 a 7"
	@echo " make screenshots - Capturas do visualizador (Módulos 1, 2 e 5)"
	@echo " make figures     - plots + screenshots"
	@echo " make clean       - Remove venv e caches (__pycache__)"
	@echo "========================================================"

install:
	$(PYTHON) -m venv venv
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt

run-3d:
	$(VENV_PY) main.py --mode viewer --model 3d

run-2d:
	$(VENV_PY) main.py --mode viewer --model 2d

plots:
	$(VENV_PY) main.py --mode plots

screenshots:
	$(VENV_PY) main.py --mode screenshots

figures: plots screenshots

clean:
	rm -rf venv __pycache__ src/__pycache__ .pytest_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@echo "[Limpeza] Venv e caches removidos com sucesso!"
