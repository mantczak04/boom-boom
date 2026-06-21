.PHONY: install test app benchmark

install:
	uv sync --dev

test:
	uv run pytest -q

app:
	uv run streamlit run app.py

benchmark:
	uv run prob-minesweeper benchmark --episodes 100
