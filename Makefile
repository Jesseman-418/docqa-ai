.PHONY: install run test clean lint

install:
	python3 -m pip install -r requirements.txt

run:
	python3 -m streamlit run app/main.py

test:
	python3 -m pytest tests/ -v

clean:
	rm -rf vector_store/ __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

lint:
	python3 -m ruff check app/ tests/
