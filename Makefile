VENV_NAME:=venv

activate_env:
	source $(VENV_NAME)/bin/activate

download_data:
	@pip install -r notebook/requirements.txt
	@cd notebook && \
	python download_data.py
