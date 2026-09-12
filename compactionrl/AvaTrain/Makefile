.PHONY: init

init:
	git config push.recurseSubmodules on-demand
	git config submodule.recurse true
	git submodule update --init --recursive
	git submodule foreach ' \
	  b=$$(git config -f "$$toplevel/.gitmodules" "submodule.$$name.branch"); \
	  if [ -n "$$b" ]; then \
	    echo "$$name -> $$b"; \
	    git switch "$$b" 2>/dev/null \
	      || git switch -c "$$b" --track "origin/$$b"; \
	  else \
	    echo "$$name: no branch set, leaving detached"; \
	  fi'
	uv sync

sync-nogpu:
	uv sync --no-install-package deep-ep --no-install-package transformer-engine-torch --no-install-package mamba-ssm --no-install-package causal-conv1d