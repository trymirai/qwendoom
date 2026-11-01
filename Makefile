# Top-level build. Each component owns its own Makefile.
.PHONY: all native doom weights third_party clean

all: third_party native doom

third_party:
	uv run --no-project scripts/fetch_third_party.py

native:
	$(MAKE) -C native/host
	$(MAKE) -C native/trace
	$(MAKE) -C native/engine
	$(MAKE) -C native/runner

doom: third_party
	$(MAKE) -C doom

weights:
	uv run python -m qwendoom build --weights

clean:
	rm -rf build
