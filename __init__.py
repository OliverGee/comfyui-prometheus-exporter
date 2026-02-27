from .exporter import setup_metrics_endpoint, try_install_instrumentation

setup_metrics_endpoint()
try_install_instrumentation()

# This package does not provide UI nodes; it only adds a server endpoint.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
