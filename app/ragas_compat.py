"""Compatibilidade: o ragas 0.4.x importa langchain_community.chat_models.vertexai, módulo que o
langchain-community 0.4+ removeu. Este arquivo cria um substituto vazio SÓ se o módulo não existir.
Importe-o antes de importar o ragas. Pode ser apagado quando o ragas corrigir o import."""
import sys
import types

try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ImportError:
    _m = types.ModuleType("langchain_community.chat_models.vertexai")
    _m.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules[_m.__name__] = _m
