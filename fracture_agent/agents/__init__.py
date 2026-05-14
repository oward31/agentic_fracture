"""Agents — each is a tightly-scoped function that reads & mutates SessionState."""
from .receptionist import receptionist
from .architect   import architect
from .strategist  import strategist
from .synthesizer import synthesizer
from .inspector   import inspector
from .debugger    import debugger
from .advisor     import advisor
from .material_helper import resolve_material
