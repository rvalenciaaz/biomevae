"""Model architectures provided by :mod:`biomevae`."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .vae import VAE, get_activation

__all__ = ["VAE", "get_activation"]

from .graph import TaxonomyGraphVAE
from .treeprior import TreeStructuredPriorVAE
from .phylo_fusion import DeepPhyloFusionVAE
from .flowxformer import FlowXFormerVAE
__all__.extend([
    "TaxonomyGraphVAE",
    "TreeStructuredPriorVAE",
    "DeepPhyloFusionVAE",
    "FlowXFormerVAE",
])

# DIVA building blocks are dependency-free — always available.
from .diva import (
    AuxClassifier,
    CategoryConditionalPrior,
    DIVAEncoderHeads,
    DIVALoss,
    DIVALossOutputs,
    gaussian_kl,
    gaussian_kl_to_standard_normal,
)
__all__.extend([
    "AuxClassifier",
    "CategoryConditionalPrior",
    "DIVAEncoderHeads",
    "DIVALoss",
    "DIVALossOutputs",
    "gaussian_kl",
    "gaussian_kl_to_standard_normal",
])

# DIVA β-VAE backbone — non-taxonomy MLP, dependency-free.
from .diva_betavae import DIVABetaVAE
__all__.append("DIVABetaVAE")

try:  # optional dependency for hyperbolic models
    from .hyperbolic import HyperbolicVAE as _HyperbolicVAE
except ImportError:  # pragma: no cover - geoopt is optional
    _HyperbolicVAE = None  # type: ignore[assignment]
else:  # pragma: no cover - executed when geoopt is installed
    HyperbolicVAE = _HyperbolicVAE
    __all__.append("HyperbolicVAE")

try:  # optional dependency for graph ZI model
    from .hgvae_zi import HGVAE_ZI as _HGVAE_ZI
except ImportError:  # pragma: no cover - torch_geometric is optional
    _HGVAE_ZI = None  # type: ignore[assignment]
else:  # pragma: no cover
    HGVAE_ZI = _HGVAE_ZI
    __all__.append("HGVAE_ZI")

from .tree_dtm_vae import TreeDTMVAE as _TreeDTMVAE
TreeDTMVAE = _TreeDTMVAE
__all__.append("TreeDTMVAE")

from .diva_treedtmvae import DIVATreeDTMVAE as _DIVATreeDTMVAE
DIVATreeDTMVAE = _DIVATreeDTMVAE
__all__.append("DIVATreeDTMVAE")

from .phylodiva_treedtmvae import PhyloDIVATreeDTMVAE as _PhyloDIVATreeDTMVAE
PhyloDIVATreeDTMVAE = _PhyloDIVATreeDTMVAE
__all__.append("PhyloDIVATreeDTMVAE")

from .taxi_treedtmvae import TAXIDIVATreeDTMVAE as _TAXIDIVATreeDTMVAE
TAXIDIVATreeDTMVAE = _TAXIDIVATreeDTMVAE
__all__.append("TAXIDIVATreeDTMVAE")

try:
    from .philrvae import PhILRVAE as _PhILRVAE
except ImportError:  # pragma: no cover - flowxformer/taxonomy deps missing
    _PhILRVAE = None  # type: ignore[assignment]
else:
    PhILRVAE = _PhILRVAE
    __all__.append("PhILRVAE")

try:
    from .dsvae import DSVAE as _DSVAE, ClassConditionalPrior as _DSPrior
except ImportError:  # pragma: no cover
    _DSVAE = None  # type: ignore[assignment]
    _DSPrior = None  # type: ignore[assignment]
else:
    DSVAE = _DSVAE
    ClassConditionalPrior = _DSPrior
    __all__.extend(["DSVAE", "ClassConditionalPrior"])

try:  # optional dependency on geoopt (same as HyperbolicVAE)
    from .hyperbolic_philrvae import HyperbolicPhILRVAE as _HyperbolicPhILRVAE
except ImportError:  # pragma: no cover - geoopt is optional
    _HyperbolicPhILRVAE = None  # type: ignore[assignment]
else:  # pragma: no cover
    HyperbolicPhILRVAE = _HyperbolicPhILRVAE
    __all__.append("HyperbolicPhILRVAE")

try:
    from .diva_hyp_philrvae import (
        DIVAHyperbolicPhILRVAE as _DIVAHyperbolicPhILRVAE,
    )
except ImportError:  # pragma: no cover - geoopt optional
    _DIVAHyperbolicPhILRVAE = None  # type: ignore[assignment]
else:  # pragma: no cover
    DIVAHyperbolicPhILRVAE = _DIVAHyperbolicPhILRVAE
    __all__.append("DIVAHyperbolicPhILRVAE")

try:
    from .phylodiva_hyp_philrvae import (
        PhyloDIVAHyperbolicPhILRVAE as _PhyloDIVAHyperbolicPhILRVAE,
    )
except ImportError:  # pragma: no cover - geoopt optional
    _PhyloDIVAHyperbolicPhILRVAE = None  # type: ignore[assignment]
else:  # pragma: no cover
    PhyloDIVAHyperbolicPhILRVAE = _PhyloDIVAHyperbolicPhILRVAE
    __all__.append("PhyloDIVAHyperbolicPhILRVAE")

try:
    from .taxi_hyp_philrvae import (
        TAXIHyperbolicPhILRVAE as _TAXIHyperbolicPhILRVAE,
    )
except ImportError:  # pragma: no cover - geoopt optional
    _TAXIHyperbolicPhILRVAE = None  # type: ignore[assignment]
else:  # pragma: no cover
    TAXIHyperbolicPhILRVAE = _TAXIHyperbolicPhILRVAE
    __all__.append("TAXIHyperbolicPhILRVAE")


def __getattr__(name: str) -> Any:
    if name == "HyperbolicVAE":
        if _HyperbolicVAE is None:
            raise ImportError(
                "HyperbolicVAE requires the optional 'geoopt' dependency. Install "
                "biomevae with the 'hyper' extra (pip install biomevae[hyper]) or "
                "install geoopt manually."
            )
        return _HyperbolicVAE
    if name == "HGVAE_ZI":
        if _HGVAE_ZI is None:
            raise ImportError(
                "HGVAE_ZI requires the optional 'torch_geometric' dependency. "
                "Install torch-geometric manually to use this model."
            )
        return _HGVAE_ZI
    if name == "TreeDTMVAE":
        return _TreeDTMVAE
    if name == "DIVATreeDTMVAE":
        return _DIVATreeDTMVAE
    if name == "PhyloDIVATreeDTMVAE":
        return _PhyloDIVATreeDTMVAE
    if name == "TAXIDIVATreeDTMVAE":
        return _TAXIDIVATreeDTMVAE
    if name == "PhILRVAE":
        if _PhILRVAE is None:
            raise ImportError(
                "PhILRVAE requires the 'flowxformer' and 'taxonomy' modules. "
                "Ensure all core dependencies are installed."
            )
        return _PhILRVAE
    if name == "DSVAE":
        if _DSVAE is None:
            raise ImportError(
                "DSVAE requires the 'flowxformer' and 'taxonomy' modules. "
                "Ensure all core dependencies are installed."
            )
        return _DSVAE
    if name == "ClassConditionalPrior":
        if _DSPrior is None:
            raise ImportError(
                "ClassConditionalPrior requires the DSVAE module."
            )
        return _DSPrior
    if name == "HyperbolicPhILRVAE":
        if _HyperbolicPhILRVAE is None:
            raise ImportError(
                "HyperbolicPhILRVAE requires the optional 'geoopt' dependency. "
                "Install biomevae with the 'hyper' extra "
                "(pip install biomevae[hyper]) or install geoopt manually."
            )
        return _HyperbolicPhILRVAE
    if name == "DIVAHyperbolicPhILRVAE":
        if _DIVAHyperbolicPhILRVAE is None:
            raise ImportError(
                "DIVAHyperbolicPhILRVAE requires the optional 'geoopt' "
                "dependency.  Install biomevae with the 'hyper' extra."
            )
        return _DIVAHyperbolicPhILRVAE
    if name == "PhyloDIVAHyperbolicPhILRVAE":
        if _PhyloDIVAHyperbolicPhILRVAE is None:
            raise ImportError(
                "PhyloDIVAHyperbolicPhILRVAE requires the optional 'geoopt' "
                "dependency.  Install biomevae with the 'hyper' extra."
            )
        return _PhyloDIVAHyperbolicPhILRVAE
    if name == "TAXIHyperbolicPhILRVAE":
        if _TAXIHyperbolicPhILRVAE is None:
            raise ImportError(
                "TAXIHyperbolicPhILRVAE requires the optional 'geoopt' "
                "dependency.  Install biomevae with the 'hyper' extra."
            )
        return _TAXIHyperbolicPhILRVAE
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from .hyperbolic import HyperbolicVAE
    from .hgvae_zi import HGVAE_ZI
    from .tree_dtm_vae import TreeDTMVAE
    from .diva_treedtmvae import DIVATreeDTMVAE
    from .phylodiva_treedtmvae import PhyloDIVATreeDTMVAE
    from .taxi_treedtmvae import TAXIDIVATreeDTMVAE
    from .philrvae import PhILRVAE
    from .dsvae import DSVAE, ClassConditionalPrior
    from .hyperbolic_philrvae import HyperbolicPhILRVAE
    from .diva_hyp_philrvae import DIVAHyperbolicPhILRVAE
    from .phylodiva_hyp_philrvae import PhyloDIVAHyperbolicPhILRVAE
    from .taxi_hyp_philrvae import TAXIHyperbolicPhILRVAE
