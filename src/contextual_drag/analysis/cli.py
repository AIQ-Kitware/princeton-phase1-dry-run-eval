"""Umbrella CLI for contextual_drag.analysis; registers the analysis subgroups."""
from __future__ import annotations

import scriptconfig as scfg

from contextual_drag.analysis.error_conditioning.cli import ErrorConditioningCLI
from contextual_drag.analysis.mitigation_buckets.cli import MitigationBucketsCLI
from contextual_drag.analysis.route_model.cli import RouteModelCLI
from contextual_drag.analysis.ted.cli import TedCLI
from contextual_drag.analysis.ccg_v2.cli import CcgV2CLI


class AnalysisCLI(scfg.ModalCLI):
    error_conditioning = ErrorConditioningCLI
    ted = TedCLI
    mitigation_buckets = MitigationBucketsCLI
    route_model = RouteModelCLI
    ccg_v2 = CcgV2CLI
