"""
inference/condition_assessment.py
----------------------------------
Rule-based condition assessment module — Chapter 5 of the project report.

Once the Random Forest has identified the tool class, this module checks
whether the measured geometry still falls within the acceptable range for
a tool of that class in good condition.

Why rules instead of a learned anomaly detector?
  Two reasons drove this design choice:

  1. Engineering knowledge is already available.  The geometric signature of
     common failure modes is well-understood: a worn drill tip loses
     circularity; a milling cutter with a broken flute shows fewer vertices;
     a parting blade that has been damaged departs from its characteristic
     high aspect ratio.  Encoding that knowledge as explicit inequalities
     produces verdicts that a maintenance technician can trace back,
     challenge, and correct — without any machine learning background.

  2. Faulty-tool data does not exist.  Training a reliable anomaly detector
     requires a large catalogue of documented faults; assembling that from
     scratch would take far longer than the project timeline allows.

Verdict scale (three tiers — more actionable than a binary pass/fail):
  ✅  Fit for Use       — all checks pass; tool can continue in service
  ⚠️  Use with Caution  — one fault flagged; redirect to non-precision tasks
  ❌  Not Suitable      — two or more faults; withdraw from service

Thresholds are stored in config.CONDITION_RULES so a domain expert can
adjust them in one place without touching the logic here.

Reference: Chapter 5 and Table 3 of the project report.
"""

from __future__ import annotations

from typing import List, Tuple

from config import CONDITION_RULES
from utils.logger import get_logger

logger = get_logger(__name__)

# Verdict constants — using strings so they are directly printable / loggable
VERDICT_FIT       = "✅  Fit for Use"
VERDICT_CAUTION   = "⚠️  Use with Caution"
VERDICT_UNSUITABLE = "❌  Not Suitable"


def check_suitability(
    label: str,
    features: List[float],
) -> Tuple[str, List[str]]:
    """
    Evaluate whether a tool's measured geometry is within acceptable limits
    for its class.

    Parameters
    ----------
    label : str
        Predicted tool class (e.g. ``"drill"``, ``"milling"``).
    features : list of float
        ``[aspect_ratio, circularity, edge_count, area]`` for the specimen.

    Returns
    -------
    verdict : str
        One of VERDICT_FIT, VERDICT_CAUTION, or VERDICT_UNSUITABLE.
    faults : list of str
        Human-readable descriptions of every rule that was triggered.
        Empty list when the tool passes all checks.
    """
    if len(features) < 4:
        raise ValueError(
            f"Expected 4 features [aspect_ratio, circularity, edge_count, area], "
            f"got {len(features)}: {features}"
        )

    aspect_ratio = features[0]
    circularity  = features[1]
    edge_count   = int(features[2])
    # area is not used in the rule engine directly, but callers may need it
    # for logging or display purposes.

    faults: List[str] = []
    rules = CONDITION_RULES.get(label, {})

    if not rules:
        logger.debug("No condition rules defined for class '%s' — skipping check.", label)
        return VERDICT_FIT, []

    # ---- Drill -----------------------------------------------------------
    # Near-circular profile, minimal vertex count.  Any deviation signals tip
    # wear, deformation, or partial breakage.
    if label == "drill":
        max_ec = rules.get("max_edge_count")
        if max_ec is not None and edge_count > max_ec:
            faults.append(
                f"Extra contour vertices detected (edge_count={edge_count} > {max_ec}) "
                "— likely tip fragmentation or surface damage"
            )
        min_circ = rules.get("min_circularity")
        if min_circ is not None and circularity < min_circ:
            faults.append(
                f"Low circularity ({circularity:.3f} < {min_circ}) "
                "— tip deformation or asymmetric wear"
            )
        min_ar = rules.get("min_aspect_ratio")
        if min_ar is not None and aspect_ratio < min_ar:
            faults.append(
                f"Aspect ratio below expected ({aspect_ratio:.3f} < {min_ar}) "
                "— possible breakage or abnormally short body"
            )

    # ---- Milling cutter --------------------------------------------------
    # Multi-flute cutter; tooth loss reduces vertex count, edge wear rounds
    # the profile and raises circularity.
    elif label == "milling":
        min_ec = rules.get("min_edge_count")
        if min_ec is not None and edge_count < min_ec:
            faults.append(
                f"Insufficient cutting edges (edge_count={edge_count} < {min_ec}) "
                "— flute loss or tooth breakage"
            )
        max_circ = rules.get("max_circularity")
        if max_circ is not None and circularity > max_circ:
            faults.append(
                f"Circularity too high ({circularity:.3f} > {max_circ}) "
                "— cutting edges worn smooth"
            )

    # ---- Reamer ----------------------------------------------------------
    # Symmetrically arranged flutes; flute loss shows as reduced vertex count
    # and reduced profile symmetry (approximated by circularity).
    elif label == "reamer":
        min_ec = rules.get("min_edge_count")
        if min_ec is not None and edge_count < min_ec:
            faults.append(
                f"Flute count below minimum (edge_count={edge_count} < {min_ec}) "
                "— cutting efficiency compromised"
            )
        min_circ = rules.get("min_circularity")
        if min_circ is not None and circularity < min_circ:
            faults.append(
                f"Low circularity ({circularity:.3f} < {min_circ}) "
                "— asymmetric wear producing non-circular profile"
            )

    # ---- Gear cutter -----------------------------------------------------
    # High tooth density is the defining characteristic; tooth loss is the
    # most obvious failure mode and shows directly as reduced vertex count.
    elif label == "gear":
        min_ec = rules.get("min_edge_count")
        if min_ec is not None and edge_count < min_ec:
            faults.append(
                f"Tooth count below expected (edge_count={edge_count} < {min_ec}) "
                "— one or more teeth missing or severely worn"
            )

    # ---- Lathe tool ------------------------------------------------------
    # Simple angular insert geometry; any unexpected profile complexity or
    # non-standard shape warrants closer inspection.
    elif label == "lathe":
        max_ec = rules.get("max_edge_count")
        if max_ec is not None and edge_count > max_ec:
            faults.append(
                f"Profile complexity beyond expectation (edge_count={edge_count} > {max_ec}) "
                "— inspect for chipping or insert fracture"
            )
        min_ar = rules.get("min_aspect_ratio")
        if min_ar is not None and aspect_ratio < min_ar:
            faults.append(
                f"Aspect ratio too low ({aspect_ratio:.3f} < {min_ar}) "
                "— non-standard geometry; confirm insert seating"
            )

    # ---- Parting tool ----------------------------------------------------
    # Thin blade; damage typically thickens or deforms the blade, reducing
    # the characteristic high aspect ratio.
    elif label == "parting":
        min_ar = rules.get("min_aspect_ratio")
        if min_ar is not None and aspect_ratio < min_ar:
            faults.append(
                f"Aspect ratio below expected ({aspect_ratio:.3f} < {min_ar}) "
                "— blade thickening or deformation detected"
            )

    # ---- Verdict ---------------------------------------------------------
    n_faults = len(faults)

    if n_faults == 0:
        verdict = VERDICT_FIT
    elif n_faults == 1:
        verdict = VERDICT_CAUTION
    else:
        verdict = VERDICT_UNSUITABLE

    if faults:
        logger.debug("Faults for '%s': %s", label, faults)

    return verdict, faults
