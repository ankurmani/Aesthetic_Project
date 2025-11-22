# ===============================================================
#  Anchor Label Encoder for All 39 Anchors
# ===============================================================

import re

def build_anchor_label_encoder(anchors):
    """
    anchors: list of Anchor(name="RT-1", x, y, kind)
    Returns:
        anchor_to_id:  dict  e.g., {"Rule-of-Thirds (RT-1)": 0}
        id_to_anchor:  dict  reverse mapping
    """
    anchor_labels = []

    # 1) Convert anchor name → human-readable composition label
    for anc in anchors:
        comp = anchor_to_composition(anc.name)
        label = f"{comp} ({anc.name})"
        anchor_labels.append(label)

    # 2) Create sorted unique list for stable ordering
    anchor_labels = sorted(set(anchor_labels))

    # 3) Encode
    anchor_to_id = {label: idx for idx, label in enumerate(anchor_labels)}
    id_to_anchor = {idx: label for label, idx in anchor_to_id.items()}

    return anchor_to_id, id_to_anchor


# ===============================================================
# Composition name generator
# ===============================================================

def anchor_to_composition(anchor_name: str) -> str:
    """Map anchor names to human-readable compositions."""
    if anchor_name.startswith("RT-"):
        return "Rule-of-Thirds"
    if anchor_name.startswith("GR-"):
        return "Golden Ratio"
    if anchor_name.startswith("Q-"):
        return "Quadrant Composition"
    if anchor_name.startswith("F5-"):
        return "Rule-of-Fifths"
    if anchor_name.startswith("CL-"):
        return "Center-Line Composition"
    if anchor_name.startswith("SYM-"):
        return "Symmetry Alignment"
    if anchor_name.startswith("DL-"):
        return "Diagonal Composition"
    if anchor_name == "CENTER":
        return "Centered Composition"
    return "Unknown Composition"


# ===============================================================
# Example usage
# ===============================================================

if __name__ == "__main__":
    # Example: define anchors artificially for testing
    anchors = [
        Anchor("RT-1", 100, 200),
        Anchor("GR-1", 120, 300),
        Anchor("DL-3", 500, 600),
        Anchor("CENTER", 540, 960),
    ]

    anchor_to_id, id_to_anchor = build_anchor_label_encoder(anchors)

    print("Anchor → ID mapping:")
    for k, v in anchor_to_id.items():
        print(f"{v:2d} : {k}")

    print("\nID → Anchor mapping:")
    for k, v in id_to_anchor.items():
        print(f"{k:2d} : {v}")
