"""Generate SOTA-calibrated editing prompts for the benchmark.

Uses image analysis metadata + prompt taxonomy to generate grounded prompts
deterministically. No LLM API calls needed — prompts are crafted from
templates calibrated to Complex-Edit difficulty levels.

Usage:
    python scripts/generate_prompts.py
    python scripts/generate_prompts.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_PATH = ROOT / "prompts" / "source_images" / "image_analysis.json"
TAXONOMY_PATH = ROOT / "config" / "prompt_taxonomy.yaml"
L1_OUTPUT = ROOT / "prompts" / "layer1_gold.json"
L2_OUTPUT = ROOT / "prompts" / "layer2_proprietary.json"

# ── Which categories apply to each image group ──
GROUP_CATS = {
    "person": [
        "color_material",
        "object_replace",
        "background_change",
        "style_transfer",
        "lighting_weather",
        "spatial_pose",
        "anatomy_identity",
        "multi_turn",
    ],
    "object": [
        "color_material",
        "object_add",
        "object_remove",
        "object_replace",
        "background_change",
        "style_transfer",
        "lighting_weather",
        "physics",
        "text_editing",
        "multi_turn",
    ],
    "scene": [
        "color_material",
        "object_add",
        "object_remove",
        "object_replace",
        "background_change",
        "style_transfer",
        "lighting_weather",
        "spatial_pose",
        "physics",
        "text_editing",
        "multi_turn",
    ],
    "style": [
        "color_material",
        "object_add",
        "style_transfer",
        "lighting_weather",
        "physics",
        "multi_turn",
    ],
}

# ═══════════════════════════════════════════════════════════════
# Prompt templates — grounded, SOTA-calibrated
# Key: (category, difficulty) → list of template functions
# Each takes image_info dict → (edit_instruction, atoms, turns)
# ═══════════════════════════════════════════════════════════════


def _objs(info):
    return info.get("objects", info.get("typical_objects", []))


def _first_obj(info):
    objs = _objs(info)
    return objs[0] if objs else "the main object"


def _second_obj(info):
    objs = _objs(info)
    return objs[1] if len(objs) > 1 else "the secondary element"


def _mat(info):
    mats = info.get("materials", [])
    return mats[0] if mats else "wood"


def _desc(info):
    return info.get("description_full", info.get("primary_subject", "the scene"))


def _person_type(info):
    cat = info.get("category", "")
    if "portrait" in cat:
        return "portrait"
    if "group" in cat:
        return "group"
    return "fullbody"


# ── Color/Material ──


def cm_easy_1(info):
    obj = _first_obj(info)
    colors = ["red", "blue", "bright yellow", "matte black", "white"]
    c = random.choice(colors)
    return (
        f"Change the color of the {obj} to {c}",
        [
            ("instruction_following", "instruction", f"Is the {obj} now {c}?"),
            ("visual_consistency", "preservation", "Is the rest of the scene unchanged?"),
        ],
        1,
    )


def cm_easy_2(info):
    obj = _first_obj(info)
    textures = ["glossy", "matte", "rough", "smooth"]
    t = random.choice(textures)
    return (
        f"Make the {obj} {t}",
        [
            ("instruction_following", "instruction", f"Does the {obj} have a {t} finish?"),
            ("visual_consistency", "preservation", "Is the background unchanged?"),
        ],
        1,
    )


def cm_medium_1(info):
    obj = _first_obj(info)
    materials = [
        ("wooden", "brushed steel"),
        ("metal", "weathered stone"),
        ("plastic", "ceramic"),
        ("concrete", "red brick"),
    ]
    old, new = random.choice(materials)
    return (
        f"Change the {obj} from {old} to {new} while keeping its shape and size",
        [
            ("instruction_following", "instruction", f"Is the {obj} now made of {new}?"),
            ("visual_consistency", "structure", f"Is the shape and size of the {obj} preserved?"),
            ("detail_preservation", "quality", f"Does the {new} texture look realistic?"),
            ("visual_consistency", "preservation", "Is the surrounding area unchanged?"),
        ],
        1,
    )


def cm_medium_2(info):
    _first_obj(info)
    return (
        f"Convert all {_mat(info)} surfaces in the scene to polished marble, keeping object shapes",
        [
            (
                "instruction_following",
                "instruction",
                "Are the specified surfaces now polished marble?",
            ),
            ("visual_consistency", "structure", "Are all object shapes preserved?"),
            ("detail_preservation", "quality", "Does the marble show realistic veining and sheen?"),
        ],
        1,
    )


def cm_hard_1(info):
    obj = _first_obj(info)
    return (
        f"Transform the {obj} from its current material to weathered copper with green verdigris patina in the crevices, while maintaining all surface details, reflections consistent with copper, and correct shadow coloring",
        [
            ("instruction_following", "instruction", f"Is the {obj} now copper-colored?"),
            (
                "instruction_following",
                "instruction",
                "Is there green verdigris patina visible in crevices?",
            ),
            ("visual_consistency", "structure", f"Are all surface details of the {obj} preserved?"),
            ("detail_preservation", "quality", "Are reflections consistent with a copper surface?"),
            ("detail_preservation", "lighting", "Is shadow coloring correct for a copper object?"),
            ("visual_consistency", "preservation", "Is the background unchanged?"),
        ],
        1,
    )


# ── Object Add ──


def oa_easy_1(info):
    objects = ["a potted plant", "a coffee mug", "a pair of sunglasses", "a small clock", "a book"]
    new_obj = random.choice(objects)
    return (
        f"Add {new_obj} to the scene",
        [
            ("instruction_following", "instruction", f"Is {new_obj} now visible in the scene?"),
            ("visual_consistency", "preservation", "Are all original objects unchanged?"),
        ],
        1,
    )


def oa_medium_1(info):
    _desc(info)[:50]
    items = [
        (
            "a red fire extinguisher on the floor to the left",
            "a red fire extinguisher on the left side",
        ),
        ("a sleeping cat curled up on the nearest surface", "a sleeping cat on a surface"),
        ("a vintage radio on the shelf in the background", "a vintage radio in the background"),
    ]
    full, short = random.choice(items)
    return (
        f"Add {full}",
        [
            ("instruction_following", "instruction", f"Is {short} now visible?"),
            (
                "instruction_following",
                "instruction",
                "Is it placed in the correct position described?",
            ),
            ("detail_preservation", "quality", "Does the added object match the scene's lighting?"),
            ("visual_consistency", "preservation", "Are all original objects undisturbed?"),
        ],
        1,
    )


def oa_hard_1(info):
    return (
        "Add a large ornate mirror leaning against the back wall, with correct reflections of the foreground objects and lighting visible in the mirror surface",
        [
            (
                "instruction_following",
                "instruction",
                "Is a large ornate mirror visible leaning against the back wall?",
            ),
            (
                "instruction_following",
                "instruction",
                "Does the mirror show reflections of foreground objects?",
            ),
            (
                "detail_preservation",
                "quality",
                "Are the reflections optically consistent with the scene geometry?",
            ),
            (
                "detail_preservation",
                "lighting",
                "Does the mirror reflect the scene's actual lighting?",
            ),
            ("visual_consistency", "preservation", "Are all original objects unchanged?"),
            (
                "visual_consistency",
                "structure",
                "Is the spatial scale of the mirror consistent with the scene?",
            ),
        ],
        1,
    )


# ── Object Remove ──


def or_easy_1(info):
    obj = _first_obj(info)
    return (
        f"Remove the {obj}",
        [
            ("instruction_following", "instruction", f"Is the {obj} removed from the scene?"),
            ("visual_consistency", "preservation", "Is the area where it was naturally filled in?"),
        ],
        1,
    )


def or_medium_1(info):
    obj = _first_obj(info)
    obj2 = _second_obj(info)
    return (
        f"Remove the {obj} without affecting the {obj2} next to it",
        [
            ("instruction_following", "instruction", f"Is the {obj} removed?"),
            ("visual_consistency", "preservation", f"Is the {obj2} completely unchanged?"),
            ("detail_preservation", "quality", "Is the infilled area seamless and natural?"),
            (
                "visual_consistency",
                "preservation",
                "Is the background behind the removed object intact?",
            ),
        ],
        1,
    )


def or_hard_1(info):
    obj = _first_obj(info)
    return (
        f"Remove the {obj} and naturally reconstruct the occluded background behind it, matching the surrounding perspective, texture continuity, and lighting gradient",
        [
            ("instruction_following", "instruction", f"Is the {obj} completely removed?"),
            ("detail_preservation", "quality", "Is the reconstructed background seamless?"),
            (
                "detail_preservation",
                "quality",
                "Does the texture continuity match surrounding areas?",
            ),
            (
                "detail_preservation",
                "lighting",
                "Is the lighting gradient consistent across the filled area?",
            ),
            (
                "visual_consistency",
                "structure",
                "Is perspective correct in the reconstructed region?",
            ),
            ("visual_consistency", "preservation", "Are all other objects undisturbed?"),
        ],
        1,
    )


# ── Object Replace ──


def rp_easy_1(info):
    obj = _first_obj(info)
    replacements = {
        "mug": "glass bottle",
        "chair": "stool",
        "table": "desk",
        "car": "truck",
        "flower": "cactus",
        "lamp": "candle",
        "bicycle": "motorcycle",
        "book": "tablet",
    }
    for k, v in replacements.items():
        if k in obj.lower():
            return (
                f"Replace the {obj} with a {v}",
                [
                    (
                        "instruction_following",
                        "instruction",
                        f"Is there now a {v} where the {obj} was?",
                    ),
                    ("visual_consistency", "preservation", "Is the background unchanged?"),
                ],
                1,
            )
    return (
        f"Replace the {obj} with a potted plant",
        [
            (
                "instruction_following",
                "instruction",
                f"Is there a potted plant where the {obj} was?",
            ),
            ("visual_consistency", "preservation", "Is the rest of the scene unchanged?"),
        ],
        1,
    )


def rp_medium_1(info):
    obj = _first_obj(info)
    return (
        f"Replace the {obj} with a vintage version of the same type, maintaining identical size and position",
        [
            (
                "instruction_following",
                "instruction",
                f"Is the {obj} replaced with a vintage-style alternative?",
            ),
            (
                "visual_consistency",
                "structure",
                "Is the replacement the same size as the original?",
            ),
            ("visual_consistency", "structure", "Is the replacement in the exact same position?"),
            ("detail_preservation", "quality", "Does the vintage style look authentic?"),
        ],
        1,
    )


def rp_hard_1(info):
    obj = _first_obj(info)
    return (
        f"Replace the {obj} with a transparent glass version of the same shape, showing correct refraction of background elements through the glass, caustic light patterns on the surface below, and specular highlights matching the scene lighting",
        [
            (
                "instruction_following",
                "instruction",
                f"Is the {obj} replaced with a glass version?",
            ),
            (
                "instruction_following",
                "instruction",
                "Is the glass version the same shape as the original?",
            ),
            (
                "detail_preservation",
                "quality",
                "Is refraction of background visible through the glass?",
            ),
            (
                "detail_preservation",
                "quality",
                "Are caustic light patterns visible on the surface below?",
            ),
            (
                "detail_preservation",
                "lighting",
                "Are specular highlights consistent with scene lighting?",
            ),
            ("visual_consistency", "preservation", "Is the rest of the scene unchanged?"),
        ],
        1,
    )


# ── Background Change ──


def bg_easy_1(info):
    bgs = ["a tropical beach", "a plain white studio", "a city skyline at dusk", "a forest path"]
    bg = random.choice(bgs)
    return (
        f"Change the background to {bg}",
        [
            ("instruction_following", "instruction", f"Is the background now {bg}?"),
            ("visual_consistency", "preservation", "Is the foreground subject unchanged?"),
        ],
        1,
    )


def bg_medium_1(info):
    return (
        "Replace the background with a rainy urban street while keeping the foreground subject's edge detail sharp and lighting consistent",
        [
            ("instruction_following", "instruction", "Is the background now a rainy urban street?"),
            ("visual_consistency", "preservation", "Is the foreground subject completely intact?"),
            ("detail_preservation", "quality", "Are the subject's edges clean without fringing?"),
            (
                "detail_preservation",
                "lighting",
                "Is the lighting on the subject consistent with the new rainy scene?",
            ),
        ],
        1,
    )


def bg_hard_1(info):
    return (
        "Transport the entire scene to the surface of Mars: red rocky terrain replacing the ground, salmon-pink sky, distant mountains on the horizon, atmospheric haze, and all existing objects casting shadows consistent with a lower sun angle",
        [
            ("instruction_following", "instruction", "Is the ground now red Martian terrain?"),
            ("instruction_following", "instruction", "Is the sky salmon-pink?"),
            (
                "instruction_following",
                "instruction",
                "Are distant mountains visible on the horizon?",
            ),
            ("detail_preservation", "quality", "Is there atmospheric haze visible?"),
            ("detail_preservation", "lighting", "Are shadows consistent with the new sun angle?"),
            ("visual_consistency", "preservation", "Are all foreground objects preserved?"),
            (
                "visual_consistency",
                "structure",
                "Is the spatial arrangement of existing objects intact?",
            ),
        ],
        1,
    )


# ── Style Transfer ──


def st_easy_1(info):
    styles = ["a pencil sketch", "a watercolor painting", "a pop art poster", "an oil painting"]
    s = random.choice(styles)
    return (
        f"Make this look like {s}",
        [
            ("instruction_following", "instruction", f"Does the image look like {s}?"),
            ("visual_consistency", "structure", "Is the original composition preserved?"),
        ],
        1,
    )


def st_medium_1(info):
    return (
        "Convert to a 1970s Kodachrome film photograph look: warm color cast, slight grain, slightly faded shadows, high saturation reds and greens",
        [
            (
                "instruction_following",
                "instruction",
                "Does the image have a warm vintage color cast?",
            ),
            ("instruction_following", "instruction", "Is film grain visible?"),
            ("instruction_following", "instruction", "Are reds and greens highly saturated?"),
            ("visual_consistency", "structure", "Is the composition and content preserved?"),
        ],
        1,
    )


def st_hard_1(info):
    return (
        "Transform into a Japanese Ukiyo-e woodblock print: flat color planes with no gradient shading, strong black outlines around all forms, traditional compositional rules with foreground/midground/background layers clearly separated, wave patterns in any water, and a cartouche with Japanese text in the upper corner",
        [
            (
                "instruction_following",
                "instruction",
                "Are flat color planes used with no gradient shading?",
            ),
            (
                "instruction_following",
                "instruction",
                "Are strong black outlines visible around forms?",
            ),
            (
                "instruction_following",
                "instruction",
                "Is a text cartouche visible in the upper corner?",
            ),
            (
                "detail_preservation",
                "quality",
                "Are foreground/midground/background clearly layered?",
            ),
            ("visual_consistency", "structure", "Is the original scene composition recognizable?"),
            (
                "detail_preservation",
                "quality",
                "Does the style look authentically Ukiyo-e, not generic?",
            ),
        ],
        1,
    )


# ── Lighting/Weather ──


def lw_easy_1(info):
    changes = ["nighttime", "sunset", "overcast and cloudy", "bright midday"]
    c = random.choice(changes)
    return (
        f"Make it {c}",
        [
            ("instruction_following", "instruction", f"Does the scene look like {c}?"),
            ("visual_consistency", "preservation", "Are all objects in the scene preserved?"),
        ],
        1,
    )


def lw_medium_1(info):
    return (
        "Change to golden hour lighting with warm tones, long shadows stretching to the right, and a warm color cast on all surfaces",
        [
            (
                "instruction_following",
                "instruction",
                "Does the scene have warm golden hour lighting?",
            ),
            ("instruction_following", "instruction", "Are long shadows cast to the right?"),
            ("detail_preservation", "lighting", "Is there a warm color cast on surfaces?"),
            ("visual_consistency", "preservation", "Are all objects and structures unchanged?"),
        ],
        1,
    )


def lw_hard_1(info):
    return (
        "Add a heavy thunderstorm: dark overcast sky with visible lightning bolt, heavy rain streaks consistent with wind from the left, wet reflections on all flat surfaces, puddles forming in low areas, and all shadows softened to match diffused storm lighting",
        [
            ("instruction_following", "instruction", "Is the sky dark and overcast?"),
            ("instruction_following", "instruction", "Are rain streaks visible?"),
            ("instruction_following", "instruction", "Are wet reflections on flat surfaces?"),
            ("detail_preservation", "quality", "Do puddles form in plausible low areas?"),
            ("detail_preservation", "lighting", "Are shadows softened to match diffused lighting?"),
            (
                "visual_consistency",
                "preservation",
                "Are all objects still present and recognizable?",
            ),
        ],
        1,
    )


# ── Spatial/Pose ──


def sp_easy_1(info):
    return (
        "Flip the image horizontally",
        [
            ("instruction_following", "instruction", "Is the image mirrored left-to-right?"),
            ("visual_consistency", "structure", "Are all objects intact after flipping?"),
        ],
        1,
    )


def sp_medium_1(info):
    obj = _first_obj(info)
    return (
        f"Move the {obj} to the opposite side of the scene and adjust its shadow to match the new position",
        [
            ("instruction_following", "instruction", f"Is the {obj} now on the opposite side?"),
            (
                "detail_preservation",
                "lighting",
                "Is the shadow direction correct for the new position?",
            ),
            ("detail_preservation", "quality", "Is the original location naturally infilled?"),
            ("visual_consistency", "preservation", "Are all other objects undisturbed?"),
        ],
        1,
    )


def sp_hard_1(info):
    return (
        "Shift the camera viewpoint 30 degrees to the right: reveal the hidden side of objects, adjust all perspective lines converging to the new vanishing point, update parallax between foreground and background layers, and ensure all shadows remain consistent with the same light source",
        [
            (
                "instruction_following",
                "instruction",
                "Has the viewpoint shifted to reveal previously hidden sides?",
            ),
            (
                "detail_preservation",
                "quality",
                "Do perspective lines converge to a consistent vanishing point?",
            ),
            (
                "detail_preservation",
                "quality",
                "Is there correct parallax between foreground and background?",
            ),
            (
                "detail_preservation",
                "lighting",
                "Are shadows still consistent with the original light source?",
            ),
            (
                "visual_consistency",
                "structure",
                "Are all objects present with correct relative positions?",
            ),
            (
                "visual_consistency",
                "preservation",
                "Are object identities and appearances preserved?",
            ),
        ],
        1,
    )


# ── Anatomy/Identity (person only) ──


def ai_easy_1(info):
    changes = ["smile", "frown", "look surprised", "close their eyes"]
    c = random.choice(changes)
    return (
        f"Make the person {c}",
        [
            (
                "instruction_following",
                "instruction",
                f"Is the person's expression now a {c.split()[0] if ' ' not in c else c}?",
            ),
            (
                "visual_consistency",
                "identity",
                "Is the person still recognizable as the same individual?",
            ),
        ],
        1,
    )


def ai_medium_1(info):
    pt = _person_type(info)
    if pt == "portrait":
        return (
            "Change the person's hair to short silver-grey while keeping their face, expression, and skin tone unchanged",
            [
                ("instruction_following", "instruction", "Is the hair now short and silver-grey?"),
                ("visual_consistency", "identity", "Is the face unchanged?"),
                ("visual_consistency", "preservation", "Is the skin tone unchanged?"),
                ("detail_preservation", "quality", "Does the hair look natural, not painted on?"),
            ],
            1,
        )
    else:
        return (
            "Change the person's outfit to a formal business suit while preserving their pose, body proportions, and face",
            [
                (
                    "instruction_following",
                    "instruction",
                    "Is the person wearing a formal business suit?",
                ),
                ("visual_consistency", "identity", "Is the person's face unchanged?"),
                ("visual_consistency", "structure", "Are body proportions and pose preserved?"),
                (
                    "detail_preservation",
                    "quality",
                    "Does the suit fabric look realistic with proper folds?",
                ),
            ],
            1,
        )


def ai_hard_1(info):
    pt = _person_type(info)
    if pt == "group":
        return (
            "Age everyone in the group by 30 years: add age-appropriate wrinkles, grey/white hair, slightly changed posture for each person, while preserving each individual's identity and their relative positions",
            [
                ("instruction_following", "instruction", "Do all people appear ~30 years older?"),
                (
                    "instruction_following",
                    "instruction",
                    "Is hair grey or white for all individuals?",
                ),
                ("instruction_following", "instruction", "Are wrinkles visible on all faces?"),
                (
                    "visual_consistency",
                    "identity",
                    "Is each person still recognizable as the same individual?",
                ),
                ("visual_consistency", "structure", "Are relative positions of people preserved?"),
                (
                    "detail_preservation",
                    "quality",
                    "Do the aging effects look natural, not like a filter?",
                ),
            ],
            1,
        )
    else:
        return (
            "Age the person by 30 years with realistic aging: add wrinkles around eyes and mouth, grey hair with receding hairline, slight skin sagging, age spots on hands, while keeping their identity recognizable and clothing unchanged",
            [
                ("instruction_following", "instruction", "Does the person appear ~30 years older?"),
                (
                    "instruction_following",
                    "instruction",
                    "Are wrinkles visible around eyes and mouth?",
                ),
                (
                    "instruction_following",
                    "instruction",
                    "Is the hair grey with receding hairline?",
                ),
                ("visual_consistency", "identity", "Is the person still recognizable?"),
                ("visual_consistency", "preservation", "Is the clothing unchanged?"),
                ("detail_preservation", "quality", "Does the aging look natural, not artificial?"),
            ],
            1,
        )


# ── Physics ──


def ph_easy_1(info):
    obj = _first_obj(info)
    return (
        f"Add a realistic shadow under the {obj}",
        [
            ("instruction_following", "instruction", f"Is there a shadow under the {obj}?"),
            (
                "detail_preservation",
                "lighting",
                "Is the shadow direction consistent with scene lighting?",
            ),
        ],
        1,
    )


def ph_medium_1(info):
    obj = _first_obj(info)
    return (
        f"Change the {obj} from matte to a chrome mirror finish, with correct reflections of the surrounding environment visible on its surface",
        [
            (
                "instruction_following",
                "instruction",
                f"Does the {obj} have a chrome mirror finish?",
            ),
            (
                "detail_preservation",
                "quality",
                "Are reflections of surroundings visible on the surface?",
            ),
            ("detail_preservation", "quality", "Are the reflections geometrically plausible?"),
            ("visual_consistency", "structure", f"Is the {obj}'s shape unchanged?"),
        ],
        1,
    )


def ph_hard_1(info):
    return (
        "Replace the still water in the scene with flowing rapids: add white foam at obstacles, motion blur in the water flow direction, elongated and distorted reflections of all bank-side objects, wet splash marks on nearby surfaces, and mist rising from the turbulent areas",
        [
            ("instruction_following", "instruction", "Does the water appear to be flowing rapids?"),
            ("instruction_following", "instruction", "Is white foam visible at obstacles?"),
            ("detail_preservation", "quality", "Are reflections elongated and distorted by flow?"),
            ("detail_preservation", "quality", "Is mist or spray visible near turbulent areas?"),
            ("detail_preservation", "lighting", "Are wet marks visible on nearby surfaces?"),
            (
                "visual_consistency",
                "preservation",
                "Are bank-side objects and landscape preserved?",
            ),
        ],
        1,
    )


# ── Text Editing ──


def te_easy_1(info):
    texts = ["OPEN", "HELLO", "2025", "SALE"]
    t = random.choice(texts)
    return (
        f"Change the visible text to '{t}'",
        [
            ("instruction_following", "instruction", f"Does the text now read '{t}'?"),
            ("visual_consistency", "preservation", "Is the sign/label surface unchanged?"),
        ],
        1,
    )


def te_medium_1(info):
    return (
        "Replace the main visible text with 'GRAND OPENING' in the same font style, size, color, and perspective distortion as the original",
        [
            ("instruction_following", "instruction", "Does the text now read 'GRAND OPENING'?"),
            ("detail_preservation", "quality", "Is the font style similar to the original?"),
            (
                "detail_preservation",
                "quality",
                "Does the text follow the same perspective distortion?",
            ),
            ("visual_consistency", "preservation", "Is the surface and background unchanged?"),
        ],
        1,
    )


def te_hard_1(info):
    return (
        "Change all visible text to Chinese characters: translate each piece of text, match the original font weight and style for each instance, follow the surface curvature and perspective of each sign/label, and ensure text rendering resolution matches the rest of the image",
        [
            (
                "instruction_following",
                "instruction",
                "Is all visible text now in Chinese characters?",
            ),
            (
                "instruction_following",
                "instruction",
                "Does each text instance match its original font weight?",
            ),
            ("detail_preservation", "quality", "Does text follow surface curvature correctly?"),
            (
                "detail_preservation",
                "quality",
                "Is text rendering resolution consistent with the image?",
            ),
            ("visual_consistency", "preservation", "Are all non-text elements unchanged?"),
            ("visual_consistency", "structure", "Are sign and label surfaces undistorted?"),
        ],
        1,
    )


# ── Multi-turn ──


def mt_easy_1(info):
    if info["group"] == "person":
        return (
            "Remove the person's accessories → Change the background to plain grey",
            [
                ("instruction_following", "instruction", "Are accessories removed?"),
                ("instruction_following", "instruction", "Is the background now plain grey?"),
                ("visual_consistency", "identity", "Is the person's face and body preserved?"),
            ],
            ["Remove the person's accessories", "Change the background to plain grey"],
        )
    obj = _first_obj(info)
    return (
        f"Remove the {obj} → Add a vase of flowers in its place",
        [
            ("instruction_following", "instruction", f"Is the {obj} removed?"),
            ("instruction_following", "instruction", "Is a vase of flowers now in that location?"),
            ("visual_consistency", "preservation", "Is the rest of the scene unchanged?"),
        ],
        [f"Remove the {obj}", "Add a vase of flowers in its place"],
    )


def mt_medium_1(info):
    if info["group"] == "person":
        return (
            "Change the clothing to all black → Add dramatic side lighting → Convert to high-contrast black and white",
            [
                ("instruction_following", "instruction", "Is the clothing now all black?"),
                ("instruction_following", "instruction", "Is there dramatic side lighting?"),
                ("instruction_following", "instruction", "Is the image in black and white?"),
                ("visual_consistency", "identity", "Is the person recognizable?"),
                ("detail_preservation", "quality", "Is the contrast appropriately high?"),
            ],
            [
                "Change the clothing to all black",
                "Add dramatic side lighting",
                "Convert to high-contrast black and white",
            ],
        )
    obj = _first_obj(info)
    return (
        f"Change the {obj} color to red → Add a spotlight on it → Blur the background",
        [
            ("instruction_following", "instruction", f"Is the {obj} now red?"),
            ("instruction_following", "instruction", "Is there a spotlight effect on it?"),
            ("instruction_following", "instruction", "Is the background blurred?"),
            ("visual_consistency", "structure", f"Is the {obj}'s shape unchanged?"),
        ],
        [f"Change the {obj} color to red", "Add a spotlight on it", "Blur the background"],
    )


def mt_hard_1(info):
    if info["group"] == "person":
        return (
            "Remove all people from the scene → Change the setting to a post-apocalyptic overgrown version → Add a single deer standing in the center → Change to foggy dawn with muted desaturated colors",
            [
                ("instruction_following", "instruction", "Are all original people removed?"),
                (
                    "instruction_following",
                    "instruction",
                    "Does the setting look post-apocalyptic and overgrown?",
                ),
                (
                    "instruction_following",
                    "instruction",
                    "Is a single deer standing in the center?",
                ),
                (
                    "instruction_following",
                    "instruction",
                    "Is the atmosphere foggy dawn with muted colors?",
                ),
                (
                    "visual_consistency",
                    "structure",
                    "Is the original architectural layout recognizable?",
                ),
                ("detail_preservation", "quality", "Does the overgrown vegetation look natural?"),
            ],
            [
                "Remove all people from the scene",
                "Change the setting to a post-apocalyptic overgrown version",
                "Add a single deer standing in the center",
                "Change to foggy dawn with muted desaturated colors",
            ],
        )
    obj = _first_obj(info)
    return (
        f"Remove the {obj} → Change the season to deep winter with snow on all surfaces → Add a person in winter clothing walking in the distance → Add warm lamppost lighting along the path",
        [
            ("instruction_following", "instruction", f"Is the {obj} removed?"),
            ("instruction_following", "instruction", "Is there snow on all surfaces?"),
            (
                "instruction_following",
                "instruction",
                "Is a person in winter clothing visible in the distance?",
            ),
            ("instruction_following", "instruction", "Are lampposts with warm lighting visible?"),
            (
                "visual_consistency",
                "structure",
                "Is the original scene layout preserved under snow?",
            ),
            (
                "detail_preservation",
                "lighting",
                "Do lamppost lights create realistic warm pools on snow?",
            ),
        ],
        [
            f"Remove the {obj}",
            "Change the season to deep winter with snow on all surfaces",
            "Add a person in winter clothing walking in the distance",
            "Add warm lamppost lighting along the path",
        ],
    )


# ═══════════════════════════════════════════════════════════════
# Template registry
# ═══════════════════════════════════════════════════════════════

TEMPLATES = {
    ("color_material", "easy"): [cm_easy_1, cm_easy_2],
    ("color_material", "medium"): [cm_medium_1, cm_medium_2],
    ("color_material", "hard"): [cm_hard_1],
    ("object_add", "easy"): [oa_easy_1],
    ("object_add", "medium"): [oa_medium_1],
    ("object_add", "hard"): [oa_hard_1],
    ("object_remove", "easy"): [or_easy_1],
    ("object_remove", "medium"): [or_medium_1],
    ("object_remove", "hard"): [or_hard_1],
    ("object_replace", "easy"): [rp_easy_1],
    ("object_replace", "medium"): [rp_medium_1],
    ("object_replace", "hard"): [rp_hard_1],
    ("background_change", "easy"): [bg_easy_1],
    ("background_change", "medium"): [bg_medium_1],
    ("background_change", "hard"): [bg_hard_1],
    ("style_transfer", "easy"): [st_easy_1],
    ("style_transfer", "medium"): [st_medium_1],
    ("style_transfer", "hard"): [st_hard_1],
    ("lighting_weather", "easy"): [lw_easy_1],
    ("lighting_weather", "medium"): [lw_medium_1],
    ("lighting_weather", "hard"): [lw_hard_1],
    ("spatial_pose", "easy"): [sp_easy_1],
    ("spatial_pose", "medium"): [sp_medium_1],
    ("spatial_pose", "hard"): [sp_hard_1],
    ("anatomy_identity", "easy"): [ai_easy_1],
    ("anatomy_identity", "medium"): [ai_medium_1],
    ("anatomy_identity", "hard"): [ai_hard_1],
    ("physics", "easy"): [ph_easy_1],
    ("physics", "medium"): [ph_medium_1],
    ("physics", "hard"): [ph_hard_1],
    ("text_editing", "easy"): [te_easy_1],
    ("text_editing", "medium"): [te_medium_1],
    ("text_editing", "hard"): [te_hard_1],
    ("multi_turn", "easy"): [mt_easy_1],
    ("multi_turn", "medium"): [mt_medium_1],
    ("multi_turn", "hard"): [mt_hard_1],
}

CAT_ABBREV = {
    "color_material": "CM",
    "object_add": "OA",
    "object_remove": "OR",
    "object_replace": "RP",
    "background_change": "BG",
    "style_transfer": "ST",
    "lighting_weather": "LW",
    "spatial_pose": "SP",
    "anatomy_identity": "AI",
    "physics": "PH",
    "text_editing": "TE",
    "multi_turn": "MT",
}


def plan_assignments(analysis: dict, taxonomy: dict) -> list[tuple[dict, str, str]]:
    random.seed(42)
    cats = taxonomy["categories"]
    target = taxonomy["targets"]
    total_target = target["total_prompts"]
    diff_targets = {d: int(total_target * f) for d, f in target["difficulty_split"].items()}

    # Track both 2D (cat×diff) and 4D (cat×diff×group) coverage
    coverage = {(c, d): 0 for c in cats for d in ["easy", "medium", "hard"]}
    coverage_4d = {}
    diff_counts = {"easy": 0, "medium": 0, "hard": 0}
    assignments = []

    all_images = list(analysis.values())
    random.shuffle(all_images)
    images_by_group = {}
    for img in all_images:
        images_by_group.setdefault(img["group"], []).append(img)

    min_per_cell = target["min_per_cell"]

    # Pass 1: ensure minimum 2D coverage (3 per cat×diff cell)
    # AND spread across applicable groups
    for cat_id, cat_info in cats.items():
        applicable = cat_info.get("applicable_groups", ["person", "object", "scene", "style"])
        for diff in ["easy", "medium", "hard"]:
            needed = min_per_cell
            # Spread across groups: 1 from each applicable group, then fill
            for g in applicable:
                if needed <= 0:
                    break
                pool = list(images_by_group.get(g, []))
                random.shuffle(pool)
                if pool:
                    img = pool[0]
                    assignments.append((img, cat_id, diff))
                    coverage[(cat_id, diff)] += 1
                    coverage_4d[(cat_id, diff, g)] = coverage_4d.get((cat_id, diff, g), 0) + 1
                    diff_counts[diff] += 1
                    needed -= 1

    # Pass 2: fill to target, prioritizing empty 4D cells
    used_count = {}
    for a in assignments:
        p = a[0]["local_path"]
        used_count[p] = used_count.get(p, 0) + 1

    remaining = list(all_images)
    random.shuffle(remaining)
    ri = 0

    while len(assignments) < total_target:
        if ri >= len(remaining):
            remaining = list(all_images)
            random.shuffle(remaining)
            ri = 0
        img = remaining[ri]
        ri += 1
        group = img["group"]
        applicable_cats = GROUP_CATS.get(group, [])
        if not applicable_cats:
            continue

        # Skip images used too many times
        if used_count.get(img["local_path"], 0) >= 3:
            continue

        # Pick category with least 4D coverage for this group
        def cat_score(c):
            total_cat = sum(coverage[(c, d)] for d in ["easy", "medium", "hard"])
            group_cat = sum(coverage_4d.get((c, d, group), 0) for d in ["easy", "medium", "hard"])
            return (group_cat, total_cat)

        cat_id = min(applicable_cats, key=cat_score)

        # Pick difficulty that needs filling (both globally and for this cat×group)
        def diff_score(d):
            global_fill = diff_counts[d] / max(diff_targets[d], 1)
            cell_fill = coverage_4d.get((cat_id, d, group), 0)
            return (cell_fill, global_fill)

        diff = min(["easy", "medium", "hard"], key=diff_score)

        assignments.append((img, cat_id, diff))
        coverage[(cat_id, diff)] += 1
        coverage_4d[(cat_id, diff, group)] = coverage_4d.get((cat_id, diff, group), 0) + 1
        diff_counts[diff] += 1
        used_count[img["local_path"]] = used_count.get(img["local_path"], 0) + 1

    return assignments


def generate_prompt(img_info: dict, cat_id: str, difficulty: str) -> dict:
    templates = TEMPLATES.get((cat_id, difficulty), [])
    if not templates:
        templates = TEMPLATES.get((cat_id, "medium"), TEMPLATES.get((cat_id, "easy"), []))
    fn = random.choice(templates)
    edit_instruction, raw_atoms, turns = fn(img_info)

    atoms = []
    for i, (dim, atype, question) in enumerate(raw_atoms):
        atoms.append(
            {
                "q_id": f"q{i + 1}",
                "question": question,
                "type": atype,
                "dimension": dim,
            }
        )

    path = img_info["local_path"]
    if path.startswith("prompts/"):
        path = path[len("prompts/") :]

    result = {
        "source_image": path,
        "category": cat_id,
        "difficulty": difficulty,
        "edit_instruction": edit_instruction,
        "atoms": atoms,
    }
    if isinstance(turns, list):
        result["turns"] = turns
    else:
        result["turns"] = turns

    return result


def main(args):
    analysis = json.loads(ANALYSIS_PATH.read_text())
    taxonomy = yaml.safe_load(TAXONOMY_PATH.read_text())

    print(f"Loaded {len(analysis)} image analyses, {len(taxonomy['categories'])} categories")

    assignments = plan_assignments(analysis, taxonomy)
    print(f"Planned {len(assignments)} prompt assignments")

    diff_dist = {}
    cat_dist = {}
    for _, cat, diff in assignments:
        diff_dist[diff] = diff_dist.get(diff, 0) + 1
        cat_dist[cat] = cat_dist.get(cat, 0) + 1
    print(f"Difficulty: {dict(sorted(diff_dist.items()))}")
    print(f"Categories: {dict(sorted(cat_dist.items()))}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting.")
        return

    random.seed(42)
    all_prompts = []
    for img_info, cat_id, diff in assignments:
        p = generate_prompt(img_info, cat_id, diff)
        all_prompts.append(p)

    print(f"\nGenerated {len(all_prompts)} prompts")

    # Split L1/L2
    random.seed(42)
    random.shuffle(all_prompts)
    split_idx = int(len(all_prompts) * 0.70)
    l1 = all_prompts[:split_idx]
    l2 = all_prompts[split_idx:]

    # Assign IDs
    counters = {}
    for p in l1:
        abbrev = CAT_ABBREV.get(p["category"], "XX")
        counters.setdefault(abbrev, 0)
        counters[abbrev] += 1
        p["prompt_id"] = f"L1_{abbrev}_{counters[abbrev]:03d}"
        p["layer"] = 1
        p["sub_category"] = p["category"]

    counters = {}
    for p in l2:
        abbrev = CAT_ABBREV.get(p["category"], "XX")
        counters.setdefault(abbrev, 0)
        counters[abbrev] += 1
        p["prompt_id"] = f"L2_{abbrev}_{counters[abbrev]:03d}"
        p["layer"] = 2
        p["sub_category"] = p["category"]

    L1_OUTPUT.write_text(json.dumps(l1, indent=2))
    L2_OUTPUT.write_text(json.dumps(l2, indent=2))

    print("\nSaved:")
    print(f"  L1 gold:        {len(l1)} prompts -> {L1_OUTPUT.name}")
    print(f"  L2 proprietary: {len(l2)} prompts -> {L2_OUTPUT.name}")

    # Summary stats
    for name, prompts in [("L1", l1), ("L2", l2)]:
        dd = {}
        for p in prompts:
            dd[p["difficulty"]] = dd.get(p["difficulty"], 0) + 1
        print(f"  {name} difficulty: {dict(sorted(dd.items()))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args)
