"""Generate a harder Layer 2 proprietary prompt set.

60 prompts = 20 per sub-category, with atomic binary decompositions pre-written.
For Complex Compositions, difficulty gradient matches the build doc:
  6 easy (3 constraints), 8 medium (4-5), 6 hard (6-8, pushed to 7-8).

Difficulty-bumping rules applied per internal guidance:

NUMERACY
  - baseline of 4+ objects (not 3 like T2I-CompBench++)
  - >= 10 prompts combine multiple counts (e.g. "5 red apples AND 3 green pears")
  - >= 6 prompts use counts >= 7 (20% more than the prior starter's 5)
  - >= 5 of the big-count (>=7) prompts bind a specific attribute to the count
  - >= 3 prompts use ordinal counting ("the 4th bottle from the left is blue")
  - every numeracy question explicitly says "exactly N"

SPATIAL_3D
  - >= 8 prompts use occlusion (object partially hidden behind another)
  - >= 5 prompts describe three depth planes (foreground / middle / background)
  - >= 3 prompts use viewer-relative perspective ("from the camera's viewpoint")
  - trivial "on top of" / "next to" relations deliberately de-emphasized in
    favor of behind / in front of / hidden by / partially occluded

COMPLEX_COMPOSITIONS
  - easy: each prompt must have >= 1 spatial OR numeracy constraint (not pure
    attribute binding)
  - medium: each prompt must have >= 1 numeracy AND >= 1 spatial constraint
  - hard: each prompt must have >= 2 numeracy, >= 1 spatial, >= 1 specific
    color attribute, and >= 1 material/texture attribute. 7-8 constraints
    is the norm here.

Question type enum is extended with "material" for questions about
texture/substance (e.g. "Is the surface wooden?"). Existing downstream code
(aggregator, judge, report) treats the type field as an opaque string so this
is a forward-compatible addition.

A runtime assertion enforces a minimum of 3 atomic questions per prompt
(guards against the L2_SP3_007-class regression).

Output: prompts/layer2_proprietary.json
"""

import json
import re
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_TYPES = {"presence", "attribute", "numeracy", "spatial", "material"}


def _build(prompt_id, sub_cat, difficulty, text, questions):
    return {
        "prompt_id": prompt_id,
        "layer": 2,
        "sub_category": sub_cat,
        "difficulty": difficulty,
        "prompt_text": text,
        "atomic_questions": [
            {"q_id": f"q{i + 1}", "question": q[0], "type": q[1]} for i, q in enumerate(questions)
        ],
    }


# ---------------------------------------------------------------------------
# Numeracy (20 prompts)
# All prompts have min_count >= 4. Counts are bumped into compound / big /
# ordinal buckets per the rules above.
# ---------------------------------------------------------------------------

NUMERACY = [
    # 1 - big_count (8) with attribute binding + compound (3 acorns)
    (
        "Eight red cardinals perched on a snow-covered branch with three small acorns hanging below them",
        [
            ("Are there cardinals in the image?", "presence"),
            ("Are the cardinals red?", "attribute"),
            ("Are there exactly 8 cardinals?", "numeracy"),
            ("Is there a snow-covered branch?", "presence"),
            ("Are the cardinals perched on the branch?", "spatial"),
            ("Are there acorns below the cardinals?", "spatial"),
            ("Are there exactly 3 acorns?", "numeracy"),
        ],
    ),
    # 2 - compound (5 + 4) with distinct attributes
    (
        "Five blue teacups and four green saucers stacked on a white marble countertop",
        [
            ("Are there teacups?", "presence"),
            ("Are the teacups blue?", "attribute"),
            ("Are there exactly 5 teacups?", "numeracy"),
            ("Are there saucers?", "presence"),
            ("Are the saucers green?", "attribute"),
            ("Are there exactly 4 saucers?", "numeracy"),
            ("Is the countertop marble?", "material"),
        ],
    ),
    # 3 - compound + big (7 yellow ducks) with attribute binding
    (
        "Seven yellow rubber ducks and two black swans swimming in a circular fountain",
        [
            ("Are there rubber ducks?", "presence"),
            ("Are the ducks yellow?", "attribute"),
            ("Are there exactly 7 ducks?", "numeracy"),
            ("Are there swans?", "presence"),
            ("Are the swans black?", "attribute"),
            ("Are there exactly 2 swans?", "numeracy"),
            ("Is the fountain circular?", "attribute"),
        ],
    ),
    # 4 - ordinal: 3rd from the left is red
    (
        "A row of six wooden crates where the third crate from the left is painted bright red and the others are unpainted",
        [
            ("Are there wooden crates?", "material"),
            ("Are there exactly 6 crates?", "numeracy"),
            ("Are the crates arranged in a row?", "spatial"),
            ("Is the third crate from the left painted red?", "spatial"),
            ("Are the other crates unpainted?", "attribute"),
        ],
    ),
    # 5 - compound + big (9 candles) with attribute binding
    (
        "Four tall copper candlesticks holding nine dripping white candles on a long oak dining table",
        [
            ("Are there candlesticks?", "presence"),
            ("Are the candlesticks copper?", "material"),
            ("Are there exactly 4 candlesticks?", "numeracy"),
            ("Are there candles in the candlesticks?", "spatial"),
            ("Are the candles white?", "attribute"),
            ("Are there exactly 9 candles total?", "numeracy"),
            ("Is the table oak?", "material"),
        ],
    ),
    # 6 - big (12) with compound attribute (orange monarch)
    (
        "Twelve orange monarch butterflies resting on the petals of a single large white hibiscus flower",
        [
            ("Are there butterflies?", "presence"),
            ("Are the butterflies orange?", "attribute"),
            ("Are there exactly 12 butterflies?", "numeracy"),
            ("Is there a hibiscus flower?", "presence"),
            ("Is the hibiscus white?", "attribute"),
            ("Is there exactly 1 hibiscus flower?", "numeracy"),
            ("Are the butterflies resting on the petals?", "spatial"),
        ],
    ),
    # 7 - compound (6 + 3)
    (
        "Six smiling jack-o-lanterns and three black crows sitting on a wooden fence at dusk",
        [
            ("Are there jack-o-lanterns?", "presence"),
            ("Are there exactly 6 jack-o-lanterns?", "numeracy"),
            ("Are there crows?", "presence"),
            ("Are the crows black?", "attribute"),
            ("Are there exactly 3 crows?", "numeracy"),
            ("Are all of them on a wooden fence?", "spatial"),
            ("Is the fence wooden?", "material"),
        ],
    ),
    # 8 - compound + big (8 puppies)
    (
        "Eight golden retriever puppies and two adult golden retrievers lying together on a patterned rug",
        [
            ("Are there puppies?", "presence"),
            ("Are the puppies golden retrievers?", "attribute"),
            ("Are there exactly 8 puppies?", "numeracy"),
            ("Are there adult dogs?", "presence"),
            ("Are there exactly 2 adult dogs?", "numeracy"),
            ("Are all of them lying on the rug?", "spatial"),
            ("Does the rug have a pattern?", "attribute"),
        ],
    ),
    # 9 - ordinal: 5th from the front wears a red bow tie
    (
        "A line of seven marching penguins where the fifth penguin from the front wears a red bow tie and the others do not",
        [
            ("Are there penguins?", "presence"),
            ("Are there exactly 7 penguins?", "numeracy"),
            ("Are the penguins in a line?", "spatial"),
            ("Is the fifth penguin from the front wearing a red bow tie?", "spatial"),
            ("Are the other penguins without bow ties?", "attribute"),
        ],
    ),
    # 10 - triple compound (10 + 6 + 4), big
    (
        "Ten identical silver forks, six identical knives, and four identical spoons arranged on a white linen tablecloth",
        [
            ("Are there forks?", "presence"),
            ("Are the forks silver?", "material"),
            ("Are there exactly 10 forks?", "numeracy"),
            ("Are there exactly 6 knives?", "numeracy"),
            ("Are there exactly 4 spoons?", "numeracy"),
            ("Is the tablecloth white linen?", "material"),
        ],
    ),
    # 11 - compound (4 + 4), with attributes
    (
        "Four toddlers wearing red sneakers and four identical teddy bears sitting in a semicircle on a pastel blue rug",
        [
            ("Are there toddlers?", "presence"),
            ("Are there exactly 4 toddlers?", "numeracy"),
            ("Are the toddlers wearing red sneakers?", "attribute"),
            ("Are there teddy bears?", "presence"),
            ("Are there exactly 4 teddy bears?", "numeracy"),
            ("Are all of them arranged in a semicircle?", "spatial"),
            ("Is the rug pastel blue?", "attribute"),
        ],
    ),
    # 12 - triple compound (7 + 5 + 3), big with attribute binding
    (
        "Seven red tulips, five white tulips, and three purple tulips bundled in a clear glass vase",
        [
            ("Are there tulips?", "presence"),
            ("Are there exactly 7 red tulips?", "numeracy"),
            ("Are there exactly 5 white tulips?", "numeracy"),
            ("Are there exactly 3 purple tulips?", "numeracy"),
            ("Is the vase clear glass?", "material"),
            ("Are the tulips inside the vase?", "spatial"),
        ],
    ),
    # 13 - big (9) with attribute binding (juggling pins)
    (
        "Nine orange juggling pins mid-air above a smiling clown wearing a polka-dot costume",
        [
            ("Are there juggling pins?", "presence"),
            ("Are the pins orange?", "attribute"),
            ("Are there exactly 9 pins?", "numeracy"),
            ("Is there a clown?", "presence"),
            ("Is the clown's costume polka-dotted?", "attribute"),
            ("Are the pins above the clown?", "spatial"),
        ],
    ),
    # 14 - ordinal: 2nd from the bottom is thicker
    (
        "Four stacked books on a shelf where the second book from the bottom is visibly thicker than the other three",
        [
            ("Are there books on a shelf?", "presence"),
            ("Are there exactly 4 books?", "numeracy"),
            ("Are the books stacked?", "spatial"),
            ("Is the second book from the bottom thicker than the others?", "spatial"),
            ("Are the other books similar in thickness to each other?", "attribute"),
        ],
    ),
    # 15 - compound (5 + 4)
    (
        "Five tall wine glasses and four short shot glasses lined up on a mirrored bar",
        [
            ("Are there wine glasses?", "presence"),
            ("Are there exactly 5 wine glasses?", "numeracy"),
            ("Are the wine glasses tall?", "attribute"),
            ("Are there shot glasses?", "presence"),
            ("Are there exactly 4 shot glasses?", "numeracy"),
            ("Are the shot glasses short?", "attribute"),
            ("Is the bar surface mirrored?", "material"),
        ],
    ),
    # 16 - compound + big (8 pawns)
    (
        "Eight identical chess pawns and two kings arranged on a black-and-white checkered board",
        [
            ("Are there chess pawns?", "presence"),
            ("Are there exactly 8 pawns?", "numeracy"),
            ("Are there kings?", "presence"),
            ("Are there exactly 2 kings?", "numeracy"),
            ("Is the board checkered black and white?", "attribute"),
            ("Are all pieces on the board?", "spatial"),
        ],
    ),
    # 17 - compound (4 + 3)
    (
        "Four children on bicycles and three children on scooters racing down a park path",
        [
            ("Are there children on bicycles?", "presence"),
            ("Are there exactly 4 children on bicycles?", "numeracy"),
            ("Are there children on scooters?", "presence"),
            ("Are there exactly 3 children on scooters?", "numeracy"),
            ("Are they all on a park path?", "spatial"),
        ],
    ),
    # 18 - compound + big (7 + 6)
    (
        "Seven green apples and six red apples arranged in a pyramid inside a woven basket",
        [
            ("Are there apples?", "presence"),
            ("Are there exactly 7 green apples?", "numeracy"),
            ("Are there exactly 6 red apples?", "numeracy"),
            ("Are the apples arranged in a pyramid?", "spatial"),
            ("Is the basket woven?", "material"),
            ("Are all the apples inside the basket?", "spatial"),
        ],
    ),
    # 19 - ordinal: 4th tipped in gold
    (
        "Six wooden pencils of increasing length lying parallel on a white sheet, with the fourth pencil from the left tipped in gold",
        [
            ("Are there pencils?", "presence"),
            ("Are there exactly 6 pencils?", "numeracy"),
            ("Are the pencils wooden?", "material"),
            ("Are the pencils parallel?", "spatial"),
            ("Does each pencil increase in length from left to right?", "spatial"),
            ("Is the fourth pencil from the left tipped in gold?", "spatial"),
        ],
    ),
    # 20 - compound + big (9 + 4) with attribute binding
    (
        "Nine brown horses and four white horses grazing in a fenced meadow at midday",
        [
            ("Are there brown horses?", "presence"),
            ("Are there exactly 9 brown horses?", "numeracy"),
            ("Are there white horses?", "presence"),
            ("Are there exactly 4 white horses?", "numeracy"),
            ("Are they grazing?", "presence"),
            ("Is there a fence around the meadow?", "spatial"),
        ],
    ),
]


# ---------------------------------------------------------------------------
# 3D-Spatial (20 prompts) - heavy on occlusion, depth planes, viewer angle
# ---------------------------------------------------------------------------

SPATIAL_3D = [
    # 1 - occlusion + three planes (watch FG, book MG, bookshelf BG)
    (
        "A silver pocket watch partially hidden behind an open leather book on a desk, with a tall bookshelf filled with volumes receding into the background",
        [
            ("Is there a silver pocket watch?", "material"),
            ("Is there an open leather book?", "material"),
            ("Is the pocket watch partially hidden behind the book?", "spatial"),
            ("Is there a bookshelf behind the desk?", "spatial"),
            ("Does the bookshelf appear farther from the camera than the book?", "spatial"),
        ],
    ),
    # 2 - occlusion (mouse behind chair leg)
    (
        "A small gray mouse hiding behind the leg of a wooden chair with only its tail and whiskers visible",
        [
            ("Is there a gray mouse?", "presence"),
            ("Is there a wooden chair?", "material"),
            ("Is the mouse mostly hidden behind the chair leg?", "spatial"),
            ("Are only the mouse's tail and whiskers visible?", "spatial"),
        ],
    ),
    # 3 - three depth planes (canoe FG, fisherman MG, trees BG)
    (
        "A red canoe in the foreground, a fisherman standing knee-deep in water in the middle distance, and a line of pine trees on the far shore",
        [
            ("Is there a red canoe?", "presence"),
            ("Is the canoe in the foreground?", "spatial"),
            ("Is there a fisherman in the middle distance?", "spatial"),
            ("Are there pine trees on the far shore?", "spatial"),
            ("Do the pine trees appear farthest from the camera?", "spatial"),
        ],
    ),
    # 4 - viewer-relative (three objects ordered by depth from camera)
    (
        "From the camera's viewpoint a crystal wine glass is closest, a lit candle is in the middle, and a ceramic vase is farthest, all on the same dinner table",
        [
            ("Is there a wine glass?", "presence"),
            ("Is there a lit candle?", "presence"),
            ("Is there a ceramic vase?", "material"),
            ("Is the wine glass closer to the camera than the candle?", "spatial"),
            ("Is the candle closer to the camera than the vase?", "spatial"),
        ],
    ),
    # 5 - occlusion (child behind door, half face visible)
    (
        "A child peeking out from behind a tall wooden door with only half their face visible",
        [
            ("Is there a child?", "presence"),
            ("Is there a tall wooden door?", "material"),
            ("Is the child mostly behind the door?", "spatial"),
            ("Is only half of the child's face visible?", "spatial"),
        ],
    ),
    # 6 - occlusion + three planes (cat MG hidden, mouse FG, fireplace BG)
    (
        "A tabby cat crouched behind a sofa stalking a rubber mouse on a rug in the foreground, with a glass-fronted fireplace at the far back of the room",
        [
            ("Is there a tabby cat?", "presence"),
            ("Is the cat crouched behind a sofa?", "spatial"),
            ("Is there a rubber mouse on a rug in the foreground?", "spatial"),
            ("Is there a glass-fronted fireplace at the back of the room?", "spatial"),
            ("Does the fireplace appear farther from the camera than the sofa?", "spatial"),
        ],
    ),
    # 7 - viewer-relative (forced perspective: statue appears larger than building)
    (
        "From the photographer's angle a small bronze statue in the foreground appears larger than the marble government building behind it, due to forced perspective",
        [
            ("Is there a bronze statue?", "material"),
            ("Is there a marble building behind the statue?", "material"),
            ("Is the statue in the foreground?", "spatial"),
            ("Does the statue appear larger than the building from this angle?", "spatial"),
        ],
    ),
    # 8 - occlusion (submarine partially hidden by fish)
    (
        "A yellow submarine partially obscured by a school of blue fish swimming in front of it underwater",
        [
            ("Is there a yellow submarine?", "presence"),
            ("Is there a school of blue fish?", "presence"),
            ("Are the fish in front of the submarine?", "spatial"),
            ("Is the submarine partially hidden by the fish?", "spatial"),
        ],
    ),
    # 9 - three planes (child FG, carousel MG, ferris wheel BG)
    (
        "A child holding a red balloon in the foreground, a carousel turning in the middle distance, and a Ferris wheel silhouetted against the evening sky far behind",
        [
            ("Is there a child holding a red balloon?", "attribute"),
            ("Is there a carousel?", "presence"),
            ("Is there a Ferris wheel?", "presence"),
            ("Is the child closest to the camera?", "spatial"),
            ("Is the carousel between the child and the Ferris wheel?", "spatial"),
            ("Is the Ferris wheel farthest from the camera?", "spatial"),
        ],
    ),
    # 10 - occlusion (thief behind brick pillar)
    (
        "A thief in a black mask peeking out from behind a brick pillar while watching a jeweler's shop window across the street",
        [
            ("Is there a thief in a black mask?", "attribute"),
            ("Is there a brick pillar?", "material"),
            ("Is the thief partially hidden behind the pillar?", "spatial"),
            ("Is there a jeweler's shop window?", "presence"),
            ("Is the shop window in the direction the thief is looking?", "spatial"),
        ],
    ),
    # 11 - occlusion (black queen hidden behind taller white rook on chessboard)
    (
        "A black chess queen concealed behind a taller white rook on a wooden chessboard, from a low angle",
        [
            ("Are there chess pieces?", "presence"),
            ("Is there a black queen?", "attribute"),
            ("Is there a white rook?", "attribute"),
            ("Is the black queen behind the rook?", "spatial"),
            ("Is the rook taller than the queen?", "spatial"),
            ("Is the chessboard wooden?", "material"),
        ],
    ),
    # 12 - viewer-relative + three planes (bee near camera, daisy MG, meadow BG)
    (
        "From a macro camera angle a bee hovers in the foreground, a single blooming daisy sits in the middle distance, and a wildflower meadow stretches toward the horizon behind them",
        [
            ("Is there a bee?", "presence"),
            ("Does the bee appear very close to the camera?", "spatial"),
            ("Is there a blooming daisy in the middle distance?", "spatial"),
            ("Is there a meadow stretching to the horizon?", "spatial"),
            ("Does the meadow appear farthest from the camera?", "spatial"),
        ],
    ),
    # 13 - occlusion (figure behind curtain, only outline visible)
    (
        "A ghostly figure mostly hidden behind a gauzy white curtain, with only the silhouette and outline visible through the fabric",
        [
            ("Is there a figure?", "presence"),
            ("Is there a white curtain?", "attribute"),
            ("Is the curtain gauzy?", "material"),
            ("Is the figure behind the curtain?", "spatial"),
            ("Is only an outline of the figure visible?", "spatial"),
        ],
    ),
    # 14 - occlusion + three planes (carriage behind hedges, castle far BG)
    (
        "A horse-drawn carriage partially blocked by a row of tall hedges in the middle distance, with a castle rising in the far background",
        [
            ("Is there a horse-drawn carriage?", "presence"),
            ("Is there a row of tall hedges?", "presence"),
            ("Are the hedges in front of the carriage?", "spatial"),
            ("Is the carriage partially hidden by the hedges?", "spatial"),
            ("Is there a castle in the far background?", "spatial"),
        ],
    ),
    # 15 - viewer-relative (driver's POV)
    (
        "From the driver's seat of a car the steering wheel is in the foreground and a row of cars on the road ahead appears to recede into the distance",
        [
            ("Is there a steering wheel?", "presence"),
            ("Does the steering wheel appear in the foreground?", "spatial"),
            ("Is there a road ahead with cars on it?", "spatial"),
            ("Do the cars appear to recede into the distance?", "spatial"),
        ],
    ),
    # 16 - occlusion (kitten inside cardboard box)
    (
        "A small kitten hiding inside an open cardboard box with only its ears sticking out above the rim",
        [
            ("Is there a kitten?", "presence"),
            ("Is there an open cardboard box?", "material"),
            ("Is the kitten mostly inside the box?", "spatial"),
            ("Are the kitten's ears visible above the rim?", "spatial"),
        ],
    ),
    # 17 - occlusion (jogger behind bus)
    (
        "A jogger mostly obscured by a passing city bus, with only the jogger's legs visible below the bus on the street",
        [
            ("Is there a jogger?", "presence"),
            ("Is there a city bus?", "presence"),
            ("Is the jogger behind the bus from the camera's view?", "spatial"),
            ("Are only the jogger's legs visible below the bus?", "spatial"),
        ],
    ),
    # 18 - three planes (pinecone FG, deer MG, mountains BG)
    (
        "A pinecone on a moss-covered log in the foreground, a deer standing alert in a clearing in the middle distance, and snow-capped mountains in the far background",
        [
            ("Is there a pinecone on a log?", "presence"),
            ("Is the log moss-covered?", "attribute"),
            ("Is there a deer in the middle distance?", "spatial"),
            ("Are there snow-capped mountains in the far distance?", "spatial"),
            ("Does the deer appear closer to the camera than the mountains?", "spatial"),
        ],
    ),
    # 19 - occlusion (singer partly hidden by amp and mic stand)
    (
        "A singer on stage with their lower body obscured by a large stage amp and part of their face hidden behind a tall microphone stand",
        [
            ("Is there a singer on a stage?", "presence"),
            ("Is there a stage amp in front of the singer?", "spatial"),
            ("Is there a tall microphone stand?", "presence"),
            ("Is the singer's lower body hidden by the amp?", "spatial"),
            ("Is part of the singer's face hidden behind the microphone stand?", "spatial"),
        ],
    ),
    # 20 - three planes (bowl FG, woman MG, garden through window BG)
    (
        "A bowl of fruit in the foreground, a seated woman reading a book in the middle of the room, and a large bay window showing a garden beyond in the background",
        [
            ("Is there a bowl of fruit?", "presence"),
            ("Is the bowl in the foreground?", "spatial"),
            ("Is there a seated woman reading?", "presence"),
            ("Is there a bay window behind the woman?", "spatial"),
            ("Is a garden visible through the window?", "spatial"),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Complex Compositions (20 prompts; 6 easy, 8 medium, 6 hard)
# ---------------------------------------------------------------------------

# EASY: 3 constraints; each must have >= 1 spatial OR numeracy (no pure attribute binding)
COMPLEX_EASY = [
    (
        "A blue bicycle leaning against a white brick wall next to a red fire hydrant",
        [
            ("Is there a blue bicycle?", "attribute"),
            ("Is the wall white and made of brick?", "material"),
            ("Is the bike next to a red fire hydrant?", "spatial"),
        ],
    ),  # <- spatial
    (
        "Two identical brown owls perched on a bare branch at dusk",
        [
            ("Are there owls?", "presence"),
            ("Are there exactly 2 owls?", "numeracy"),  # <- numeracy
            ("Are the owls perched on a bare branch?", "spatial"),
        ],
    ),
    (
        "A porcelain teapot on a wooden tray beside a steaming cup",
        [
            ("Is there a porcelain teapot?", "material"),
            ("Is the teapot on a wooden tray?", "spatial"),  # <- spatial
            ("Is there a steaming cup beside the teapot?", "spatial"),
        ],
    ),
    (
        "Three fresh lemons next to a glass pitcher of water on a marble counter",
        [
            ("Are there exactly 3 lemons?", "numeracy"),  # <- numeracy
            ("Is there a glass pitcher of water beside the lemons?", "spatial"),
            ("Is the counter marble?", "material"),
        ],
    ),
    (
        "A golden retriever lying on a red carpet in front of a stone fireplace",
        [
            ("Is there a golden retriever?", "attribute"),
            ("Is the dog lying on a red carpet?", "spatial"),  # <- spatial
            ("Is the fireplace made of stone?", "material"),
        ],
    ),
    (
        "A straw sunhat hanging on a metal hook above a wooden bench",
        [
            ("Is there a straw sunhat?", "material"),
            ("Is the sunhat hanging on a metal hook?", "spatial"),  # <- spatial
            ("Is the bench directly below the sunhat?", "spatial"),
        ],
    ),
]

# MEDIUM: 4-5 constraints; each must have >= 1 numeracy AND >= 1 spatial
COMPLEX_MEDIUM = [
    (
        "A chef holding three red tomatoes over a wooden cutting board in a bright kitchen next to a white stove",
        [
            ("Is there a chef?", "presence"),
            ("Are there exactly 3 red tomatoes?", "numeracy"),  # num
            ("Is the chef holding the tomatoes above a cutting board?", "spatial"),  # spat
            ("Is the cutting board wooden?", "material"),
            ("Is there a white stove beside the chef?", "spatial"),
        ],
    ),
    (
        "Four schoolchildren in yellow raincoats walking through a puddle with a single red umbrella held above them",
        [
            ("Are there children?", "presence"),
            ("Are there exactly 4 children?", "numeracy"),  # num
            ("Are the children wearing yellow raincoats?", "attribute"),
            ("Are the children walking through a puddle?", "spatial"),  # spat
            ("Is a red umbrella held above the children?", "spatial"),
        ],
    ),
    (
        "A woman in a blue silk dress standing in front of two tall marble columns with a single red rose at her feet",
        [
            ("Is the woman's dress blue silk?", "material"),
            ("Are there exactly 2 marble columns behind the woman?", "numeracy"),  # num + spat
            ("Is the woman in front of the columns?", "spatial"),  # spat
            ("Is there exactly 1 red rose at the woman's feet?", "numeracy"),
            ("Is the rose red?", "attribute"),
        ],
    ),
    (
        "An old fisherman with a gray beard holding a net containing five silver fish while standing knee-deep in green water",
        [
            ("Is there an old fisherman with a gray beard?", "attribute"),
            ("Is the fisherman holding a net?", "presence"),
            ("Are there exactly 5 silver fish in the net?", "numeracy"),  # num
            ("Is the fisherman knee-deep in water?", "spatial"),  # spat
            ("Is the water green?", "attribute"),
        ],
    ),
    (
        "A baker placing three loaves of bread onto a wooden rack in a flour-dusted kitchen",
        [
            ("Is there a baker?", "presence"),
            ("Are there exactly 3 loaves of bread?", "numeracy"),  # num
            ("Is the rack made of wood?", "material"),
            ("Is the baker placing the loaves onto the rack?", "spatial"),  # spat
            ("Is there visible flour dust in the kitchen?", "attribute"),
        ],
    ),
    (
        "A young girl with two long braids holding four yellow balloons in a park at golden hour",
        [
            ("Is there a young girl?", "presence"),
            ("Does the girl have exactly 2 long braids?", "numeracy"),  # num
            ("Is the girl holding balloons?", "presence"),
            ("Are there exactly 4 yellow balloons?", "numeracy"),
            ("Is the girl in a park?", "spatial"),
        ],
    ),  # spat
    (
        "A musician sitting on a wooden stool playing an acoustic guitar with five copper-colored strings visible",
        [
            ("Is there a musician?", "presence"),
            ("Is the musician seated on a wooden stool?", "spatial"),  # spat + material
            ("Is the guitar acoustic?", "attribute"),
            ("Are there exactly 5 strings visible on the guitar?", "numeracy"),  # num
            ("Are the strings copper-colored?", "attribute"),
        ],
    ),
    (
        "A potter shaping a clay vase on a spinning wheel with three finished vases on a shelf behind him",
        [
            ("Is there a potter?", "presence"),
            ("Is the potter shaping a vase?", "presence"),
            ("Is the vase material clay?", "material"),
            ("Are there exactly 3 finished vases on the shelf?", "numeracy"),  # num
            ("Is the shelf behind the potter?", "spatial"),
        ],
    ),  # spat
]

# HARD: 7-8 constraints; each must have >= 2 numeracy, >= 1 spatial, >= 1 color, >= 1 material
COMPLEX_HARD = [
    # 1
    (
        "A Victorian parlor with a brass chandelier hanging from the ceiling, a red velvet sofa holding four lace pillows, three gilt-framed portraits mounted on the wall behind the sofa, and two taxidermy peacocks on a mahogany side table",
        [
            ("Is there a brass chandelier hanging from the ceiling?", "material"),  # material
            ("Is there a red velvet sofa?", "attribute"),  # color (red)
            ("Is the sofa upholstered in velvet?", "material"),
            ("Are there exactly 4 lace pillows on the sofa?", "numeracy"),  # num 1
            ("Are there exactly 3 gilt-framed portraits on the wall?", "numeracy"),  # num 2
            ("Are the portraits on the wall behind the sofa?", "spatial"),  # spatial
            ("Are there exactly 2 taxidermy peacocks on the side table?", "numeracy"),
            ("Is the side table made of mahogany?", "material"),
        ],
    ),
    # 2
    (
        "A young knight in polished silver armor riding a brown horse across a stone bridge, holding a red banner with a golden lion on it, with five crows flying overhead and two gray wolves running behind the horse",
        [
            (
                "Is there a knight in polished silver armor?",
                "material",
            ),  # material + color (silver)
            ("Is the horse brown?", "attribute"),  # color
            ("Is the bridge made of stone?", "material"),
            ("Is the knight holding a red banner?", "attribute"),
            ("Is there a golden lion design on the banner?", "attribute"),
            ("Are there exactly 5 crows flying overhead?", "numeracy"),  # num 1
            ("Are there exactly 2 gray wolves behind the horse?", "numeracy"),  # num 2
            ("Are the wolves behind the horse?", "spatial"),
        ],
    ),  # spatial
    # 3
    (
        "A wooden-floored greenhouse with three tall terracotta pots of red geraniums lined up on the left, four identical ceramic pots of yellow marigolds on the right, a marble bust in the center, and a wrought-iron watering can hanging from a rusted hook",
        [
            ("Is the greenhouse floor wooden?", "material"),  # material
            (
                "Are there exactly 3 terracotta pots of red geraniums?",
                "numeracy",
            ),  # num 1 + color (red)
            ("Are the geranium pots on the left side of the greenhouse?", "spatial"),  # spatial
            (
                "Are there exactly 4 ceramic pots of yellow marigolds?",
                "numeracy",
            ),  # num 2 + color (yellow)
            ("Are the marigold pots on the right side of the greenhouse?", "spatial"),
            ("Is there a marble bust in the center?", "material"),
            ("Is there a wrought-iron watering can?", "material"),
            ("Is the hook rusted?", "attribute"),
        ],
    ),
    # 4
    (
        "A bustling farmers market stall with six wicker baskets of golden oranges on the top shelf, three wooden crates of red apples on the middle shelf, two tin buckets of green pears on the bottom shelf, and a chalkboard sign reading FRESH in the foreground",
        [
            (
                "Are there exactly 6 wicker baskets of oranges on the top shelf?",
                "numeracy",
            ),  # num 1 + material (wicker) + spatial (top)
            ("Are the oranges golden in color?", "attribute"),  # color
            (
                "Are there exactly 3 wooden crates of red apples on the middle shelf?",
                "numeracy",
            ),  # num 2 + material (wooden) + color
            ("Are the apples on the middle shelf?", "spatial"),  # spatial
            ("Are there exactly 2 tin buckets of green pears on the bottom shelf?", "numeracy"),
            ("Are the pear buckets made of tin?", "material"),
            ("Is there a chalkboard sign reading FRESH?", "attribute"),
            ("Is the chalkboard sign in the foreground?", "spatial"),
        ],
    ),
    # 5
    (
        "A modern kitchen with a black granite countertop holding four copper saucepans of decreasing size, a wooden knife block with five knives visible, two crystal wine glasses to the right, and a silver tray of green grapes in the center",
        [
            ("Is there a black granite countertop?", "attribute"),  # color (black)
            ("Is the countertop made of granite?", "material"),  # material
            ("Are there exactly 4 copper saucepans?", "numeracy"),  # num 1
            ("Do the saucepans decrease in size from left to right?", "spatial"),  # spatial
            ("Are there exactly 5 knives in a wooden knife block?", "numeracy"),  # num 2
            ("Are there exactly 2 crystal wine glasses to the right?", "numeracy"),
            ("Is the tray silver?", "attribute"),
            ("Is the tray in the center of the countertop?", "spatial"),
        ],
    ),
    # 6
    (
        "A stone courtyard at twilight with three bronze statues standing in a triangle around a central fountain, four arched windows glowing yellow on the far wall, and two black cats sitting on opposite low walls",
        [
            ("Is the courtyard made of stone?", "material"),  # material
            ("Are there exactly 3 bronze statues?", "numeracy"),  # num 1 + material (bronze)
            (
                "Are the statues arranged in a triangle around a central fountain?",
                "spatial",
            ),  # spatial
            ("Are there exactly 4 arched windows on the far wall?", "numeracy"),  # num 2
            ("Are the windows glowing yellow?", "attribute"),  # color
            ("Are the windows behind the statues?", "spatial"),
            ("Are there exactly 2 black cats?", "numeracy"),
            ("Are the two cats on opposite walls?", "spatial"),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_COLOR_WORDS = {
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "black",
    "white",
    "gray",
    "grey",
    "brown",
    "gold",
    "golden",
    "silver",
    "copper",
    "bronze",
}


def _count_type(questions, t):
    return sum(1 for q in questions if q["type"] == t)


def _numeracy_counts(prompt):
    """Return the list of integers extracted from 'exactly N' in numeracy qs."""
    nums = []
    for q in prompt["atomic_questions"]:
        if q["type"] != "numeracy":
            continue
        m = re.search(r"\bexactly\s+(\d+)", q["question"], re.IGNORECASE)
        if m:
            nums.append(int(m.group(1)))
    return nums


def _is_ordinal(prompt):
    txt = prompt["prompt_text"].lower()
    ordinal_words = (
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
    )
    if any(f"{w} " in txt for w in ordinal_words) and "from the" in txt:
        return True
    return bool(re.search(r"\b\d+(?:st|nd|rd|th)\b", txt))


def _has_color_attribute(prompt):
    """True if any atomic question names a specific color. We scan all types
    (not just 'attribute') because a numeracy question like
    'Are there exactly 3 red geraniums?' also binds a color constraint."""
    for q in prompt["atomic_questions"]:
        toks = re.findall(r"[a-z]+", q["question"].lower())
        if any(tok in _COLOR_WORDS for tok in toks):
            return True
    return False


def _validate_question_shape(prompts):
    """Enforce the minimum question count + valid type enum per prompt."""
    bad = []
    for p in prompts:
        n = len(p["atomic_questions"])
        if n < 3:
            bad.append(f"{p['prompt_id']} has only {n} questions (min 3)")
        for q in p["atomic_questions"]:
            if q["type"] not in VALID_TYPES:
                bad.append(
                    f"{p['prompt_id']} has invalid question type "
                    f"{q['type']!r} (valid: {sorted(VALID_TYPES)})"
                )
    if bad:
        raise AssertionError("Prompt validation failed:\n  " + "\n  ".join(bad))


def _validate_cohort_targets(prompts):
    """Enforce the per-sub-category difficulty targets documented above."""
    failures = []

    numeracy = [p for p in prompts if p["sub_category"] == "numeracy"]
    [p for p in prompts if p["sub_category"] == "spatial_3d"]
    cmp_easy = [
        p
        for p in prompts
        if p["sub_category"] == "complex_compositions" and p["difficulty"] == "easy"
    ]
    cmp_medium = [
        p
        for p in prompts
        if p["sub_category"] == "complex_compositions" and p["difficulty"] == "medium"
    ]
    cmp_hard = [
        p
        for p in prompts
        if p["sub_category"] == "complex_compositions" and p["difficulty"] == "hard"
    ]

    # --- Numeracy targets ---
    # (a) every numeracy prompt's max exactly-N is >= 4
    for p in numeracy:
        nums = _numeracy_counts(p)
        if not nums:
            failures.append(f"{p['prompt_id']}: no 'exactly N' question")
        elif max(nums) < 4:
            failures.append(f"{p['prompt_id']}: max count {max(nums)} < 4 minimum")

    # (b) >= 10 compound prompts (2+ numeracy questions)
    compound = [p for p in numeracy if _count_type(p["atomic_questions"], "numeracy") >= 2]
    if len(compound) < 10:
        failures.append(f"numeracy: only {len(compound)} compound prompts, need >= 10")

    # (c) >= 6 big-count prompts (max count >= 7)
    big = [p for p in numeracy if _numeracy_counts(p) and max(_numeracy_counts(p)) >= 7]
    if len(big) < 6:
        failures.append(f"numeracy: only {len(big)} prompts with count >= 7, need >= 6")

    # (d) >= 5 big-count prompts that also bind a color/material attribute to the count
    big_with_attr = [
        p
        for p in big
        if _has_color_attribute(p) or _count_type(p["atomic_questions"], "material") >= 1
    ]
    if len(big_with_attr) < 5:
        failures.append(
            f"numeracy: only {len(big_with_attr)} big-count prompts with attribute binding, need >= 5"
        )

    # (e) >= 3 ordinal prompts
    ordinal = [p for p in numeracy if _is_ordinal(p)]
    if len(ordinal) < 3:
        failures.append(f"numeracy: only {len(ordinal)} ordinal prompts, need >= 3")

    # --- Complex easy: each must have >= 1 spatial or numeracy ---
    for p in cmp_easy:
        if (
            _count_type(p["atomic_questions"], "spatial") == 0
            and _count_type(p["atomic_questions"], "numeracy") == 0
        ):
            failures.append(f"{p['prompt_id']}: easy prompt has no spatial/numeracy constraint")

    # --- Complex medium: each must have >= 1 numeracy AND >= 1 spatial ---
    for p in cmp_medium:
        nq = _count_type(p["atomic_questions"], "numeracy")
        sq = _count_type(p["atomic_questions"], "spatial")
        if nq < 1 or sq < 1:
            failures.append(
                f"{p['prompt_id']}: medium prompt needs >=1 numeracy AND >=1 spatial "
                f"(got num={nq}, spat={sq})"
            )

    # --- Complex hard: >= 2 numeracy, >= 1 spatial, >= 1 color attr, >= 1 material ---
    for p in cmp_hard:
        nq = _count_type(p["atomic_questions"], "numeracy")
        sq = _count_type(p["atomic_questions"], "spatial")
        mq = _count_type(p["atomic_questions"], "material")
        cq = 1 if _has_color_attribute(p) else 0
        if nq < 2 or sq < 1 or mq < 1 or cq < 1:
            failures.append(
                f"{p['prompt_id']}: hard prompt needs >=2 num, >=1 spat, >=1 color-attr, >=1 material "
                f"(got num={nq}, spat={sq}, mat={mq}, color={cq})"
            )

    if failures:
        raise AssertionError("Cohort target validation failed:\n  " + "\n  ".join(failures))


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------


def generate_all():
    out = []
    for i, (text, qs) in enumerate(NUMERACY, start=1):
        out.append(_build(f"L2_NUM_{i:03d}", "numeracy", "medium", text, qs))
    for i, (text, qs) in enumerate(SPATIAL_3D, start=1):
        out.append(_build(f"L2_SP3_{i:03d}", "spatial_3d", "medium", text, qs))
    i = 1
    for text, qs in COMPLEX_EASY:
        out.append(_build(f"L2_CMP_{i:03d}", "complex_compositions", "easy", text, qs))
        i += 1
    for text, qs in COMPLEX_MEDIUM:
        out.append(_build(f"L2_CMP_{i:03d}", "complex_compositions", "medium", text, qs))
        i += 1
    for text, qs in COMPLEX_HARD:
        out.append(_build(f"L2_CMP_{i:03d}", "complex_compositions", "hard", text, qs))
        i += 1

    _validate_question_shape(out)
    _validate_cohort_targets(out)
    return out


def _print_numeracy_distribution(prompts):
    numeracy = [p for p in prompts if p["sub_category"] == "numeracy"]
    buckets = Counter()
    compound_count = 0
    big_with_attr_count = 0
    ordinal_count = 0
    for p in numeracy:
        nums = _numeracy_counts(p)
        mx = max(nums) if nums else 0
        if mx >= 7:
            buckets["7+"] += 1
        elif mx in (4, 5, 6):
            buckets[str(mx)] += 1
        else:
            buckets[f"<{4}"] += 1
        if _count_type(p["atomic_questions"], "numeracy") >= 2:
            compound_count += 1
        if mx >= 7 and (
            _has_color_attribute(p) or _count_type(p["atomic_questions"], "material") >= 1
        ):
            big_with_attr_count += 1
        if _is_ordinal(p):
            ordinal_count += 1
    print("Numeracy count distribution (max 'exactly N' per prompt):")
    for k in ("4", "5", "6", "7+"):
        print(f"  max = {k:<3}: {buckets.get(k, 0):2d} prompts")
    print(f"  compound (>= 2 numeracy questions):           {compound_count:2d} prompts")
    print(f"  big-count (>= 7) WITH attribute binding:      {big_with_attr_count:2d} prompts")
    print(f"  ordinal counting prompts:                     {ordinal_count:2d} prompts")


def main():
    prompts = generate_all()
    path = Path(__file__).resolve().parent.parent / "prompts" / "layer2_proprietary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"Wrote {len(prompts)} prompts to {path}")
    c = Counter((p["sub_category"], p["difficulty"]) for p in prompts)
    for (sub, diff), n in sorted(c.items()):
        print(f"  {sub:25s} {diff:8s}: {n}")
    print()
    _print_numeracy_distribution(prompts)


if __name__ == "__main__":
    main()
